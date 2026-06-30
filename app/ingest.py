import re
from pathlib import Path
from elasticsearch import Elasticsearch
from pypdf import PdfReader
from app.config import ES_URL, ES_INDEX
from app.retriever import get_embed_model

ARTICLE_PATTERN = re.compile(r"(第\s*\d+(?:-\d+)?\s*條)")

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "content":   {"type": "text", "analyzer": "standard"},
            "article":   {"type": "keyword"},
            "source":    {"type": "keyword"},
            "category":  {"type": "keyword"},
            "embedding": {"type": "dense_vector", "dims": 1024, "index": True, "similarity": "cosine"},
        }
    }
}

def _read_file(path: Path) -> str:
    if path.suffix == ".pdf":
        return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")

def chunk_by_article(text: str, source: str) -> list[dict]:
    parts = ARTICLE_PATTERN.split(text)
    chunks, i = [], 1
    while i < len(parts) - 1:
        article = parts[i].strip()
        content = parts[i + 1].strip()
        if content:
            chunks.append({"article": article, "content": f"{article} {content}", "source": source})
        i += 2
    return chunks

def list_indexed_docs() -> list[dict]:
    es = Elasticsearch(ES_URL)
    if not es.indices.exists(index=ES_INDEX):
        return []
    r = es.search(index=ES_INDEX, body={
        "size": 0,
        "aggs": {"sources": {"terms": {"field": "source", "size": 100},
            "aggs": {"category": {"terms": {"field": "category", "size": 1}}}
        }}
    })
    result = []
    for b in r["aggregations"]["sources"]["buckets"]:
        cats = b["category"]["buckets"]
        result.append({
            "source": b["key"],
            "chunks": b["doc_count"],
            "category": cats[0]["key"] if cats else "—",
        })
    return result

def list_categories() -> list[str]:
    es = Elasticsearch(ES_URL)
    if not es.indices.exists(index=ES_INDEX):
        return []
    r = es.search(index=ES_INDEX, body={
        "size": 0,
        "aggs": {"cats": {"terms": {"field": "category", "size": 50}}}
    })
    return [b["key"] for b in r["aggregations"]["cats"]["buckets"]]

def delete_source(source: str) -> int:
    es = Elasticsearch(ES_URL)
    if not es.indices.exists(index=ES_INDEX):
        return 0
    r = es.delete_by_query(index=ES_INDEX, body={"query": {"term": {"source": source}}})
    return r.get("deleted", 0)

def ingest_file(path: Path, category: str = "") -> int:
    print(f"[ingest] start {path.name} category={category!r}", flush=True)
    es = Elasticsearch(ES_URL)
    if not es.indices.exists(index=ES_INDEX):
        es.indices.create(index=ES_INDEX, body=INDEX_MAPPING)
    chunks = chunk_by_article(_read_file(path), path.stem)
    if not chunks:
        print(f"[ingest] {path.name}: no articles found", flush=True)
        return 0
    embeddings = get_embed_model().encode([c["content"] for c in chunks])
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
        if category:
            chunk["category"] = category
        es.index(index=ES_INDEX, document=chunk)
    es.indices.refresh(index=ES_INDEX)
    print(f"[ingest] {path.name}: {len(chunks)} chunks done", flush=True)
    return len(chunks)
