"""
一次性腳本：把 docs/ 底下的法規文件切塊、embedding、寫入 ES。
支援格式：.pdf / .md / .txt
用法：python scripts/ingest.py
"""
import os, re
from pathlib import Path
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "citefund_laws")
DOCS_DIR = Path("docs")

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "content":   {"type": "text",       "analyzer": "standard"},
            "article":   {"type": "keyword"},
            "source":    {"type": "keyword"},
            "category":  {"type": "keyword"},
            "embedding": {"type": "dense_vector", "dims": 1024, "index": True, "similarity": "cosine"},
        }
    }
}

# 支援「第 2-1 條」、「第 27-3 條」等複合條號
ARTICLE_PATTERN = re.compile(r"(第\s*\d+(?:-\d+)?\s*條)")

def read_pdf(path: Path) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def chunk_by_article(text: str, source: str) -> list[dict]:
    parts = ARTICLE_PATTERN.split(text)
    chunks = []
    i = 1
    while i < len(parts) - 1:
        article = parts[i].strip()
        content = parts[i + 1].strip()
        if content:
            chunks.append({"article": article, "content": f"{article} {content}", "source": source})
        i += 2
    return chunks

def iter_docs():
    for path in [*DOCS_DIR.glob("*.pdf"), *DOCS_DIR.glob("*.md"), *DOCS_DIR.glob("*.txt")]:
        if path.suffix == ".pdf":
            text = read_pdf(path)
        else:
            text = path.read_text(encoding="utf-8")
        yield path, text

def _already_indexed(es: Elasticsearch, source: str) -> bool:
    r = es.count(index=ES_INDEX, body={"query": {"term": {"source": source}}})
    return r["count"] > 0

def main():
    es = Elasticsearch(ES_URL)
    model = SentenceTransformer("BAAI/bge-m3")

    if not es.indices.exists(index=ES_INDEX):
        es.indices.create(index=ES_INDEX, body=INDEX_MAPPING)
        print(f"Index {ES_INDEX} created.")

    for path, text in iter_docs():
        if _already_indexed(es, path.stem):
            print(f"  {path.name}: already indexed, skipping")
            continue

        chunks = chunk_by_article(text, path.stem)
        if not chunks:
            print(f"  {path.name}: no articles found, skipping")
            continue

        embeddings = model.encode([c["content"] for c in chunks], show_progress_bar=True)
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb.tolist()
            es.index(index=ES_INDEX, document=chunk)

        print(f"  {path.name}: {len(chunks)} chunks indexed")

    es.indices.refresh(index=ES_INDEX)
    print("Done.")

if __name__ == "__main__":
    main()
