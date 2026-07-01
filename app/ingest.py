import re
from pathlib import Path
from elasticsearch import Elasticsearch
from pypdf import PdfReader
from app.config import ES_URL, ES_INDEX
from app.retriever import get_embed_model

# 比對「第 N 條」或「第 N-M 條」格式，用於切塊分隔符
ARTICLE_PATTERN = re.compile(r"(第\s*\d+(?:-\d+)?\s*條)")

# ES index mapping：text 欄位走 BM25，embedding 走 kNN cosine，category 必須是 keyword 才能做 terms aggregation
INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "content": {"type": "text", "analyzer": "standard"},
            "article": {"type": "keyword"},
            "source": {"type": "keyword"},
            "category": {"type": "keyword"},
            "embedding": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "cosine",
            },
        }
    }
}


def _read_file(path: Path) -> str:
    """讀取檔案內容：PDF 逐頁取文字，其餘直接讀 UTF-8。"""
    if path.suffix == ".pdf":
        return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")


def chunk_by_article(text: str, source: str) -> list[dict]:
    """依「第N條」regex 切塊，每塊含條號、內文、來源檔名。"""
    # split 後奇數位是條號、偶數位是條文內容，成對處理
    parts = ARTICLE_PATTERN.split(text)
    chunks, i = [], 1
    while i < len(parts) - 1:
        article = parts[i].strip()
        content = parts[i + 1].strip()
        if content:
            chunks.append(
                {
                    "article": article,
                    "content": f"{article} {content}",
                    "source": source,
                }
            )
        i += 2
    return chunks


def list_indexed_docs() -> list[dict]:
    """列出 ES 內所有已索引文件：用 terms aggregation 取 source，並 sub-agg 取各 source 的 category。"""
    es = Elasticsearch(ES_URL)
    if not es.indices.exists(index=ES_INDEX):
        return []
    r = es.search(
        index=ES_INDEX,
        body={
            "size": 0,
            "aggs": {
                "sources": {
                    "terms": {"field": "source", "size": 100},
                    # 每個 source 取第一個 category（同 source 的 chunks 分類應相同）
                    "aggs": {"category": {"terms": {"field": "category", "size": 1}}},
                }
            },
        },
    )
    result = []
    for b in r["aggregations"]["sources"]["buckets"]:
        cats = b["category"]["buckets"]
        result.append(
            {
                "source": b["key"],
                "chunks": b["doc_count"],
                "category": cats[0]["key"] if cats else "—",
            }
        )
    return result


def list_categories() -> list[str]:
    """列出 ES 內所有現有分類值，供前端 datalist 顯示。"""
    es = Elasticsearch(ES_URL)
    if not es.indices.exists(index=ES_INDEX):
        return []
    r = es.search(
        index=ES_INDEX,
        body={
            "size": 0,
            "aggs": {"cats": {"terms": {"field": "category", "size": 50}}},
        },
    )
    return [b["key"] for b in r["aggregations"]["cats"]["buckets"]]


def delete_source(source: str) -> int:
    """刪除 ES 內指定 source 的所有 chunks，回傳刪除筆數。"""
    es = Elasticsearch(ES_URL)
    if not es.indices.exists(index=ES_INDEX):
        return 0
    r = es.delete_by_query(index=ES_INDEX, body={"query": {"term": {"source": source}}})
    return r.get("deleted", 0)


def ingest_file(path: Path, category: str = "") -> int:
    """讀取檔案 → 切塊 → embed → 寫入 ES；回傳成功寫入的 chunk 數。"""
    print(f"[ingest] start {path.name} category={category!r}", flush=True)
    es = Elasticsearch(ES_URL)
    # index 不存在時自動建立（含 mapping）
    if not es.indices.exists(index=ES_INDEX):
        es.indices.create(index=ES_INDEX, body=INDEX_MAPPING)
    chunks = chunk_by_article(_read_file(path), path.stem)
    if not chunks:
        print(f"[ingest] {path.name}: no articles found", flush=True)
        return 0
    embeddings = get_embed_model().encode([c["content"] for c in chunks])
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
        # 空字串不寫 category 欄位，保持與 CLI ingest 舊文件的一致性
        if category:
            chunk["category"] = category
        es.index(index=ES_INDEX, document=chunk)
    # refresh 確保此次 ingest 立即可被搜尋到
    es.indices.refresh(index=ES_INDEX)
    print(f"[ingest] {path.name}: {len(chunks)} chunks done", flush=True)
    return len(chunks)
