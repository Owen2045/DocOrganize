import re
import time
from pathlib import Path
from elasticsearch import Elasticsearch
from pypdf import PdfReader
from app.config import ES_URL, ES_INDEX, DOCS_DIR
from app.retriever import get_embed_model

# 比對「第 N 條」或「第 N-M 條」格式，用於切塊分隔符
ARTICLE_PATTERN = re.compile(r"(第\s*\d+(?:-\d+)?\s*條)")

# chunk_generic 的句子邊界：切在標點/換行「之後」，讓標點留在前一句尾端
_SENTENCE_END = re.compile(r"(?<=[。！？\n])")

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


def read_file(path: Path) -> str:
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


def _break_into_units(text: str, size: int) -> list[str]:
    """把文字切成不超過 size 的最小語意單元：優先照段落切，太長的段落改照句子切，
    連句子都超過 size（例如整段沒有標點的流水帳文字）才退回固定字數硬切。"""
    units = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            units.append(para)
            continue
        for sent in _SENTENCE_END.split(para):
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) <= size:
                units.append(sent)
            else:
                units.extend(sent[i : i + size] for i in range(0, len(sent), size))
    return units


def chunk_generic(text: str, source: str, size: int = 600, overlap: int = 100) -> list[dict]:
    """無條文結構的文件（如比較用的一般 PDF）：優先照段落/句子邊界切，不切斷語意單元；
    把小單元貪心地塞進接近 size 的 chunk，塊與塊之間保留 overlap 字元維持前後文連貫。"""
    chunks, buf = [], ""
    for unit in _break_into_units(text, size):
        if buf and len(buf) + len(unit) > size:
            chunks.append(buf)
            buf = (buf[-overlap:] if overlap else "") + unit
        else:
            buf += unit
    if buf:
        chunks.append(buf)
    return [
        {"article": f"段落{idx + 1}", "content": c.strip(), "source": source}
        for idx, c in enumerate(chunks)
        if c.strip()
    ]


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
    text = read_file(path)
    chunks = chunk_by_article(text, path.stem)
    if not chunks:
        # 找不到「第N條」結構（如比較用的一般文件），退回固定長度切塊
        chunks = chunk_generic(text, path.stem)
    if not chunks:
        print(f"[ingest] {path.name}: empty file", flush=True)
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


_ingesting: set[str] = set()


def mark_ingesting(source: str) -> None:
    _ingesting.add(source)


def mark_done(source: str) -> None:
    _ingesting.discard(source)


def is_ingesting(source: str) -> bool:
    return source in _ingesting


def sweep_orphaned_docs(max_age_hours: int = 48) -> int:
    """啟動時清掉 docs/ 裡超過 max_age_hours、且從未被索引進 ES 的孤兒檔案
    （例如 compare_documents 讀過但沒被 ingest_documents 歸檔的暫存檔），避免磁碟無限成長。"""
    if not DOCS_DIR.exists():
        return 0
    indexed = {d["source"] for d in list_indexed_docs()}
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for p in DOCS_DIR.iterdir():
        if p.is_file() and p.stem not in indexed and p.stat().st_mtime < cutoff:
            p.unlink()
            removed += 1
    return removed
