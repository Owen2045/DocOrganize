"""
一次性腳本：把 docs/ 底下的法規 markdown 切塊、embedding、寫入 ES。
用法：python scripts/ingest.py
"""
import os, re, json
from pathlib import Path
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "citefund_laws")
DOCS_DIR = Path("docs")

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "content":   {"type": "text",       "analyzer": "ik_max_word"},
            "article":   {"type": "keyword"},
            "source":    {"type": "keyword"},
            "embedding": {"type": "dense_vector", "dims": 1024, "index": True, "similarity": "cosine"},
        }
    }
}

def chunk_by_article(text: str, source: str) -> list[dict]:
    """依「第X條」切塊，保留條號。"""
    parts = re.split(r"(第\s*\d+\s*條)", text)
    chunks = []
    i = 1
    while i < len(parts) - 1:
        article = parts[i].strip()
        content = parts[i + 1].strip()
        if content:
            chunks.append({"article": article, "content": f"{article} {content}", "source": source})
        i += 2
    return chunks

def main():
    es = Elasticsearch(ES_URL)
    model = SentenceTransformer("BAAI/bge-m3")

    if es.indices.exists(index=ES_INDEX):
        es.indices.delete(index=ES_INDEX)
    es.indices.create(index=ES_INDEX, body=INDEX_MAPPING)
    print(f"Index {ES_INDEX} created.")

    for md_file in DOCS_DIR.glob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        chunks = chunk_by_article(text, md_file.stem)
        if not chunks:
            print(f"  {md_file.name}: no articles found, skipping")
            continue

        embeddings = model.encode([c["content"] for c in chunks], show_progress_bar=True)
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb.tolist()
            es.index(index=ES_INDEX, document=chunk)

        print(f"  {md_file.name}: {len(chunks)} chunks indexed")

    es.indices.refresh(index=ES_INDEX)
    print("Done.")

if __name__ == "__main__":
    main()
