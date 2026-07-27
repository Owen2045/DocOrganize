"""
一次性腳本：把 docs/ 底下尚未索引的法規文件透過 app.ingest.ingest_file() 建檔索引。
用法：python scripts/ingest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from elasticsearch import Elasticsearch
from app.config import ES_URL, ES_INDEX
from app.ingest import ingest_file

DOCS_DIR = Path("docs")


def _already_indexed(es: Elasticsearch, source: str) -> bool:
    if not es.indices.exists(index=ES_INDEX):
        return False
    r = es.count(index=ES_INDEX, body={"query": {"term": {"source": source}}})
    return r["count"] > 0


def main():
    es = Elasticsearch(ES_URL)
    for path in [*DOCS_DIR.glob("*.pdf"), *DOCS_DIR.glob("*.md"), *DOCS_DIR.glob("*.txt")]:
        if _already_indexed(es, path.stem):
            print(f"  {path.name}: already indexed, skipping")
            continue
        ingest_file(path)


if __name__ == "__main__":
    main()
