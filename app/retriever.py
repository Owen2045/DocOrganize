import time
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from app.config import ES_URL, ES_INDEX

_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("BAAI/bge-m3")
    return _embed_model

def _rrf_merge(lists: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranked in lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.__getitem__, reverse=True)

def search(query: str, top_k: int = 5) -> list[dict]:
    es = Elasticsearch(ES_URL)

    t0 = time.time()
    vector = get_embed_model().encode(query).tolist()
    print(f"[retriever] embed: {time.time()-t0:.2f}s", flush=True)

    fetch = top_k * 3
    t1 = time.time()
    bm25_resp = es.search(index=ES_INDEX, body={
        "size": fetch,
        "query": {"match": {"content": {"query": query}}}
    })
    knn_resp = es.search(index=ES_INDEX, body={
        "size": fetch,
        "knn": {"field": "embedding", "query_vector": vector, "k": fetch, "num_candidates": 100}
    })
    print(f"[retriever] es search: {time.time()-t1:.2f}s", flush=True)

    docs = {h["_id"]: h["_source"] for h in bm25_resp["hits"]["hits"] + knn_resp["hits"]["hits"]}
    bm25_ids = [h["_id"] for h in bm25_resp["hits"]["hits"]]
    knn_ids  = [h["_id"] for h in knn_resp["hits"]["hits"]]
    merged_ids = _rrf_merge([bm25_ids, knn_ids])

    hits = [
        {"content": docs[i]["content"], "article": docs[i].get("article", "")}
        for i in merged_ids if i in docs
    ]

    # ponytail: reranker skipped — bge-reranker-v2-m3 takes 60s on CPU; RRF already good enough
    return hits[:top_k]
