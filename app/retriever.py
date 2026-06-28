from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, CrossEncoder
from app.config import ES_URL, ES_INDEX

_embed_model = None
_reranker = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("BAAI/bge-m3")
    return _embed_model

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return _reranker

def search(query: str, top_k: int = 5) -> list[dict]:
    es = Elasticsearch(ES_URL)
    vector = get_embed_model().encode(query).tolist()

    resp = es.search(index=ES_INDEX, body={
        "size": top_k * 2,  # 多撈一些給 reranker 篩
        "retriever": {
            "rrf": {
                "retrievers": [
                    {
                        "standard": {
                            "query": {
                                "match": {"content": {"query": query}}
                            }
                        }
                    },
                    {
                        "knn": {
                            "field": "embedding",
                            "query_vector": vector,
                            "num_candidates": 50
                        }
                    }
                ]
            }
        }
    })

    hits = [
        {"content": h["_source"]["content"], "article": h["_source"].get("article", ""), "score": h["_score"]}
        for h in resp["hits"]["hits"]
    ]

    # rerank
    reranker = get_reranker()
    pairs = [[query, h["content"]] for h in hits]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, hits), reverse=True)
    return [h for _, h in ranked[:top_k]]
