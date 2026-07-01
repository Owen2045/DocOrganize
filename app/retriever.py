import time
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from openai import OpenAI as _OAI
from app.config import ES_URL, ES_INDEX, OPENAI_API_KEY

# 模組層級單例，避免每次查詢重新載入模型（BGE-M3 約需 10 秒）
_embed_model = None
_oai: _OAI | None = None


def _get_oai() -> _OAI:
    """lazy init OpenAI client 單例。"""
    global _oai
    if _oai is None:
        _oai = _OAI(api_key=OPENAI_API_KEY)
    return _oai


def _hyde_query(query: str) -> str:
    """HyDE：讓 LLM 生成一段假設性條文，用其 embedding 代替原始 query 做 kNN，提升語意匹配精度。"""
    try:
        r = _get_oai().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"請用台灣金融法規的條文語氣，寫一段回答以下問題的法規內容（50字內），不需要條號：{query}",
                }
            ],
            max_tokens=200,
        )
        return r.choices[0].message.content or query
    except Exception:
        # ponytail: fallback if OpenAI quota/error
        return query


def get_embed_model():
    """lazy init BGE-M3 embedding model 單例；lifespan 預載後後續呼叫直接回傳快取。"""
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("BAAI/bge-m3")
    return _embed_model


def _rrf_merge(lists: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion：將多個排序列表合併，k=60 為標準超參數。"""
    scores: dict[str, float] = {}
    for ranked in lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.__getitem__, reverse=True)


def _rewrite_queries(query: str) -> list[str]:
    """用 GPT 將原始問題改寫成 3 個不同角度的查詢，擴大 ES 召回範圍。"""
    try:
        r = _get_oai().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content":
                f"請將以下問題改寫成 3 個不同角度的查詢字串，用於搜尋台灣金融法規知識庫。"
                f"每行一個，不要編號，不要額外說明：\n{query}"}],
            max_tokens=200,
        )
        lines = (r.choices[0].message.content or "").strip().splitlines()
        rewrites = [line.strip() for line in lines if line.strip()][:3]
        return rewrites if rewrites else [query]
    except Exception:
        return [query]


def search_fusion(query: str, top_k: int = 5) -> list[dict]:
    """Multi-Query RAG-Fusion：GPT 改寫 3 個 query，批次 embed，各自 BM25+kNN，最終 RRF 合併回傳 top_k。"""
    es = Elasticsearch(ES_URL)

    t0 = time.time()
    rewrites = _rewrite_queries(query)
    print(f"[retriever] rewrites: {rewrites}", flush=True)
    # 批次 encode 所有改寫 query，比逐一呼叫快約 2-3 倍
    vectors = get_embed_model().encode(rewrites)
    print(f"[retriever] rewrite+embed: {time.time()-t0:.2f}s", flush=True)

    fetch = top_k * 3
    all_ranked: list[list[str]] = []
    docs: dict = {}

    t1 = time.time()
    for rewrite, vector in zip(rewrites, vectors):
        bm25_resp = es.search(
            index=ES_INDEX,
            body={"size": fetch, "query": {"match": {"content": {"query": rewrite}}}},
        )
        knn_resp = es.search(
            index=ES_INDEX,
            body={
                "size": fetch,
                "knn": {"field": "embedding", "query_vector": vector.tolist(), "k": fetch, "num_candidates": 100},
            },
        )
        docs.update({h["_id"]: h["_source"] for h in bm25_resp["hits"]["hits"] + knn_resp["hits"]["hits"]})
        bm25_ids = [h["_id"] for h in bm25_resp["hits"]["hits"]]
        knn_ids = [h["_id"] for h in knn_resp["hits"]["hits"]]
        # 每個 query 的 BM25+kNN 先做一次 RRF，再集合所有 query 做最終 RRF
        all_ranked.append(_rrf_merge([bm25_ids, knn_ids]))
    print(f"[retriever] fusion es search: {time.time()-t1:.2f}s", flush=True)

    merged_ids = _rrf_merge(all_ranked)
    hits = [
        {"content": docs[i]["content"], "article": docs[i].get("article", "")}
        for i in merged_ids
        if i in docs
    ]
    # ponytail: reranker skipped — RRF already good enough
    return hits[:top_k]


def search(query: str, top_k: int = 5) -> list[dict]:
    """混合檢索：HyDE embed 做 kNN + 原始 query 做 BM25，兩路結果 RRF 合併後回傳 top_k chunks。"""
    es = Elasticsearch(ES_URL)

    t0 = time.time()
    hyde = _hyde_query(query)
    print(f"[retriever] hyde: {hyde[:60]!r}", flush=True)
    # kNN 用假設條文的 embedding，BM25 仍用原始 query，兩者互補
    vector = get_embed_model().encode(hyde).tolist()
    print(f"[retriever] hyde+embed: {time.time()-t0:.2f}s", flush=True)

    # 各路多取 top_k*3，RRF 後再截 top_k，避免截斷有潛力的結果
    fetch = top_k * 3
    t1 = time.time()
    bm25_resp = es.search(
        index=ES_INDEX,
        body={"size": fetch, "query": {"match": {"content": {"query": query}}}},
    )
    knn_resp = es.search(
        index=ES_INDEX,
        body={
            "size": fetch,
            "knn": {
                "field": "embedding",
                "query_vector": vector,
                "k": fetch,
                "num_candidates": 100,
            },
        },
    )
    print(f"[retriever] es search: {time.time()-t1:.2f}s", flush=True)

    docs = {
        h["_id"]: h["_source"]
        for h in bm25_resp["hits"]["hits"] + knn_resp["hits"]["hits"]
    }
    bm25_ids = [h["_id"] for h in bm25_resp["hits"]["hits"]]
    knn_ids = [h["_id"] for h in knn_resp["hits"]["hits"]]
    merged_ids = _rrf_merge([bm25_ids, knn_ids])

    hits = [
        {"content": docs[i]["content"], "article": docs[i].get("article", "")}
        for i in merged_ids
        if i in docs
    ]

    # ponytail: reranker skipped — bge-reranker-v2-m3 takes 60s on CPU; RRF already good enough
    return hits[:top_k]
