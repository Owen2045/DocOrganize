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


def search_fusion(
    query: str, top_k: int = 5, source: str | None = None, category: str | None = None
) -> list[dict]:
    """Multi-Query RAG-Fusion + HyDE：GPT 改寫 3 個 query，各自生成假設條文，
    批次 embed 假設條文做 kNN、原始改寫 query 做 BM25，各自 RRF 後再做一次跨 query RRF，回傳 top_k。

    source 給定時只搜尋該份文件（比較兩份文件時用來分別鎖定各自範圍）；
    category 給定時只搜尋該分類（知識庫存有多種文件類型時用來縮小範圍）；兩者可併用。"""
    es = Elasticsearch(ES_URL)

    t0 = time.time()
    rewrites = _rewrite_queries(query)
    print(f"[retriever] rewrites: {rewrites}", flush=True)
    # HyDE：每個改寫 query 各生成一段假設條文，用其 embedding 代替原始 query 做 kNN
    hydes = [_hyde_query(r) for r in rewrites]
    # 批次 encode 所有假設條文，比逐一呼叫快約 2-3 倍
    vectors = get_embed_model().encode(hydes)
    print(f"[retriever] rewrite+hyde+embed: {time.time()-t0:.2f}s", flush=True)

    fetch = top_k * 3
    all_ranked: list[list[str]] = []
    docs: dict = {}

    t1 = time.time()
    filters = []
    if source:
        filters.append({"term": {"source": source}})
    if category:
        filters.append({"term": {"category": category}})
    for rewrite, vector in zip(rewrites, vectors):
        bm25_resp = es.search(
            index=ES_INDEX,
            body={
                "size": fetch,
                "query": {"bool": {"must": {"match": {"content": {"query": rewrite}}}, "filter": filters}},
            },
        )
        knn_query = {"field": "embedding", "query_vector": vector.tolist(), "k": fetch, "num_candidates": 100}
        if filters:
            knn_query["filter"] = {"bool": {"filter": filters}}
        knn_resp = es.search(index=ES_INDEX, body={"size": fetch, "knn": knn_query})
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
