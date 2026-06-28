"""
快速測試檢索結果是否合理。
用法：python scripts/test_retrieval.py
"""
from app.retriever import search

QUERIES = [
    "基金經理人的資格條件是什麼",
    "基金淨資產價值如何計算",
    "境外基金引進台灣需要哪些條件",
    "投資人贖回基金的程序",
    "基金違規時主管機關可以採取什麼措施",
]

for q in QUERIES:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    results = search(q, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['article']} — {r['content'][:80]}...")
