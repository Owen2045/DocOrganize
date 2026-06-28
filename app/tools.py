from llama_index.core.tools import FunctionTool
from app.retriever import search

def _search_knowledge_base(query: str) -> str:
    """搜尋金融法規知識庫，回傳相關條文內容與條號。"""
    results = search(query)
    if not results:
        return "查無相關法規條文。"
    parts = [f"【{r['article']}】{r['content']}" for r in results]
    return "\n\n".join(parts)

search_tool = FunctionTool.from_defaults(fn=_search_knowledge_base)
