import json
from pathlib import Path
from app.llm_clients import get_llm_clients
from app.retriever import search_fusion
from app.ingest import (
    list_indexed_docs,
    ingest_file,
    delete_source,
    read_file,
    chunk_generic,
    mark_ingesting,
    mark_done,
)
from app.pending_files import get_pending, remove_pending
from app.judge import evaluate_action

# 超過此字數才觸發 compare_documents 的分段摘要降級
COMPARE_DIRECT_LIMIT = 40_000

_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "搜尋知識庫，回傳相關條文/段落內容。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜尋查詢字串"},
                "source": {
                    "type": "string",
                    "description": "選填，只搜尋指定文件（用 list_documents 取得的文件名稱）；不填則搜尋整個知識庫，用於比較兩份文件時分別鎖定範圍。",
                },
                "category": {
                    "type": "string",
                    "description": "選填，只搜尋指定分類（用 list_documents 取得的分類名稱），知識庫存有多種文件類型時用來縮小搜尋範圍；與 source 可擇一使用。",
                },
            },
            "required": ["query"],
        },
    },
}

_LIST_DOCS_TOOL = {
    "type": "function",
    "function": {
        "name": "list_documents",
        "description": "列出知識庫中所有已索引的文件名稱，比較兩份文件前用來確認正確名稱。",
        "parameters": {"type": "object", "properties": {}},
    },
}

_INGEST_TOOL = {
    "type": "function",
    "function": {
        "name": "ingest_documents",
        "description": "將暫存（已上傳但尚未歸檔）的文件寫入知識庫。使用者明確表達歸檔意圖時才呼叫，執行前會經過系統審核，可能被拒絕。",
        "parameters": {
            "type": "object",
            "properties": {
                "filenames": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要歸檔的暫存檔名清單；留空表示歸檔使用者最新一次上傳批次的所有檔案（不含更早之前還沒處理的舊檔案）。",
                },
                "category": {"type": "string", "description": "選填，這批文件的分類"},
            },
        },
    },
}

_COMPARE_TOOL = {
    "type": "function",
    "function": {
        "name": "compare_documents",
        "description": "唯讀讀取兩份以上暫存（尚未歸檔）的文件內容供比較，不會寫入知識庫。",
        "parameters": {
            "type": "object",
            "properties": {
                "filenames": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "description": "要比較的暫存檔名清單，至少 2 個",
                },
                "focus": {"type": "string", "description": "選填，比較時想特別關注的重點"},
            },
            "required": ["filenames"],
        },
    },
}


def _run_search(query: str, source: str | None = None, category: str | None = None) -> str:
    """查知識庫"""
    results = search_fusion(query, source=source, category=category)
    if not results:
        if source:
            return f"文件「{source}」查無相關內容。"
        if category:
            return f"分類「{category}」查無相關內容。"
        return "查無相關內容。"
    return "\n\n".join(f"【{r['article']}】{r['content']}" for r in results)


def _condense_large_doc(text: str, source: str) -> str:
    """map 步驟：分段各自摘要，壓縮成精簡版供 agent 自己做比較（reduce 交給主 agent，不在這裡比較）。"""
    client, model = get_llm_clients()[0]
    chunks = chunk_generic(text, source, size=3000, overlap=200)[:15]
    summaries = []
    for c in chunks:
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"用 3 句話內摘要以下內容重點：\n{c['content']}"}],
                max_tokens=200,
            )
            summaries.append(r.choices[0].message.content or "")
        except Exception:
            summaries.append(c["content"][:200])
    return "\n".join(summaries)


def _run_compare(args: dict, session_id: str) -> str:
    """讀取暫存檔案內容並回傳，實際比較交由主 agent 在下一輪回答時完成（跟 search_knowledge_base 同一種模式）。"""
    filenames = args.get("filenames") or []
    pending = {p.filename: p for p in get_pending(session_id)}
    parts, missing = [], []
    for fname in filenames:
        p = pending.get(fname)
        if not p:
            missing.append(fname)
            continue
        text = read_file(Path(p.path))
        truncated = len(text) > COMPARE_DIRECT_LIMIT
        if truncated:
            text = _condense_large_doc(text, fname)
        note = "（僅分析前段內容，原文較長已截斷）" if truncated else ""
        parts.append(f"【{fname}】{note}\n{text}")
    if missing:
        parts.append(f"（找不到暫存檔案：{'、'.join(missing)}，可能已被歸檔或移除）")
    return "\n\n---\n\n".join(parts) if parts else "找不到要比較的暫存檔案。"


def _run_ingest(args: dict, session_id: str, user_message: str, model: str) -> dict:
    """將暫存檔案寫入知識庫，回傳執行結果。"""
    pending = get_pending(session_id)
    if args.get("filenames"):
        requested = args["filenames"]
    else:
        # 沒指名時只歸檔「最新一批」上傳的檔案（pending 是依上傳順序 append 的，
        # 最後一筆的 batch_id 就是最新批次），避免誤觸更早留在 pending 裡還沒處理的舊檔案
        latest_batch = pending[-1].batch_id if pending else None
        requested = [p.filename for p in pending if p.batch_id == latest_batch]
    matched = [p for p in pending if p.filename in requested]
    if not matched:
        return {"executed": False, "reason": "找不到符合的暫存檔案，請確認檔名或重新上傳。"}

    category = args.get("category") or matched[0].category or ""
    proposed_action = {
        "tool": "ingest_documents",
        "filenames": [p.filename for p in matched],
        "category": category,
    }
    # 明確傳入 exclude_model（目前正在使用的主模型），不依賴 client 清單順序的隱含假設
    gate = evaluate_action(user_message, proposed_action, exclude_model=model)
    if not gate["approved"]:
        return {"executed": False, "reason": gate["reason"]}

    ingested, total_chunks = [], 0
    for p in matched:
        path = Path(p.path)
        mark_ingesting(path.stem)
        try:
            delete_source(path.stem)  # 同名已索引則先刪除，等於 upsert；不存在時為 no-op
            total_chunks += ingest_file(path, category or p.category)
            ingested.append(p.filename)
            path.unlink(missing_ok=True)  # 內容已寫入 ES，Mac 磁碟上的原始檔不再需要
        finally:
            mark_done(path.stem)
    remove_pending(session_id, ingested)
    return {"executed": True, "reason": gate["reason"], "ingested": ingested, "chunks": total_chunks}


def _wrap_search(args: dict, session_id: str, user_message: str, model: str) -> str:
    return _run_search(args.get("query", ""), args.get("source"), args.get("category"))


def _wrap_list_documents(args: dict, session_id: str, user_message: str, model: str) -> str:
    return json.dumps(list_indexed_docs(), ensure_ascii=False)


def _wrap_ingest(args: dict, session_id: str, user_message: str, model: str) -> str:
    return json.dumps(_run_ingest(args, session_id, user_message, model), ensure_ascii=False)


def _wrap_compare(args: dict, session_id: str, user_message: str, model: str) -> str:
    return _run_compare(args, session_id)


# 單一真相來源：工具名稱 → schema、實際執行的 runner、結果是否算檢索內容（要餵給 Answer Gate 覆核）。
# 新增/修改工具只需要改這裡一個地方，不用再到處找散落的 if/elif。
_REGISTRY = {
    "search_knowledge_base": {"schema": _SEARCH_TOOL, "run": _wrap_search, "contributes_context": True},
    "list_documents": {"schema": _LIST_DOCS_TOOL, "run": _wrap_list_documents, "contributes_context": False},
    "ingest_documents": {"schema": _INGEST_TOOL, "run": _wrap_ingest, "contributes_context": False},
    "compare_documents": {"schema": _COMPARE_TOOL, "run": _wrap_compare, "contributes_context": False},
}

TOOLS = [entry["schema"] for entry in _REGISTRY.values()]


def run_tool(name: str, args: dict, session_id: str, user_message: str, model: str) -> tuple[str, bool]:
    """依名稱查表執行工具，回傳 (結果字串, 是否算檢索內容)。

    查無此工具名稱時回傳錯誤訊息當作工具結果交回給模型，讓它在同一輪迴圈裡自己重試或改用別的工具，
    不會讓一次可恢復的模型判斷失誤中斷整個對話。"""
    entry = _REGISTRY.get(name)
    if entry is None:
        return f"找不到工具「{name}」，請確認工具名稱是否正確。", False
    return entry["run"](args, session_id, user_message, model), entry["contributes_context"]
