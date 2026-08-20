"""
run_chat_thread 的檢查：streaming token 拼接、provider rate-limit fallback。
clients 全部注入假物件，不連真的 OpenAI/Gemini。
用法：python scripts/test_agent.py
"""
import sys
import json as json_mod
import queue as queue_mod
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from openai import RateLimitError
import app.agent as agent
import app.tools as tools
from app.agent import run_chat_thread
from app.pending_files import PendingFile


class ImmediateLoop:
    """call_soon_threadsafe 直接同步執行，測試不需要真的跑 event loop。"""

    def call_soon_threadsafe(self, fn, *args):
        fn(*args)


def _chunk(content=None, finish_reason=None):
    delta = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def _fake_client(chunks=None, raises=None):
    def create(**kwargs):
        if raises:
            raise raises
        return iter(chunks)

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def _fake_rate_limit():
    req = httpx.Request("POST", "https://example.test")
    resp = httpx.Response(429, request=req)
    return RateLimitError("rate limited", response=resp, body=None)


def _tool_call_chunk(name, arguments, id_="call_1", index=0):
    tc = SimpleNamespace(index=index, id=id_, function=SimpleNamespace(name=name, arguments=arguments))
    delta = SimpleNamespace(content=None, tool_calls=[tc])
    choice = SimpleNamespace(delta=delta, finish_reason=None)
    return SimpleNamespace(choices=[choice])


def _sequenced_client(responses):
    """依序回傳每次 create() 呼叫要吐出的 chunk 序列，模擬多輪 stream（例如 tool_calls → 最終答案 → 修正）。"""
    calls = {"n": 0}

    def create(**kwargs):
        chunks = responses[calls["n"]]
        calls["n"] += 1
        return iter(chunks)

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _patch(module, **attrs):
    """暫時覆寫 module 屬性，回傳還原函式；每個 monkeypatch 都要 try/finally 呼叫還原。"""
    original = {k: getattr(module, k) for k in attrs}
    for k, v in attrs.items():
        setattr(module, k, v)

    def restore():
        for k, v in original.items():
            setattr(module, k, v)

    return restore


def _run(clients, session_id="test-session"):
    q = queue_mod.Queue()
    thread_exc = []
    run_chat_thread(
        [{"role": "user", "content": "hi"}], q, ImmediateLoop(), thread_exc, session_id, clients=clients
    )
    tokens = []
    while not q.empty():
        tokens.append(q.get_nowait())
    return tokens, thread_exc


# 情境 1：正常回答，無 tool call
client = _fake_client(
    chunks=[_chunk(content="你"), _chunk(content="好"), _chunk(finish_reason="stop")]
)
tokens, thread_exc = _run([(client, "test-model")])
assert tokens == ["你", "好", None]
assert thread_exc == []

# 情境 3：第一個 provider rate-limit，第二個成功 fallback
bad_client = _fake_client(raises=_fake_rate_limit())
good_client = _fake_client(chunks=[_chunk(content="ok"), _chunk(finish_reason="stop")])
tokens, thread_exc = _run([(bad_client, "model-a"), (good_client, "model-b")])
assert tokens == ["ok", None]
assert thread_exc == []

# 情境 4：全部 provider 都 rate-limit
tokens, thread_exc = _run(
    [
        (_fake_client(raises=_fake_rate_limit()), "model-a"),
        (_fake_client(raises=_fake_rate_limit()), "model-b"),
    ]
)
assert tokens == [None]
assert len(thread_exc) == 1
assert isinstance(thread_exc[0], RateLimitError)

# 情境 5：_run_ingest 核准路徑 —— 真的寫入且 pending 被清除
# _run_ingest 移到 app/tools.py 後（見架構審查候選 #1 的工具註冊表重構），要патch的 module 也跟著換成 tools
ingest_calls, remove_calls = [], []
restore = _patch(
    tools,
    get_pending=lambda session_id: [PendingFile("a.pdf", "/tmp/a.pdf", 10, "")],
    evaluate_action=lambda user_message, proposed_action, exclude_model=None: {"approved": True, "reason": "同意"},
    delete_source=lambda source: 0,
    ingest_file=lambda path, category="": ingest_calls.append((path, category)) or 3,
    mark_ingesting=lambda source: None,
    mark_done=lambda source: None,
    remove_pending=lambda session_id, filenames: remove_calls.append(filenames),
)
try:
    result = tools._run_ingest({"filenames": ["a.pdf"]}, "s1", "存到知識庫", "model-a")
    assert result == {"executed": True, "reason": "同意", "ingested": ["a.pdf"], "chunks": 3}
    assert len(ingest_calls) == 1
    assert remove_calls == [["a.pdf"]]
finally:
    restore()

# 情境 6：_run_ingest 拒絕路徑 —— ingest_file 完全沒被呼叫、pending 不變
ingest_calls, remove_calls = [], []
restore = _patch(
    tools,
    get_pending=lambda session_id: [PendingFile("a.pdf", "/tmp/a.pdf", 10, "")],
    evaluate_action=lambda user_message, proposed_action, exclude_model=None: {"approved": False, "reason": "意圖不明確"},
    ingest_file=lambda path, category="": ingest_calls.append((path, category)) or 3,
    remove_pending=lambda session_id, filenames: remove_calls.append(filenames),
)
try:
    result = tools._run_ingest({"filenames": ["a.pdf"]}, "s1", "這份文件", "model-a")
    assert result == {"executed": False, "reason": "意圖不明確"}
    assert ingest_calls == []
    assert remove_calls == []
finally:
    restore()

# 情境 7：Answer Gate 不合格 —— 觸發一次修正，且只觸發一次
# search_fusion 被 _run_search 使用、現在也在 tools.py 裡；evaluate_answer 仍是 agent.py 直接呼叫，patch 留在 agent
gate_calls = []
restore_tools = _patch(
    tools,
    search_fusion=lambda query, top_k=5, source=None, category=None: [{"article": "第1條", "content": "測試條文"}],
)
restore = _patch(
    agent,
    evaluate_answer=lambda question, context, answer, primary_model, clients=None: (
        gate_calls.append(1)
        or {"passed": False, "faithfulness": False, "relevance": True, "reason": "答案跟參考內容對不上"}
    ),
)
try:
    client = _sequenced_client(
        [
            [_tool_call_chunk("search_knowledge_base", json_mod.dumps({"query": "測試"})), _chunk(finish_reason="tool_calls")],
            [_chunk(content="初步答案"), _chunk(finish_reason="stop")],
            [_chunk(content="修正後答案"), _chunk(finish_reason="stop")],
        ]
    )
    tokens, thread_exc = _run([(client, "model-a")])
    assert thread_exc == []
    assert gate_calls == [1]  # 只觸發一次，不會遞迴
    full = "".join(t for t in tokens if t)
    assert "初步答案" in full
    assert "系統自動覆核後修正" in full
    assert "修正後答案" in full
finally:
    restore()
    restore_tools()

# 情境 8：Answer Gate 通過 —— 不出現修正段落
gate_calls = []
restore_tools = _patch(
    tools,
    search_fusion=lambda query, top_k=5, source=None, category=None: [{"article": "第1條", "content": "測試條文"}],
)
restore = _patch(
    agent,
    evaluate_answer=lambda question, context, answer, primary_model, clients=None: (
        gate_calls.append(1)
        or {"passed": True, "faithfulness": True, "relevance": True, "reason": "ok"}
    ),
)
try:
    client = _sequenced_client(
        [
            [_tool_call_chunk("search_knowledge_base", json_mod.dumps({"query": "測試"})), _chunk(finish_reason="tool_calls")],
            [_chunk(content="正常答案"), _chunk(finish_reason="stop")],
        ]
    )
    tokens, thread_exc = _run([(client, "model-a")])
    assert thread_exc == []
    assert gate_calls == [1]
    full = "".join(t for t in tokens if t)
    assert full == "正常答案"
    assert "系統自動覆核後修正" not in full
finally:
    restore()
    restore_tools()

# 情境 9：模型呼叫不存在的工具名稱 —— 回傳錯誤訊息當工具結果，不中斷整輪對話
client = _sequenced_client(
    [
        [_tool_call_chunk("no_such_tool", json_mod.dumps({})), _chunk(finish_reason="tool_calls")],
        [_chunk(content="收到"), _chunk(finish_reason="stop")],
    ]
)
tokens, thread_exc = _run([(client, "model-a")])
assert thread_exc == []
full = "".join(t for t in tokens if t)
assert full == "收到"

print("all agent checks passed")
