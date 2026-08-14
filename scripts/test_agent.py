"""
run_chat_thread 的檢查：streaming token 拼接、provider rate-limit fallback。
clients 全部注入假物件，不連真的 OpenAI/Gemini。
用法：python scripts/test_agent.py
"""
import sys
import queue as queue_mod
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from openai import RateLimitError
from app.agent import run_chat_thread


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


def _run(clients):
    q = queue_mod.Queue()
    thread_exc = []
    run_chat_thread(
        [{"role": "user", "content": "hi"}], q, ImmediateLoop(), thread_exc, clients=clients
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

print("all agent checks passed")
