import json
import redis as redis_lib
from app.config import REDIS_URL

_r = None

def get_redis():
    global _r
    if _r is None:
        _r = redis_lib.from_url(REDIS_URL)
    return _r

def get_history(session_id: str) -> list[dict]:
    raw = get_redis().get(f"chat:{session_id}")
    return json.loads(raw) if raw else []

def append_turn(session_id: str, role: str, content: str):
    history = get_history(session_id)
    history.append({"role": role, "content": content})
    get_redis().set(f"chat:{session_id}", json.dumps(history, ensure_ascii=False))
