import json
from app.memory import get_redis

# 跟 app.ingest.sweep_orphaned_docs 的孤兒檔案掃描窗口對齊，
# 避免「Redis 指標消失但磁碟檔案還在」的空窗期。
PENDING_TTL = 60 * 60 * 48


def get_pending(session_id: str) -> list[dict]:
    raw = get_redis().get(f"pending:{session_id}")
    return json.loads(raw) if raw else []


def add_pending(session_id: str, filename: str, path: str, size: int, category: str = "") -> None:
    """同檔名視為取代（重新上傳同名檔案會覆蓋原本的暫存紀錄）。"""
    pending = [p for p in get_pending(session_id) if p["filename"] != filename]
    pending.append({"filename": filename, "path": path, "size": size, "category": category})
    get_redis().set(f"pending:{session_id}", json.dumps(pending, ensure_ascii=False), ex=PENDING_TTL)


def remove_pending(session_id: str, filenames: list[str]) -> None:
    pending = [p for p in get_pending(session_id) if p["filename"] not in filenames]
    get_redis().set(f"pending:{session_id}", json.dumps(pending, ensure_ascii=False), ex=PENDING_TTL)
