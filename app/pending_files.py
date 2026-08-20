import json
from dataclasses import dataclass, asdict
from app.memory import get_redis

# 跟 app.ingest.sweep_orphaned_docs 的孤兒檔案掃描窗口對齊，
# 避免「Redis 指標消失但磁碟檔案還在」的空窗期。
PENDING_TTL = 60 * 60 * 48


@dataclass(frozen=True)
class PendingFile:
    """session 已上傳但尚未歸檔/比較的暫存檔案；記錄永遠整份重寫回 Redis，不會原地修改，故設為不可變。"""

    filename: str
    path: str
    size: int
    category: str = ""


def get_pending(session_id: str) -> list[PendingFile]:
    """取得暫存檔案清單。"""
    raw = get_redis().get(f"pending:{session_id}")
    return [PendingFile(**d) for d in json.loads(raw)] if raw else []


def _save(session_id: str, pending: list[PendingFile]) -> None:
    get_redis().set(
        f"pending:{session_id}",
        json.dumps([asdict(p) for p in pending], ensure_ascii=False),
        ex=PENDING_TTL,
    )


def add_pending(session_id: str, filename: str, path: str, size: int, category: str = "") -> None:
    """同檔名視為取代（重新上傳同名檔案會覆蓋原本的暫存紀錄）。"""
    pending = [p for p in get_pending(session_id) if p.filename != filename]
    pending.append(PendingFile(filename, path, size, category))
    _save(session_id, pending)


def remove_pending(session_id: str, filenames: list[str]) -> None:
    """移除暫存檔案清單。"""
    pending = [p for p in get_pending(session_id) if p.filename not in filenames]
    _save(session_id, pending)
