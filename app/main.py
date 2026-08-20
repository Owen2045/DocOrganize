from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from app.memory import get_history, append_turn
from app.retriever import get_embed_model
from app.ingest import list_indexed_docs, list_categories, is_ingesting, sweep_orphaned_docs
from app.pending_files import get_pending, add_pending, remove_pending
from app.agent import build_messages, run_chat_thread
from app.config import DOCS_DIR
from pathlib import Path
import asyncio
import json

DOCS_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時預載 BGE-M3 embedding model，避免第一次請求時 cold start 延遲
    await asyncio.to_thread(get_embed_model)
    # 清掉超過 48h、從未被歸檔的孤兒暫存檔（compare 完沒被 ingest 的檔案）
    await asyncio.to_thread(sweep_orphaned_docs)
    yield


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    history = get_history(req.session_id)
    append_turn(req.session_id, "user", req.message)
    messages = build_messages(history, req.message)

    async def event_generator():
        try:
            loop = asyncio.get_event_loop()
            queue: asyncio.Queue = asyncio.Queue()
            thread_exc: list = []

            # run_in_executor：將 sync 函式丟到 thread pool，不阻塞 async event loop
            future = loop.run_in_executor(
                None, run_chat_thread, messages, queue, loop, thread_exc, req.session_id
            )

            answer = ""
            while True:
                token = await queue.get()
                if token is None:
                    break
                answer += token
                yield {"event": "token", "data": token}

            await future
            if thread_exc:
                raise thread_exc[0]

            append_turn(req.session_id, "assistant", answer)
            yield {
                "event": "final_answer",
                "data": json.dumps({"answer": answer}, ensure_ascii=False),
            }
        except Exception:
            import logging, traceback

            logging.error(traceback.format_exc())
            yield {"event": "error", "data": "系統發生錯誤，請稍後再試。"}

    return EventSourceResponse(event_generator())


ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt"}
MAX_FILE_SIZE = 20 * 1024 * 1024


def _safe_filename(name: str) -> str:
    return Path(name).name


def _valid_content(name: str, data: bytes) -> bool:
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False
    if ext == ".pdf" and not data.startswith(b"%PDF"):
        return False
    return True


@app.post("/upload")
async def upload(
    session_id: str = Form(...),
    files: list[UploadFile] = File(...),
    category: str = Form(""),
):
    """只暫存檔案，不自動 ingest。是否寫入知識庫、要不要覆蓋同名文件，
    完全交給聊天流程裡的 ingest_documents + Action Gate 判斷。"""
    staged, rejected = [], []
    for f in files:
        filename = _safe_filename(f.filename)
        data = await f.read()
        if len(data) > MAX_FILE_SIZE:
            rejected.append(f"{filename}（檔案超過 20MB）")
            continue
        if not _valid_content(filename, data):
            rejected.append(f"{filename}（不支援的檔案類型）")
            continue
        dest = DOCS_DIR / filename
        dest.write_bytes(data)
        add_pending(session_id, filename, str(dest), len(data), category)
        staged.append(filename)
    return {"staged": staged, "rejected": rejected, "pending": get_pending(session_id)}


@app.get("/pending")
def pending(session_id: str):
    return {"pending": get_pending(session_id)}


@app.delete("/pending")
def delete_pending(session_id: str, filename: str):
    match = next((p for p in get_pending(session_id) if p.filename == filename), None)
    if match:
        Path(match.path).unlink(missing_ok=True)
        remove_pending(session_id, [filename])
    return {"pending": get_pending(session_id)}


@app.get("/ingest-status")
def ingest_status(source: str):
    return {"done": not is_ingesting(source)}


@app.get("/indexed-docs")
def list_docs():
    return {"docs": list_indexed_docs()}


@app.get("/categories")
def get_categories():
    return {"categories": list_categories()}


@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
