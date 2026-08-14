from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from app.memory import get_history, append_turn
from app.retriever import get_embed_model
from app.ingest import ingest_file, delete_source, list_indexed_docs, list_categories
from app.agent import build_messages, run_chat_thread
from pathlib import Path
import asyncio
import json

DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)

_ingesting: set[str] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時預載 BGE-M3 embedding model，避免第一次請求時 cold start 延遲
    await asyncio.to_thread(get_embed_model)
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
                None, run_chat_thread, messages, queue, loop, thread_exc
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
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    overwrite: bool = Form(False),
    category: str = Form(""),
):
    saved, skipped, rejected, reindexing = [], [], [], []
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
        if dest.exists() and not overwrite:
            skipped.append(filename)
            continue
        if dest.exists() and overwrite:
            delete_source(dest.stem)
            reindexing.append(filename)
        else:
            saved.append(filename)
        dest.write_bytes(data)
        _ingesting.add(dest.stem)

        # 用預設參數捕捉當前值，避免 closure 共享迴圈變數
        async def _bg(p=dest, cat=category):
            await asyncio.to_thread(ingest_file, p, cat)
            _ingesting.discard(p.stem)

        background_tasks.add_task(_bg)
    return {
        "uploaded": saved,
        "skipped": skipped,
        "rejected": rejected,
        "reindexing": reindexing,
    }


@app.get("/ingest-status")
def ingest_status(source: str):
    return {"done": source not in _ingesting}


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
