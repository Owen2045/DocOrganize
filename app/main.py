from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from app.agent import build_agent
from app.memory import get_history, append_turn
from app.retriever import get_embed_model
from pathlib import Path
import asyncio
import json

DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時預載模型，避免第一次請求 timeout
    await asyncio.to_thread(get_embed_model)
    yield

app = FastAPI(lifespan=lifespan)

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    history = get_history(req.session_id)
    agent = build_agent()

    # 把歷史塞進 context
    context = ""
    if history:
        context = "\n".join(f"{t['role']}: {t['content']}" for t in history[-6:])
        context = f"對話歷史：\n{context}\n\n"

    full_message = context + req.message
    append_turn(req.session_id, "user", req.message)

    async def event_generator():
        try:
            response = await asyncio.to_thread(agent.chat, full_message)
            answer = str(response)
            append_turn(req.session_id, "assistant", answer)
            for char in answer:
                yield {"event": "token", "data": char}
                await asyncio.sleep(0)
            yield {"event": "final_answer", "data": json.dumps({"answer": answer}, ensure_ascii=False)}
        except Exception as e:
            import logging, traceback
            logging.error(traceback.format_exc())
            yield {"event": "error", "data": "系統發生錯誤，請稍後再試。"}

    return EventSourceResponse(event_generator())

ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

def _safe_filename(name: str) -> str:
    return Path(name).name  # strip any directory components

def _valid_content(name: str, data: bytes) -> bool:
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False
    if ext == ".pdf" and not data.startswith(b"%PDF"):
        return False
    return True

@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    saved, skipped, rejected = [], [], []
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
        if dest.exists():
            skipped.append(filename)
            continue

        dest.write_bytes(data)
        saved.append(filename)
    return {"uploaded": saved, "skipped": skipped, "rejected": rejected}

@app.get("/health")
def health():
    return {"status": "ok"}

app.mount("/", StaticFiles(directory="static", html=True), name="static")
