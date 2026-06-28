from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from app.agent import build_agent
from app.memory import get_history, append_turn
from pathlib import Path
import asyncio
import json

DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)

app = FastAPI()

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
        response = await asyncio.to_thread(agent.chat, full_message)
        answer = str(response)
        append_turn(req.session_id, "assistant", answer)

        # stream token by token
        for char in answer:
            yield {"event": "token", "data": char}
            await asyncio.sleep(0)

        yield {"event": "final_answer", "data": json.dumps({"answer": answer}, ensure_ascii=False)}

    return EventSourceResponse(event_generator())

@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    saved = []
    for f in files:
        dest = DOCS_DIR / f.filename
        dest.write_bytes(await f.read())
        saved.append(f.filename)
    return {"uploaded": saved}

@app.get("/health")
def health():
    return {"status": "ok"}

app.mount("/", StaticFiles(directory="static", html=True), name="static")
