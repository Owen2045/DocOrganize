# CiteFund — 台灣金融法規 Q&A Agent

針對證券投資信託及顧問相關法規的智能問答系統，支援對話記憶與串流輸出。

## 功能

- **混合檢索**：Elasticsearch BM25 + kNN + RRF fusion，搭配 BGE reranker 精排
- **語意嵌入**：BGE-M3 本地模型，無需外部 Embedding API
- **對話記憶**：Redis 儲存每個 session 的歷史對話
- **串流輸出**：SSE (Server-Sent Events) 即時回應
- **文件上傳**：透過 UI 上傳 .md / .txt 法規文件

## 技術棧

| 層 | 技術 |
|----|------|
| LLM | OpenAI GPT-4o |
| Agent | LlamaIndex FunctionCallingAgent |
| Embedding | BAAI/bge-m3（本地） |
| Reranker | BAAI/bge-reranker-v2-m3（本地） |
| 向量庫 | Elasticsearch 8.13（hybrid retrieval） |
| 記憶 | Redis 7 |
| API | FastAPI + sse-starlette |
| 部署 | Docker Compose |

## 快速開始

### 1. 設定環境變數

```bash
cp .env.example .env
# 填入 OPENAI_API_KEY
```

### 2. 啟動服務

```bash
docker compose up -d --build
```

### 3. 放入法規文件

將 `.md` 或 `.txt` 格式的法規文件放到 `docs/` 目錄，或透過 UI 上傳。

### 4. 建立索引

```bash
docker compose exec app python scripts/ingest.py
```

### 5. 開啟 UI

瀏覽器打開 `http://localhost:8000`

## 專案結構

```
CiteFund/
├── app/
│   ├── main.py        # FastAPI 入口，/chat/stream, /upload
│   ├── agent.py       # LlamaIndex FunctionCallingAgent
│   ├── retriever.py   # ES hybrid retrieval + reranker
│   ├── memory.py      # Redis 對話記憶
│   ├── tools.py       # Agent FunctionTool
│   └── config.py      # 環境變數
├── static/
│   └── index.html     # 聊天 UI
├── scripts/
│   ├── ingest.py      # 文件切分 + 向量化 + 寫入 ES
│   ├── test_retrieval.py
│   └── deploy.sh      # rsync + restart
├── docs/              # 法規文件（不上 git）
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## 環境變數

| 變數 | 說明 | 預設值 |
|------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 金鑰 | 必填 |
| `ES_URL` | Elasticsearch URL | `http://elasticsearch:9200` |
| `ES_INDEX` | ES 索引名稱 | `citefund_laws` |
| `REDIS_URL` | Redis URL | `redis://redis:6379` |
