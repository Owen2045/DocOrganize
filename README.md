# CiteFund — 台灣金融法規 Q&A Agent

針對證券投資信託及顧問相關法規的智能問答系統，支援對話記憶、串流輸出與動態 LLM 切換。

> **Demo**：
https://d4fa-118-166-165-74.ngrok-free.app
- 直接在對話框輸入問題即可使用，無需登入。

## 功能

- **Multi-Query RAG-Fusion**：同一問題改寫成 3 個角度的 query，各自檢索後 RRF 合併，提升口語問法的召回率
- **混合檢索**：Elasticsearch BM25 + kNN + RRF fusion，兼顧關鍵字命中與語意相似
- **HyDE**：先讓 LLM 生成假設條文再 embed，縮短白話問法與法規用詞的語意鴻溝
- **語意嵌入**：BAAI/bge-m3 本地模型（1024 dim），無需外部 Embedding API
- **對話記憶**：每個 session 儲存最近 6 輪歷史，支援多輪追問
- **串流輸出**：SSE 即時逐 token 推送，回答邊生成邊顯示
- **動態 LLM fallback**：主 LLM 額度耗盡時自動切換備援，保持服務可用

## 技術棧

| 層 | 技術 |
|----|------|
| LLM | OpenAI gpt-4o-mini（+ Gemini 備援）|
| Agent | OpenAI Function Calling（原生，非框架）|
| Embedding | BAAI/bge-m3（本地，1024 dim）|
| 向量 / 全文 DB | Elasticsearch 8.x（BM25 + kNN + RRF）|
| 對話記憶 | Redis 7 |
| API | FastAPI + sse-starlette |
| 前端 | 純 HTML/CSS/JS + marked.js |
| 部署 | Docker Compose |

## 系統架構

```
瀏覽器
  │  POST /chat/stream（SSE）
  ▼
FastAPI
  ├── Redis — 對話歷史（最近 6 輪）
  └── OpenAI Function Calling（stream=True）
        第一輪：LLM 決定是否呼叫 search_knowledge_base
        finish_reason == "tool_calls" → retriever.py → ES
        第二輪：LLM stream 最終回答 → SSE token by token

retriever.py（search_fusion）
  ├── LLM 改寫 3 個 query → 批次 BGE-M3 embed
  ├── 各 query：BM25（原始文字）+ kNN（embed）→ RRF
  └── 3 路結果再次 RRF merge → top 5 chunks
```

## 專案結構

```
CiteFund/
├── app/
│   ├── main.py        # FastAPI 入口
│   ├── ingest.py      # 文件切塊、embedding、寫入 ES
│   ├── retriever.py   # HyDE + hybrid retrieval（BM25+kNN+RRF）
│   ├── memory.py      # Redis 對話記憶
│   └── config.py      # 環境變數
├── static/
│   └── index.html     # 聊天 UI
├── scripts/
│   └── ingest.py      # CLI 批次 ingest
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## 設計決策

- **原生 Function Calling 取代框架**：直接使用 OpenAI SDK 的 tool use 流程，保留對 streaming、tool 執行順序的完整控制
- **Multi-Query RAG-Fusion**：GPT 將問題改寫為 3 個不同角度，批次 embed 後各自 BM25+kNN，再做第二層 RRF；有效覆蓋使用者口語問法與法規術語之間的差距
- **HyDE**：kNN 用假設條文 embed，BM25 仍用原始 query，兩者 RRF 合併，兼顧召回率與精準度
- **手動 RRF**：ES basic license 不支援內建 RRF pipeline，以 Python 實作兩次查詢 + reciprocal rank fusion
- **本地 Embedding**：bge-m3 在本機推理，避免每次查詢呼叫外部 API 產生延遲與費用
