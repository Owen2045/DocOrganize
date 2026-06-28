# CLAUDE.md

## 記憶恢復

新 session 開始時，依照以下步驟還原上下文：

1. 去上一層目錄的 `claude_memory/` 找與當前專案同名的資料夾
   - 路徑範例：`../claude_memory/spotify_tidy/`
2. 先讀 `MEMORY.md`（索引檔，列出所有記憶檔案與摘要）
3. 再依需要讀取其他 `.md` 檔案（架構、伺服器、開發守則、交接文件等）
4. 如果找不到 `claude_memory/` 資料夾，改讀當前專案的 `README.md`

記憶資料夾由獨立私人 git repo 管理（`claude_memory/` repo），跨機器透過 `git pull` 同步。
