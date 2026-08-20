"""
純函式的最小 assert 檢查：chunk_by_article、_rrf_merge、上傳驗證。
不連 ES/OpenAI/Redis，可獨立執行。
用法：python scripts/test_units.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest import chunk_by_article, chunk_generic
from app.retriever import _rrf_merge
from app.main import _safe_filename, _valid_content
from app.judge import _pick_judge_client
from app.tools import run_tool

# chunk_by_article：按「第N條」regex 切塊
chunks = chunk_by_article("第1條 內容一\n第2-1條 內容二", "test")
assert len(chunks) == 2
assert chunks[0] == {"article": "第1條", "content": "第1條 內容一", "source": "test"}
assert chunks[1] == {"article": "第2-1條", "content": "第2-1條 內容二", "source": "test"}

assert chunk_by_article("沒有條號的文字", "test") == []
assert chunk_by_article("第99條", "test") == []  # 條號後沒內容，該塊被丟棄

# _rrf_merge：Reciprocal Rank Fusion
merged = _rrf_merge([["a", "b", "c"], ["b", "a"]])
assert set(merged) == {"a", "b", "c"}
assert merged[-1] == "c"  # c 只在一份排名出現，分數最低，排最後
assert _rrf_merge([[], []]) == []

# _safe_filename：防路徑穿越
assert _safe_filename("../../etc/passwd") == "passwd"
assert _safe_filename("normal.pdf") == "normal.pdf"

# _valid_content：副檔名 + PDF magic bytes
assert _valid_content("a.pdf", b"%PDF-1.4 rest") is True
assert _valid_content("a.pdf", b"not a pdf") is False
assert _valid_content("a.txt", b"hello") is True
assert _valid_content("a.exe", b"MZ...") is False

# run_tool：查無此工具名稱時回傳錯誤訊息，不拋例外，也不算檢索內容
result, contributes = run_tool("no_such_tool", {}, "s1", "hi", "model-a")
assert contributes is False
assert "找不到工具" in result

# chunk_generic：固定長度滑動視窗切塊（compare_documents 的大文件降級依賴它）
chunks = chunk_generic("A" * 1500, "test", size=600, overlap=100)
assert [c["article"] for c in chunks] == ["段落1", "段落2", "段落3"]
assert all(c["source"] == "test" for c in chunks)
assert chunk_generic("", "test") == []

# _pick_judge_client：三種選擇情境
# 1. 雙 provider，排除 primary，應選另一個且非同模型自我檢查
client_a, client_b = object(), object()
c, m, same = _pick_judge_client([(client_a, "model-a"), (client_b, "model-b")], exclude_model="model-a")
assert (c, m, same) == (client_b, "model-b", False)

# 2. 只有單一 provider：退化為同模型自我檢查
c, m, same = _pick_judge_client([(client_a, "model-a")], exclude_model="model-a")
assert (c, m, same) == (client_a, "model-a", True)

# 3. 沒有 exclude_model 可排除（呼叫端理論上永遠會明確傳入，這裡只驗證保底行為）：直接取第一個
c, m, same = _pick_judge_client([(client_a, "model-a"), (client_b, "model-b")], exclude_model=None)
assert (c, m, same) == (client_a, "model-a", False)

print("all unit checks passed")
