"""
純函式的最小 assert 檢查：chunk_by_article、_rrf_merge、上傳驗證。
不連 ES/OpenAI/Redis，可獨立執行。
用法：python scripts/test_units.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest import chunk_by_article
from app.retriever import _rrf_merge
from app.main import _safe_filename, _valid_content

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

print("all unit checks passed")
