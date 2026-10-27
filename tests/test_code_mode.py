"""Kiểm thử Code Mode (Challenge B2):

1. Thực thi mã Python với SDK tools.* trong runtime cục bộ.
2. Đọc và lọc log trên file lớn mà không đưa raw log vào context của LLM.
3. Bắt lỗi runtime, cú pháp an toàn, chặn truy cập ngoài workspace.
4. Đo lường kích thước output trả về context (chứng minh raw log không vào context).
"""
import json
import paths
import pytest
from tools.code_runner import execute_code, run_code


def test_run_code_executes_python_and_captures_stdout(lab_dirs):
    code = """
total = sum(i * 2 for i in range(5))
print(f"Tổng: {total}")
"""
    result = execute_code(code, workspace=paths.WORKSPACE_DIR)
    assert result["ok"] is True
    assert result["stdout"] == "Tổng: 20"


def test_run_code_reads_and_aggregates_server_log(lab_dirs):
    # Kiểm tra kịch bản phân tích log server.log (Challenge B2 trong slide)
    code = """
content = tools.read_file("data/server.log")
lines = content.splitlines()

# Đếm số lỗi DB_POOL_EXHAUSTED và tổng số lỗi ERROR
pool_errors = sum(1 for line in lines if "DB_POOL_EXHAUSTED" in line)
all_errors = sum(1 for line in lines if " ERROR " in line)

print(f"DB_POOL_EXHAUSTED: {pool_errors}/{all_errors}")
"""
    result = execute_code(code, workspace=paths.WORKSPACE_DIR)
    assert result["ok"] is True
    assert "DB_POOL_EXHAUSTED: 38/39" in result["stdout"]


def test_run_code_list_files_and_iterates(lab_dirs):
    code = """
files = tools.list_files("data")
names = [f["name"] for f in files if f["type"] == "file"]
print(", ".join(sorted(names)))
"""
    result = execute_code(code, workspace=paths.WORKSPACE_DIR)
    assert result["ok"] is True
    assert "server.log" in result["stdout"]
    assert "weekly_notes.md" in result["stdout"]


def test_code_mode_context_savings(lab_dirs):
    """Chứng minh Challenge B2: Raw log không đi vào LLM context."""
    server_log_path = paths.WORKSPACE_DIR / "data" / "server.log"
    raw_log_size = len(server_log_path.read_text(encoding="utf-8"))
    assert raw_log_size > 25_000, "File log gốc phải lớn hơn 25k ký tự"

    # Chạy Code Mode để lấy kết quả phân tích
    code = """
log_data = tools.read_file("data/server.log")
leak_lines = [l for l in log_data.splitlines() if "leak" in l]
print(f"Tìm thấy {len(leak_lines)} dòng nghi ngờ leak: {leak_lines[0].strip() if leak_lines else ''}")
"""
    result_raw = run_code.invoke({"code": code})
    result = json.loads(result_raw)

    assert result["ok"] is True
    stdout = result["stdout"]
    assert "leak" in stdout
    assert "settlement.py:88" in stdout

    # Kích thước trả về context chỉ dưới 350 ký tự thay vì >29,000 ký tự của raw log
    output_chars_sent_to_model = len(result_raw)
    assert output_chars_sent_to_model < 500, f"Context trả về quá lớn ({output_chars_sent_to_model} chars)"
    print(f"\n--- Code Mode Context Saving: Raw log = {raw_log_size} chars -> Context trả về = {output_chars_sent_to_model} chars ---")


def test_run_code_handles_syntax_and_runtime_errors(lab_dirs):
    # 1. Lỗi cú pháp
    bad_syntax = "def foo(:\n    pass"
    res1 = execute_code(bad_syntax, workspace=paths.WORKSPACE_DIR)
    assert res1["ok"] is False
    assert res1["error"]["code"] == "SyntaxError"

    # 2. Lỗi ngoại lệ runtime (chia cho 0)
    zero_div = "x = 1 / 0"
    res2 = execute_code(zero_div, workspace=paths.WORKSPACE_DIR)
    assert res2["ok"] is False
    assert res2["error"]["code"] == "ZeroDivisionError"


def test_run_code_blocks_outside_workspace(lab_dirs):
    code = """
try:
    content = tools.read_file("../secret.txt")
    print("FAILED_TO_BLOCK")
except Exception as e:
    print(f"BLOCKED: {e}")
"""
    result = execute_code(code, workspace=paths.WORKSPACE_DIR)
    assert result["ok"] is True
    assert "BLOCKED" in result["stdout"]
