"""Code Mode runner: thực thi mã Python trong runtime cục bộ.

Model gửi tool call run_code(code=...), runtime chạy script với SDK tools.*,
và chỉ trả lại kết quả tóm tắt (findings/stdout) về cho context của model.
Raw file logs không bao giờ được gửi vào context của model.
"""
from __future__ import annotations

import collections
import contextlib
import datetime
import io
import itertools
import json
import math
import re
import sys
import traceback
from pathlib import Path

from langchain_core.tools import tool

import paths
from tools.files import _list, _read, _write, read_log_time_range


class ToolProxy:
    """SDK proxy cung cấp các hàm thao tác file/log an toàn trong workspace."""

    def __init__(self, workspace: Path = paths.WORKSPACE_DIR):
        self.workspace = workspace

    def list_files(self, path: str = ".") -> list[dict]:
        """Liệt kê các file và thư mục trong đường dẫn tương đối workspace."""
        res = _list(self.workspace, path)
        if not res.get("ok"):
            err = res.get("error", {})
            raise RuntimeError(f"list_files({path}): [{err.get('code')}] {err.get('message')}")
        return res.get("items", [])

    def read_file(self, path: str) -> str:
        """Đọc toàn bộ nội dung file văn bản trong workspace trả về chuỗi (string)."""
        res = _read(self.workspace, path)
        if not res.get("ok"):
            err = res.get("error", {})
            raise RuntimeError(f"read_file({path}): [{err.get('code')}] {err.get('message')}")
        return res.get("content", "")

    def read_log_time_range(self, path: str, start_time: str, end_time: str) -> str:
        """Đọc log trong khoảng giờ HH:MM:SS."""
        return read_log_time_range.invoke({"path": path, "start_time": start_time, "end_time": end_time})

    def write_file(self, path: str, content: str) -> dict:
        """Ghi nội dung văn bản vào file dưới output/."""
        res = _write(self.workspace, path, content)
        if not res.get("ok"):
            err = res.get("error", {})
            raise RuntimeError(f"write_file({path}): [{err.get('code')}] {err.get('message')}")
        return res


def execute_code(code: str, workspace: Path = paths.WORKSPACE_DIR) -> dict:
    """Thực thi mã Python trong runtime và bắt stdout / kết quả."""
    stdout_buffer = io.StringIO()
    proxy = ToolProxy(workspace)

    # Cung cấp sẵn các thư viện phổ biến và SDK tools
    env = {
        "tools": proxy,
        "re": re,
        "json": json,
        "math": math,
        "collections": collections,
        "datetime": datetime,
        "itertools": itertools,
        "__builtins__": __builtins__,
    }

    old_stdout = sys.stdout
    sys.stdout = stdout_buffer
    try:
        exec(code, env)
        output = stdout_buffer.getvalue().strip()
        if not output and "result" in env:
            res_val = env["result"]
            output = json.dumps(res_val, ensure_ascii=False, indent=2) if isinstance(res_val, (dict, list)) else str(res_val)
        if not output:
            output = "(Script thực thi thành công nhưng không có output nào được in ra qua print())"
        return {"ok": True, "stdout": output}
    except Exception as exc:
        return {
            "ok": False,
            "error": {
                "code": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
    finally:
        sys.stdout = old_stdout


@tool
def run_code(code: str) -> str:
    """Thực thi mã Python trong runtime để đọc, duyệt và tổng hợp dữ liệu từ nhiều file trong workspace.

    Trong script Python, bạn có sẵn đối tượng `tools`:
    - `tools.list_files(path=".")` -> list[dict]: danh sách file/thư mục (name, path, type).
    - `tools.read_file(path)` -> str: đọc toàn bộ nội dung file.
    - `tools.read_log_time_range(path, start_time, end_time)` -> str: đọc log theo khoảng giờ.
    - `tools.write_file(path, content)` -> dict: ghi file vào output/.

    Sử dụng `print(...)` để in ra các kết quả/findings đã tổng hợp.
    Dùng tool này thay vì đọc trực tiếp nhiều file để tránh làm phình to context (Code Mode).
    """
    return json.dumps(execute_code(code), ensure_ascii=False)
