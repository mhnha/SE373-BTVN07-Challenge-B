"""Cấu hình project: hằng số hiển thị, giới hạn thực thi và settings đọc từ .env."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

import paths

PROJECT_NAME = "stage-02-skills"
APP_TITLE = "Stage 02: Skills & Context Management"
CAPABILITY_TEXT = "Tools: read_file, write_file, list_files, read_log_time_range, memory, run_code (Code Mode). Skills: catalog tự phát hiện từ workspace/skills."

# Giới hạn số lần gọi model trong một lượt chat (ModelCallLimitMiddleware).
MODEL_CALL_LIMIT = 8
# Giới hạn số tool calls trong một lượt chat (ToolCallLimitMiddleware).
TOOL_CALL_LIMIT = 20
# Giới hạn số bước của LangGraph (recursion limit), không phải số tool calls.
RECURSION_LIMIT = 50

# Compact session: khi tổng ký tự history vượt ngưỡng thì tóm tắt phần cũ, giữ COMPACT_KEEP_LAST message cuối.
# Ngưỡng thấp để demo; thực tế nên tính theo token (70-80% context window).
COMPACT_TRIGGER_CHARS = 6000
COMPACT_KEEP_LAST = 6


@dataclass(frozen=True)
class Settings:
    api_key: str
    model_name: str
    base_url: str | None


def load_settings() -> tuple[Settings | None, list[str]]:
    """Đọc .env của project này. Trả (settings, []) hoặc (None, danh sách biến còn thiếu)."""
    load_dotenv(paths.ENV_PATH, override=False)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model_name = os.getenv("MODEL_NAME", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    missing = [name for name, value in (("OPENAI_API_KEY", api_key), ("MODEL_NAME", model_name)) if not value]
    if missing:
        return None, missing
    return Settings(api_key=api_key, model_name=model_name, base_url=base_url), []
