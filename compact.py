"""Compaction không gọi model (không tốn thêm token/TPM).

Giữ lại: yêu cầu của người dùng, các câu trả lời đã chốt, bằng chứng lỗi (gộp theo mẫu, có số lần + khoảng giờ).
Bỏ đi: log dài, dòng lặp, tool call/result cũ. Luôn giữ nguyên các lượt chat cuối (tail) bắt đầu bằng HumanMessage
nên không bao giờ làm mồ côi ToolMessage. Compact nhiều lần liên tiếp thì bản tóm tắt cũ được cộng dồn, không lồng nhau.
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

SUMMARY_PREFIX = "[TÓM TẮT PHIÊN TRƯỚC - đã compact]\n"
ERROR_RE = re.compile(r"ERROR|WARN|CRITICAL|FATAL|EXCEPTION|TIMEOUT|FAILED|DB_POOL_EXHAUSTED|leak|conn\.close", re.I)
SEVERE_RE = re.compile(r"ERROR|CRITICAL|FATAL|EXCEPTION|leak|conn\.close", re.I)

# Giới hạn kích thước bản tóm tắt (ngưỡng compact demo chỉ 6000 ký tự nên summary phải nhỏ).
MAX_USER_NOTES, USER_NOTE_CHARS = 6, 240
MAX_ANSWERS, ANSWER_CHARS = 4, 320
MAX_EVIDENCE, EVIDENCE_CHARS = 8, 220

H_USER = "## Mục tiêu / yêu cầu và quyết định người dùng"
H_ANSWER = "## Kết luận / vấn đề chưa giải quyết (từ câu trả lời cũ; cần đối chiếu bằng chứng)"
H_EVIDENCE = "## Bằng chứng lỗi trích từ tool (gộp theo mẫu, đã bỏ log trùng)"
H_NOTE = "## Lưu ý"

_LINE_PREFIX = re.compile(r"^\[line \d+\]\s*")
_DATETIME = re.compile(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\b")
_TIME = re.compile(r"\b(\d{2}:\d{2}:\d{2})\b")
_VOLATILE_ID = re.compile(r"\b(txn_id|user_id|request_id|trace_id)=\S+")
_EV_SUFFIX = re.compile(r"\s+\(×(\d+)(?:; (\S+)→(\S+))?\)$")


# ---------------------------------------------------------------- đo kích thước / điều kiện kích hoạt


def msg_chars(message: BaseMessage) -> int:
    content = message.text or ""
    if isinstance(message, AIMessage):
        content += json.dumps(message.tool_calls, ensure_ascii=False, default=str)
    return len(content)


def total_chars(messages: list[BaseMessage]) -> int:
    return sum(msg_chars(m) for m in messages)


def should_compact(messages: list[BaseMessage], trigger: int) -> bool:
    return total_chars(messages) > trigger


def compaction_reasons(messages: list[BaseMessage], trigger_chars: int,
                       milestone: bool = False, cold_start: bool = False) -> list[str]:
    reasons = []
    if should_compact(messages, trigger_chars):
        reasons.append("context_threshold")
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    if len(tool_messages) >= 2:
        # Output trùng nhau hoặc nhiều output dài; tránh kích hoạt vì tool nhỏ.
        fingerprints = [m.text[:500] for m in tool_messages if len(m.text) > 600]
        if len(fingerprints) >= 2 and (len(set(fingerprints)) < len(fingerprints)
                                       or sum(len(m.text) for m in tool_messages) > trigger_chars // 2):
            reasons.append("repeated_or_large_tool_outputs")
    if milestone:
        reasons.append("milestone")
    if cold_start:
        reasons.append("cold_start")
    return reasons


# ---------------------------------------------------------------- tách history


def split_history(messages: list[BaseMessage], keep_last: int = 6):
    """Trả (old, tail). tail luôn bắt đầu bằng HumanMessage và có ít nhất keep_last message nếu có thể.

    Nếu không lượt nào đủ keep_last message thì giữ đúng lượt chat cuối. Không có gì cũ để tóm tắt: ([], messages).
    """
    human = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    eligible = [i for i in human if i > 0 and len(messages) - i >= keep_last]
    if eligible:
        cut = eligible[-1]
    elif human and human[-1] > 0:
        cut = human[-1]
    else:
        return [], messages
    return messages[:cut], messages[cut:]


# ---------------------------------------------------------------- bằng chứng lỗi


def _one_line(text: str) -> str:
    return " ".join((text or "").split())


def _tool_text(message: ToolMessage) -> str:
    """Bóc JSON của read_file/run_code lấy phần content hoặc stdout; tool result lỗi (ok=false) là nhiễu nên bỏ qua."""
    text = message.text or ""
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return text
    if isinstance(payload, dict):
        if isinstance(payload.get("content"), str):
            return payload["content"]
        if isinstance(payload.get("stdout"), str):
            return payload["stdout"]
        if payload.get("ok") is False:
            return ""
    return text


def _template(line: str) -> str:
    key = _DATETIME.sub("<T>", line)
    key = _TIME.sub("<T>", key)
    key = _VOLATILE_ID.sub(lambda m: f"{m.group(1)}=<ID>", key)
    return re.sub(r"\d+", "#", key)


def _add_group(groups: dict, text: str, count: int, first: str | None, last: str | None) -> None:
    key = _template(text)
    group = groups.get(key)
    if group is None:
        groups[key] = {"text": text, "count": count, "first": first, "last": last}
        return
    group["count"] += count
    group["first"] = group["first"] or first
    group["last"] = last or group["last"]


def _collect_evidence(texts: list[str], groups: dict) -> None:
    seen_lines: set[str] = set()
    for text in texts:
        for raw in text.splitlines():
            line = _LINE_PREFIX.sub("", raw.strip())
            if not line or line.startswith("[") or not ERROR_RE.search(line) or line in seen_lines:
                continue
            seen_lines.add(line)  # hai lần đọc chồng nhau không bị đếm đôi
            time = _TIME.search(line)
            stamp = time.group(1) if time else None
            _add_group(groups, line[:EVIDENCE_CHARS], 1, stamp, stamp)


def _parse_evidence_line(line: str) -> tuple[str, int, str | None, str | None]:
    match = _EV_SUFFIX.search(line)
    if not match:
        return line, 1, None, None
    return line[: match.start()], int(match.group(1)), match.group(2), match.group(3)


def _format_evidence(groups: dict) -> list[str]:
    ranked = sorted(enumerate(groups.values()), key=lambda p: (not SEVERE_RE.search(p[1]["text"]), p[0]))
    lines = []
    for _, g in ranked[:MAX_EVIDENCE]:
        suffix = ""
        if g["count"] > 1:
            suffix = f" (×{g['count']}; {g['first']}→{g['last']})" if g["first"] and g["last"] else f" (×{g['count']})"
        lines.append(g["text"] + suffix)
    return lines


# ---------------------------------------------------------------- tóm tắt cũ (để cộng dồn khi compact lần 2, 3, ...)


def _parse_summary(text: str) -> dict[str, list[str]]:
    sections = {"user": [], "answer": [], "evidence": []}
    current = None
    for line in text[len(SUMMARY_PREFIX):].splitlines():
        if line.startswith("## "):
            current = ("user" if line.startswith("## Mục tiêu") else
                       "answer" if line.startswith("## Kết luận") else
                       "evidence" if line.startswith("## Bằng chứng") else None)
        elif current and line.startswith("- "):
            sections[current].append(line[2:])
    return sections


def _is_summary(message: BaseMessage) -> bool:
    return isinstance(message, HumanMessage) and (message.text or "").startswith(SUMMARY_PREFIX)


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


# ---------------------------------------------------------------- compact


def _compact(messages: list[BaseMessage], keep_last: int, before: int):
    old, tail = split_history(messages, keep_last)
    if not old:
        return messages, {"compacted": False, "reason": "chưa có lượt chat cũ để compact", "before_chars": before}
    if all(_is_summary(m) for m in old):
        return messages, {"compacted": False, "reason": "phần cũ chỉ là bản tóm tắt trước đó", "before_chars": before}

    users, answers, tool_texts, groups = [], [], [], {}
    for m in old:
        if _is_summary(m):
            prev = _parse_summary(m.text)
            users += prev["user"]
            answers += prev["answer"]
            for line in prev["evidence"]:
                _add_group(groups, *_parse_evidence_line(line))
        elif isinstance(m, HumanMessage):
            users.append(_one_line(m.text)[:USER_NOTE_CHARS])
        elif isinstance(m, AIMessage) and m.text.strip():
            answers.append(_one_line(m.text)[:ANSWER_CHARS])
        elif isinstance(m, ToolMessage):
            tool_texts.append(_tool_text(m))
    _collect_evidence(tool_texts, groups)

    bullets = lambda items: "\n".join("- " + s for s in items) or "- (không có)"
    summary = (
        f"{H_USER}\n{bullets(_unique(users)[-MAX_USER_NOTES:])}\n"
        f"{H_ANSWER}\n{bullets(_unique(answers)[-MAX_ANSWERS:])}\n"
        f"{H_EVIDENCE}\n{bullets(_format_evidence(groups))}\n"
        f"{H_NOTE}\nĐã rút gọn lịch sử; nếu cần số liệu đầy đủ, hãy đọc lại file gốc theo đoạn (read_log_time_range).\n"
    )
    new = [HumanMessage(content=SUMMARY_PREFIX + summary)] + tail
    return new, {
        "compacted": True, "before_chars": before, "after_chars": total_chars(new),
        "messages_before": len(messages), "messages_after": len(new),
        "summarized_messages": len(old), "kept_messages": len(tail),
        "evidence_templates": len(groups), "summary": summary,
    }


def compact_without_model(messages: list[BaseMessage], keep_last: int = 6):
    """Compact an toàn: nếu có lỗi bất ngờ thì giữ nguyên history và báo lý do thay vì làm hỏng lượt chat."""
    before = total_chars(messages)
    try:
        return _compact(messages, keep_last, before)
    except Exception as exc:  # noqa: BLE001
        return messages, {"compacted": False, "reason": f"{type(exc).__name__}: {exc}", "before_chars": before}


# Giữ tên cũ cho nơi khác import. Không gọi provider.
def compact_messages(messages: list[BaseMessage], model=None, keep_last: int = 6):
    return compact_without_model(messages, keep_last=keep_last)
