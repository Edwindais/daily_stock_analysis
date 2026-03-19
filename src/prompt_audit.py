# -*- coding: utf-8 -*-
"""
Prompt audit helpers.

Writes each model prompt to ``model_prompts/`` and emits a short INFO-level
summary for easier troubleshooting from the normal runtime log.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_PROMPT_WRITE_LOCK = threading.Lock()
_PROMPT_DIR = Path(__file__).resolve().parents[1] / "model_prompts"
_FILENAME_SAFE = re.compile(r"[^0-9A-Za-z._-]+")


def _normalize_text(value: Any) -> str:
    """Convert arbitrary content into readable text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        return str(value)


def _condense_text(text: str, limit: int = 180) -> str:
    """Collapse whitespace and trim to a concise preview."""
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


def _safe_part(value: Optional[str], fallback: str) -> str:
    """Sanitize a filename component."""
    raw = (value or "").strip()
    if not raw:
        return fallback
    cleaned = _FILENAME_SAFE.sub("_", raw).strip("._-")
    return cleaned or fallback


def _extract_stock_code(messages: Iterable[Dict[str, Any]]) -> str:
    """Best-effort stock-code extraction from prompt content."""
    patterns = (
        r"股票代码[:：]\s*([A-Za-z0-9._-]+)",
        r"\b(?:Analyze|analyze)\s+(?:stock\s+)?([A-Za-z0-9._-]{4,12})\b",
    )
    for msg in messages:
        content = _normalize_text(msg.get("content"))
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1)
    return ""


def _find_message_preview(messages: List[Dict[str, Any]], role: str) -> str:
    """Return the last matching message preview."""
    for msg in reversed(messages):
        if msg.get("role") == role:
            return _condense_text(_normalize_text(msg.get("content")))
    return ""


def _build_summary(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build short prompt summary metadata."""
    total_chars = 0
    role_counts: Dict[str, int] = {}
    for msg in messages:
        role = str(msg.get("role", "unknown"))
        role_counts[role] = role_counts.get(role, 0) + 1
        total_chars += len(_normalize_text(msg.get("content")))
    return {
        "message_count": len(messages),
        "total_chars": total_chars,
        "role_counts": role_counts,
        "system_preview": _find_message_preview(messages, "system"),
        "user_preview": _find_message_preview(messages, "user"),
        "assistant_preview": _find_message_preview(messages, "assistant"),
        "stock_code": _extract_stock_code(messages),
    }


def write_prompt_snapshot(
    *,
    logger: logging.Logger,
    source: str,
    model: str,
    messages: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Persist one prompt snapshot and emit an INFO summary."""
    metadata = metadata or {}
    summary = _build_summary(messages)

    prompt_kind = _safe_part(str(metadata.get("prompt_kind") or source), "prompt")
    stock_code = _safe_part(str(metadata.get("stock_code") or summary["stock_code"]), "")
    phase = _safe_part(str(metadata.get("phase") or ""), "")
    step = metadata.get("step")

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
    filename_parts = [timestamp, prompt_kind]
    if stock_code:
        filename_parts.append(stock_code)
    if phase:
        filename_parts.append(phase)
    if step is not None:
        filename_parts.append(f"step{step}")
    filename = "_".join(part for part in filename_parts if part) + ".txt"

    with _PROMPT_WRITE_LOCK:
        _PROMPT_DIR.mkdir(parents=True, exist_ok=True)
        file_path = _PROMPT_DIR / filename

        lines: List[str] = [
            f"timestamp: {now.isoformat(timespec='seconds')}",
            f"source: {source}",
            f"model: {model or 'unknown'}",
            f"prompt_kind: {prompt_kind}",
            f"stock_code: {metadata.get('stock_code') or summary['stock_code'] or '-'}",
            f"phase: {metadata.get('phase') or '-'}",
            f"step: {step if step is not None else '-'}",
            f"message_count: {summary['message_count']}",
            f"total_chars: {summary['total_chars']}",
            f"role_counts: {json.dumps(summary['role_counts'], ensure_ascii=False, sort_keys=True)}",
            "",
            "=== Prompt Summary ===",
            f"system_preview: {summary['system_preview'] or '-'}",
            f"user_preview: {summary['user_preview'] or '-'}",
            f"assistant_preview: {summary['assistant_preview'] or '-'}",
            "",
            "=== Full Messages ===",
        ]

        for index, msg in enumerate(messages, start=1):
            role = msg.get("role", "unknown")
            lines.append(f"--- Message {index} | role={role} ---")
            content = _normalize_text(msg.get("content"))
            lines.append(content or "<empty>")
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                lines.append("[tool_calls]")
                lines.append(_normalize_text(tool_calls))
            tool_name = msg.get("name")
            if tool_name:
                lines.append(f"[tool_name] {tool_name}")
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id:
                lines.append(f"[tool_call_id] {tool_call_id}")
            lines.append("")

        file_path.write_text("\n".join(lines), encoding="utf-8")

    try:
        relative_path = file_path.relative_to(Path.cwd())
    except ValueError:
        relative_path = file_path

    logger.info(
        "[Prompt摘要] source=%s model=%s stock=%s messages=%d chars=%d file=%s",
        source,
        model or "unknown",
        metadata.get("stock_code") or summary["stock_code"] or "-",
        summary["message_count"],
        summary["total_chars"],
        relative_path,
    )
    if summary["system_preview"]:
        logger.info("[Prompt摘要][system] %s", summary["system_preview"])
    if summary["user_preview"]:
        logger.info("[Prompt摘要][user] %s", summary["user_preview"])
    elif summary["assistant_preview"]:
        logger.info("[Prompt摘要][assistant] %s", summary["assistant_preview"])

    return file_path
