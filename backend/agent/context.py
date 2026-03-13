"""Context management — token estimation, usage tracking, and auto-compaction.

Ported from cursor_ivan pattern: session-level context tracking with a UI
indicator (circle) showing how much context is left.  Compaction only triggers
when the whole session context approaches the model's context window — NOT
per-query within a single agent loop.

Key concepts:
  - ``estimate_tokens`` — hybrid char+word+line heuristic (more accurate than
    chars/4 alone)
  - ``context_usage_snapshot`` — returns a dict the frontend renders as the
    context-usage circle (percent, level, tokens, needs_compaction)
  - ``compact_if_needed`` — session-level compaction; when triggered, uses
    semantic compression for tool results (keeps clause IDs, titles, scores)
    and summarises older turns into a single system message
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

# ── Token estimation constants ────────────────────────────────────────
# Hybrid heuristic: chars + words + newlines.  More accurate than chars/4
# alone, especially for JSON-heavy tool results.
CHAR_PER_TOKEN = 4.0
WORD_BONUS_RATIO = 0.18          # each word adds ~0.18 extra tokens
LINE_BONUS = 0.25                # each newline adds ~0.25 extra tokens
MESSAGE_OVERHEAD = 6             # per-message framing overhead
SYSTEM_OVERHEAD = 16             # system prompt framing overhead

# ── Compaction settings ───────────────────────────────────────────────
COMPACTION_TRIGGER_RATIO = 0.85  # trigger at 85% of context window
COMPACTION_TARGET_RATIO = 0.55   # compact down to ~55%
MIN_KEEP_MESSAGES = 6            # always keep last N messages at full fidelity
MAX_SUMMARY_CHARS = 6500         # max length of compaction summary
SUMMARY_PREFIX = "Conversation memory (auto-compacted):"
_TOOL_CONTEXT_BLOCK_RE = re.compile(
    r"\s*(<tool-context>[\s\S]*?</tool-context>)\s*$",
    re.IGNORECASE,
)
_RAW_TOOL_MARKER_RE = re.compile(
    r"(?:^|\n)\s*(?=(?:<tool-context>|\[tool_call\]|\[tool_result\]))",
    re.IGNORECASE,
)


# ── Token estimation ──────────────────────────────────────────────────


def estimate_tokens(text: str) -> int:
    """Hybrid token estimate using chars, words, and line breaks."""
    if not text:
        return 0
    chars = len(text)
    words = len(re.findall(r"\S+", text))
    lines = text.count("\n")
    estimate = (chars / CHAR_PER_TOKEN) + (words * WORD_BONUS_RATIO) + (lines * LINE_BONUS)
    return max(1, int(estimate))


def estimate_content_tokens(content: Any) -> int:
    """Estimate tokens for message content (string, list of parts, or dict)."""
    if content is None:
        return 0
    if isinstance(content, str):
        return estimate_tokens(content)
    if isinstance(content, list):
        return sum(estimate_content_tokens(part) for part in content)
    if isinstance(content, dict):
        kind = str(content.get("type", "")).lower()
        if kind == "text":
            return estimate_tokens(content.get("text", ""))
        if kind == "image_url":
            return 260  # base cost for image
        # Fallback: estimate all text-like fields
        total = 0
        for key in ("text", "content", "arguments", "reasoning"):
            if key in content:
                total += estimate_tokens(str(content[key]))
        return total or estimate_tokens(json.dumps(content, ensure_ascii=False))
    return estimate_tokens(str(content))


def estimate_message_tokens(message: dict) -> int:
    """Estimate tokens for a single message including content + tool calls."""
    if not isinstance(message, dict):
        return 0
    tokens = MESSAGE_OVERHEAD
    content = message.get("content")
    tokens += estimate_content_tokens(content)
    if message.get("tool_calls"):
        tokens += estimate_tokens(json.dumps(message["tool_calls"], default=str))
    if message.get("tool_call_id"):
        tokens += 6
    return tokens


def estimate_messages_tokens(messages: list[dict], system_prompt: str) -> int:
    """Estimate total tokens for a full messages payload."""
    total = estimate_tokens(system_prompt) + SYSTEM_OVERHEAD
    for msg in messages:
        total += estimate_message_tokens(msg)
    return total


# ── Context usage snapshot (for frontend circle indicator) ────────────


def context_usage_snapshot(
    messages: list[dict],
    system_prompt: str,
    context_window: int = 1_000_000,
) -> dict[str, Any]:
    """Return a snapshot of current context usage.

    The frontend renders this as a circle indicator showing:
      - Percentage used / remaining
      - Token counts
      - Color level (low/medium/high/critical)
      - Whether compaction is needed

    Returns:
        dict with keys: estimated_tokens, context_window, used_percent,
        tokens_left, level, needs_compaction
    """
    estimated = estimate_messages_tokens(messages, system_prompt)
    tokens_left = max(0, context_window - estimated)
    used_ratio = (estimated / context_window) if context_window else 0.0
    used_percent = round(used_ratio * 100.0, 1)

    if used_ratio >= 0.95:
        level = "critical"
    elif used_ratio >= 0.85:
        level = "high"
    elif used_ratio >= 0.65:
        level = "medium"
    else:
        level = "low"

    trigger = int(context_window * COMPACTION_TRIGGER_RATIO)

    return {
        "estimated_tokens": int(estimated),
        "context_window": int(context_window),
        "tokens_left": int(tokens_left),
        "used_percent": used_percent,
        "level": level,
        "needs_compaction": estimated >= trigger,
    }


# ── Semantic compression for tool results ─────────────────────────────
# Used during session-level compaction: parses JSON tool results and
# strips verbose fields while keeping IDs, titles, scores, and outputs.


def _semantic_compress_tool_content(content: str) -> str:
    """Semantically compress a tool result for session compaction.

    Preserves clause IDs, titles, scores, calculation outputs.
    Drops full text bodies, verbose HTML, raw content.
    """
    # Strip system-reminder suffixes before parsing
    json_part = content.split("\n\n<system-reminder>")[0].strip()

    try:
        data = json.loads(json_part)
    except (json.JSONDecodeError, TypeError):
        # Not JSON — truncate with generous limit
        return content[:600] if len(content) > 600 else content

    if not isinstance(data, dict):
        s = json_part[:600] if len(json_part) > 600 else json_part
        return s

    # Search results: keep clause_id, title, standard, score — drop text
    if "clauses" in data and "total_found" in data:
        compressed = [{
            "clause_id": c.get("clause_id", ""),
            "title": c.get("clause_title", c.get("title", "")),
            "standard": c.get("standard", ""),
            "score": c.get("score", 0),
        } for c in data.get("clauses", [])]
        result: dict[str, Any] = {"clauses": compressed, "total_found": data["total_found"]}
        if "_referenced_but_not_retrieved" in data:
            result["_referenced_but_not_retrieved"] = data["_referenced_but_not_retrieved"]
        return json.dumps(result)

    # Read clause results: keep ID + title + short text
    if "clauses" in data and "total_found" not in data:
        compressed_c = [{
            "clause_id": c.get("clause_id", ""),
            "title": c.get("clause_title", c.get("title", "")),
            "standard": c.get("standard", ""),
            "text": c.get("text", "")[:300] + ("..." if len(c.get("text", "")) > 300 else ""),
            **({"cross_references": c["cross_references"]} if "cross_references" in c else {}),
        } for c in data.get("clauses", [])]
        return json.dumps({"clauses": compressed_c})

    # Math calculator: keep inputs + outputs
    if "outputs" in data and "inputs_used" in data:
        result_m: dict[str, Any] = {"inputs_used": data["inputs_used"], "outputs": data["outputs"]}
        if data.get("notes"):
            result_m["notes"] = data["notes"]
        return json.dumps(result_m)

    # Errors: keep intact
    if "error" in data:
        return json.dumps(data)

    # Plan/todo: keep intact (small)
    if "plan" in data:
        return json.dumps(data)

    # Fallback
    serialized = json.dumps(data)
    return serialized[:600] if len(serialized) > 600 else serialized


# ── Session-level compaction ──────────────────────────────────────────


def _message_summary_line(msg: dict) -> str:
    """Build a one-line summary of a message for the compaction summary."""
    role = msg.get("role", "unknown")
    content = msg.get("content")

    if role == "tool":
        compressed = _semantic_compress_tool_content(str(content or ""))
        # Further compress for the summary line
        if len(compressed) > 400:
            compressed = compressed[:400] + "..."
        return f"- Tool: {compressed}"

    if isinstance(content, str) and content.strip():
        text = re.sub(r"\s+", " ", content).strip()
        return f"- {role.title()}: {text[:400]}"

    return ""


def compact_if_needed(
    messages: list[dict[str, Any]],
    system_prompt: str,
    context_window: int = 1_000_000,
) -> list[dict[str, Any]]:
    """Auto-compact old messages when session context budget is exceeded.

    Only triggers when the whole session approaches the context window limit.
    Keeps recent messages at full fidelity, compresses older tool results
    semantically, and summarises older turns into a single system message.
    """
    estimated = estimate_messages_tokens(messages, system_prompt)
    trigger = int(context_window * COMPACTION_TRIGGER_RATIO)

    if estimated < trigger or len(messages) <= MIN_KEEP_MESSAGES:
        return messages

    keep = min(MIN_KEEP_MESSAGES, len(messages) - 1)
    old = messages[:-keep]
    recent = messages[-keep:]

    # Build a structured summary of older messages
    lines = [
        SUMMARY_PREFIX,
        "Keep this context in mind when answering future turns.",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
    ]

    for msg in old:
        line = _message_summary_line(msg)
        if not line:
            continue
        # Check we don't exceed max summary size
        candidate = "\n".join(lines + [line])
        if len(candidate) > MAX_SUMMARY_CHARS:
            break
        lines.append(line)

    if len(lines) <= 3:
        lines.append("- Earlier turns were compacted to preserve context budget.")

    summary = "\n".join(lines)
    if len(summary) > MAX_SUMMARY_CHARS:
        summary = summary[:MAX_SUMMARY_CHARS - 1].rstrip() + "…"

    return [{"role": "user", "content": summary}] + recent


# ── Frontend history conversion ───────────────────────────────────────


def split_visible_and_tool_context(content: Any) -> tuple[str, str]:
    """Split assistant content into visible text and hidden tool context."""
    if content is None:
        return "", ""
    if not isinstance(content, str):
        content = str(content)
    match = _TOOL_CONTEXT_BLOCK_RE.search(content)
    if match:
        visible = content[:match.start()].rstrip()
        tool_context = match.group(1).strip()
        return visible, tool_context
    raw_marker = _RAW_TOOL_MARKER_RE.search(content)
    if raw_marker:
        visible = content[:raw_marker.start()].rstrip()
        tool_context = content[raw_marker.start():].strip()
        return visible, tool_context
    return content, ""


def _message_response_payload(message: Any) -> dict[str, Any]:
    if hasattr(message, "response_payload"):
        payload = getattr(message, "response_payload")
    elif isinstance(message, dict):
        payload = message.get("response_payload")
        if payload is None:
            payload = message.get("responsePayload")
    else:
        payload = getattr(message, "responsePayload", None)
    return payload if isinstance(payload, dict) else {}


def extract_assistant_session_memory(message: Any) -> dict[str, Any]:
    """Extract visible content and hidden session memory from an assistant turn."""
    role = message.role if hasattr(message, "role") else message.get("role", "")
    raw_content = message.content if hasattr(message, "content") else message.get("content", "")
    visible_content, legacy_tool_context = split_visible_and_tool_context(raw_content)

    payload = _message_response_payload(message)
    session_memory = payload.get("session_memory", {}) if isinstance(payload, dict) else {}
    if not isinstance(session_memory, dict):
        session_memory = {}

    tool_context = str(session_memory.get("tool_context") or legacy_tool_context or "").strip()
    if role == "assistant" and not visible_content and isinstance(payload.get("answer"), str):
        visible_content, leaked_tool_context = split_visible_and_tool_context(payload["answer"])
        if leaked_tool_context and not tool_context:
            tool_context = leaked_tool_context

    state = str(session_memory.get("state") or "").strip() or (
        "waiting_for_user" if "[tool_call] ask_user(" in tool_context else "final"
    )

    ask_user = session_memory.get("ask_user")
    if not isinstance(ask_user, dict):
        ask_user = None

    return {
        "role": role,
        "visible_content": visible_content,
        "tool_context": tool_context,
        "state": state,
        "ask_user": ask_user,
        "response_payload": payload,
    }


def last_assistant_message_waiting_for_user(history: list[Any]) -> bool:
    """True when the latest assistant turn paused on ask_user."""
    for message in reversed(history or []):
        role = message.role if hasattr(message, "role") else message.get("role", "")
        if role != "assistant":
            return False
        memory = extract_assistant_session_memory(message)
        if memory["state"] == "waiting_for_user":
            return True
        tool_context = memory["tool_context"]
        return (
            "[tool_call] ask_user(" in tool_context
            or '"status": "waiting_for_user"' in tool_context
        )
    return False


def should_continue_from_ask_user(history: list[Any], is_ask_user_reply: bool) -> bool:
    """Only resume a paused ask_user run when the client marks it explicitly."""
    return bool(is_ask_user_reply and last_assistant_message_waiting_for_user(history))


def convert_frontend_history(history: list) -> list[dict[str, Any]]:
    """Convert ChatMessage objects from frontend to OpenAI message format.

    Assistant turns may carry hidden session memory in ``response_payload`` or
    legacy ``<tool-context>`` blocks. That hidden memory is appended here so
    the model receives the full pre-compaction session state.
    """
    messages: list[dict[str, Any]] = []
    for msg in history or []:
        role = msg.role if hasattr(msg, "role") else msg.get("role", "user")
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        if role == "assistant":
            memory = extract_assistant_session_memory(msg)
            content = memory["visible_content"]
            if memory["tool_context"]:
                content = (
                    f"{content}\n\n{memory['tool_context']}"
                    if content
                    else memory["tool_context"]
                )
        messages.append({"role": role, "content": content})
    return messages
