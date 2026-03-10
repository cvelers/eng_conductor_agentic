"""Context management — token estimation and auto-compaction."""

from __future__ import annotations

import json
from typing import Any

CHAR_PER_TOKEN = 4.0
MESSAGE_OVERHEAD = 6
COMPACTION_TRIGGER_RATIO = 0.85
COMPACTION_TARGET_RATIO = 0.55
MIN_KEEP_MESSAGES = 6


def estimate_tokens(text: str) -> int:
    """Quick char-based token estimate."""
    return max(1, int(len(text) / CHAR_PER_TOKEN))


def estimate_messages_tokens(messages: list[dict], system_prompt: str) -> int:
    """Estimate total tokens for a messages payload."""
    total = estimate_tokens(system_prompt) + 16
    for msg in messages:
        total += MESSAGE_OVERHEAD
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += estimate_tokens(part.get("text", ""))
        if msg.get("tool_calls"):
            total += estimate_tokens(json.dumps(msg["tool_calls"], default=str))
    return total


def compact_if_needed(
    messages: list[dict[str, Any]],
    system_prompt: str,
    context_window: int = 128_000,
) -> list[dict[str, Any]]:
    """Auto-compact old messages when context budget is exceeded.

    Keeps recent messages intact, summarises older ones into a single
    system message so the LLM retains key context without exceeding the window.
    """
    estimated = estimate_messages_tokens(messages, system_prompt)
    trigger = int(context_window * COMPACTION_TRIGGER_RATIO)

    if estimated < trigger or len(messages) <= MIN_KEEP_MESSAGES:
        return messages

    keep = min(MIN_KEEP_MESSAGES, len(messages) - 1)
    old = messages[:-keep]
    recent = messages[-keep:]

    summary_parts = ["[Conversation summary — older messages compacted]\n"]
    for msg in old:
        role = msg.get("role", "")
        content = msg.get("content")
        if role == "tool":
            # Summarise tool results briefly
            snippet = str(content or "")[:150]
            summary_parts.append(f"- Tool result: {snippet}")
        elif isinstance(content, str) and content.strip():
            snippet = content[:250]
            summary_parts.append(f"- {role}: {snippet}")

    summary = "\n".join(summary_parts[:30])
    return [{"role": "user", "content": summary}] + recent


def convert_frontend_history(history: list) -> list[dict[str, Any]]:
    """Convert ChatMessage objects from frontend to OpenAI message format."""
    messages: list[dict[str, Any]] = []
    for msg in history or []:
        role = msg.role if hasattr(msg, "role") else msg.get("role", "user")
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        messages.append({"role": role, "content": content})
    return messages
