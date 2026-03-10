"""Translate agent loop events → frontend NDJSON format.

The frontend expects specific event shapes — this adapter ensures
backward compatibility without coupling the agent loop to the UI.
"""

from __future__ import annotations

import json
from typing import Any


def adapt_event(event: dict[str, Any]) -> dict[str, Any]:
    """Convert an agent loop event to the frontend-expected NDJSON shape."""
    t = event.get("type", "")

    if t == "delta":
        return {"type": "delta", "delta": event.get("content", "")}

    if t == "thinking":
        return {"type": "thinking", "content": event.get("content", "")}

    if t == "tool_start":
        return {
            "type": "tool_start",
            "tool": event.get("tool", ""),
            "args": event.get("args", {}),
        }

    if t == "tool_result":
        # Parse result JSON if possible for structured display
        raw = event.get("result", "")
        try:
            result_obj = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            result_obj = raw
        return {
            "type": "tool_result",
            "tool": event.get("tool", ""),
            "result": result_obj,
            "status": event.get("status", "ok"),
            "summary": event.get("summary", ""),
        }

    if t == "done":
        return {
            "type": "final",
            "response": {
                "answer": event.get("content", ""),
                "supported": True,
                "sources": [],
            },
        }

    if t == "error":
        return {"type": "error", "detail": event.get("message", "Unknown error")}

    # Pass through unknown events
    return event
