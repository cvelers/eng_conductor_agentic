from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable, Type

from pydantic import BaseModel, ValidationError


def run_cli(
    *,
    tool_name: str,
    input_model: Type[BaseModel],
    handler: Callable[[BaseModel], dict[str, Any]],
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=False, default="")
    args = parser.parse_args()

    raw = args.input_json.strip() or sys.stdin.read().strip()
    if not raw:
        _emit_error(
            tool_name,
            "Missing input JSON payload.",
            action="Pass --input-json '{...}' with required fields.",
        )
        sys.exit(1)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        _emit_error(tool_name, f"Invalid JSON: {exc}", action="Send valid JSON object input.")
        sys.exit(1)

    try:
        validated = input_model.model_validate(payload)
    except ValidationError as exc:
        _emit_error(
            tool_name,
            "Input validation failed.",
            details=exc.errors(),
            action="Fix field types/units according to tool registry schema.",
        )
        sys.exit(1)

    try:
        result = handler(validated)
    except Exception as exc:  # noqa: BLE001
        _emit_error(
            tool_name,
            f"Tool execution failed: {exc}",
            action="Check required inputs and units; see tool registry examples.",
        )
        sys.exit(1)

    print(
        json.dumps(
            {
                "tool": tool_name,
                "status": "ok",
                "result": result,
            }
        )
    )


def _emit_error(
    tool_name: str,
    message: str,
    *,
    details: Any | None = None,
    action: str | None = None,
) -> None:
    payload = {
        "tool": tool_name,
        "status": "error",
        "error": {
            "message": message,
            "details": details,
            "action": action,
        },
    }
    print(json.dumps(payload))
