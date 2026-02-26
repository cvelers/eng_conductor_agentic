from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ToolRegistryEntry(BaseModel):
    tool_name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    script_path: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    units: dict[str, str] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    examples: list[dict[str, Any]] = Field(default_factory=list)


def load_tool_registry(path: Path) -> list[ToolRegistryEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Tool registry must be a list.")
    return [ToolRegistryEntry.model_validate(item) for item in payload]
