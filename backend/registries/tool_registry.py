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


# The 5 generic tools the orchestrator should see.
# All other hardcoded calculation tools remain in the registry / MCP server
# for direct use, but are hidden from the orchestrator's decompose step.
ORCHESTRATOR_ACTIVE_TOOLS: frozenset[str] = frozenset({
    "math_calculator",
    "section_properties",
    "unit_converter",
    "steel_grade_properties",
    "section_classification_ec3",
})


def load_tool_registry(
    path: Path,
    *,
    active_only: bool = False,
) -> list[ToolRegistryEntry]:
    """Load tool registry entries from JSON.

    When *active_only* is True, only tools in ``ORCHESTRATOR_ACTIVE_TOOLS``
    are returned.  The MCP server passes ``active_only=False`` (default) to
    expose every tool; the orchestrator passes ``active_only=True`` so the
    LLM only sees the 5 generic tools.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Tool registry must be a list.")
    entries = [ToolRegistryEntry.model_validate(item) for item in payload]
    if active_only:
        entries = [e for e in entries if e.tool_name in ORCHESTRATOR_ACTIVE_TOOLS]
    return entries
