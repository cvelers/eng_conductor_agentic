"""Execute eurocodepy tools by name with standardised output format."""

from __future__ import annotations

import importlib
import json
import logging
from typing import Any

from backend.eurocodepy.registry import TOOL_INDEX

logger = logging.getLogger(__name__)


def execute_engineering_tool(tool_name: str, params: dict[str, Any]) -> str:
    """Run a registered eurocodepy tool and return JSON result.

    Output format matches the existing tool convention::

        {
            "inputs_used": { ... },
            "outputs": { ... },
            "clause_references": [ ... ],
            "notes": [ ... ],
        }
    """
    entry = TOOL_INDEX.get(tool_name)
    if not entry:
        available = sorted(TOOL_INDEX.keys())
        return json.dumps({
            "error": f"Unknown engineering tool: '{tool_name}'",
            "available_tools": available,
            "_hint": "Use search_engineering_tools to find the right tool name.",
        })

    try:
        module = importlib.import_module(entry.handler_module)
        func = getattr(module, entry.handler_function)

        raw = func(**params)

        # Normalise heterogeneous returns to standard dict
        outputs: dict[str, Any]
        if isinstance(raw, dict):
            outputs = raw
        elif isinstance(raw, tuple):
            # Some ec2 functions return (value1, value2, value3)
            outputs = {f"value_{i}": v for i, v in enumerate(raw)}
        elif isinstance(raw, (int, float)):
            outputs = {"result": raw}
        elif hasattr(raw, "__dataclass_fields__"):
            # Dataclass instances (e.g. SectionCheckResult)
            import dataclasses
            outputs = dataclasses.asdict(raw)
        elif hasattr(raw, "__dict__"):
            # Other objects with attributes
            outputs = {
                k: v for k, v in vars(raw).items()
                if not k.startswith("_")
            }
        else:
            outputs = {"result": str(raw)}

        result = {
            "inputs_used": params,
            "outputs": outputs,
            "clause_references": entry.clause_references,
            "notes": [
                f"Computed via eurocodepy — {entry.handler_module}.{entry.handler_function}",
            ],
        }
        return json.dumps(result, default=str)

    except (TypeError, ValueError) as e:
        return json.dumps({
            "error": str(e),
            "tool_name": tool_name,
            "expected_parameters": entry.parameters,
            "_hint": "Check parameter names and types match the schema.",
        })
    except Exception as e:
        logger.exception("Engineering tool %s failed", tool_name)
        return json.dumps({
            "error": f"{tool_name} failed: {e}",
            "tool_name": tool_name,
        })
