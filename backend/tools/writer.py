from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from backend.llm.base import LLMProvider
from backend.registries.tool_registry import ToolRegistryEntry
from backend.retrieval.agentic_search import AgenticRetriever

logger = logging.getLogger(__name__)

TOOL_WRITER_SYSTEM = """You are an expert MCP tool writer for a Eurocodes engineering chatbot.
You write Python MCP calculator tools that follow this exact pattern:

```python
from __future__ import annotations
from pydantic import BaseModel, Field, PositiveFloat
from tools.mcp.cli import run_cli

TOOL_NAME = "tool_name_here"

class ToolInput(BaseModel):
    # input fields with types and descriptions

def calculate(inp: ToolInput) -> dict:
    # computation logic
    return {
        "inputs_used": { ... },
        "intermediate": { ... },
        "outputs": { ... },
        "clause_references": [
            {"doc_id": "ec3.en1993-1-1.2005", "clause_id": "...", "title": "...", "pointer": "..."},
        ],
        "notes": ["..."],
    }

if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=ToolInput, handler=calculate)
```

RULES:
- You MUST only use formulas/rules from the provided clause evidence. Do NOT invent formulas.
- Use Pydantic for input validation with proper types and descriptions.
- All outputs must include clause_references pointing to the source clauses.
- Return ONLY the Python code, no explanation. The code must be complete and runnable.
- Use clear variable names matching engineering notation (fy, fu, Wpl, etc.).
- Include proper error handling for invalid inputs.
- All numerical outputs should be rounded appropriately.
"""


class ToolWriter:
    def __init__(
        self,
        *,
        llm: LLMProvider,
        retriever: AgenticRetriever,
        tool_registry: dict[str, ToolRegistryEntry],
        project_root: Path,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.tool_registry = tool_registry
        self.project_root = project_root

    def generate(self, description: str) -> dict[str, Any]:
        clauses = self.retriever.retrieve(description, top_k=6)

        evidence_text = "\n".join(
            f"[{c.clause.doc_id} | {c.clause.clause_id}] {c.clause.clause_title}: {c.clause.text[:300]}"
            for c in clauses
        )

        existing_tools = ", ".join(sorted(self.tool_registry.keys()))

        prompt = (
            "###TASK:GENERATE_TOOL###\n"
            f"Tool description: {description}\n\n"
            f"Existing tools (do not duplicate): {existing_tools}\n\n"
            f"Relevant clause evidence from the database:\n{evidence_text}\n\n"
            "Generate a complete Python MCP tool following the pattern. "
            "Base ALL formulas on the clause evidence above. "
            "Return ONLY the Python code."
        )

        if not self.llm.available:
            return {
                "status": "error",
                "error": "LLM provider not available. Set ORCHESTRATOR_API_KEY.",
            }

        try:
            raw = self.llm.generate(
                system_prompt=TOOL_WRITER_SYSTEM,
                user_prompt=prompt,
                temperature=0,
                max_tokens=2000,
            )
        except Exception as exc:
            logger.warning("tool_writer_failed", extra={"error": str(exc)})
            return {"status": "error", "error": str(exc)}

        code = self._extract_code(raw)

        tool_name = self._extract_tool_name(code)

        return {
            "status": "ok",
            "tool_name": tool_name,
            "code": code,
            "clauses_used": [
                {
                    "doc_id": c.clause.doc_id,
                    "clause_id": c.clause.clause_id,
                    "title": c.clause.clause_title,
                }
                for c in clauses[:4]
            ],
            "note": "Review the generated code before saving. The tool is NOT auto-registered for safety.",
        }

    def _extract_code(self, raw: str) -> str:
        match = re.search(r"```python\s*\n(.*?)```", raw, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"```\s*\n(.*?)```", raw, re.DOTALL)
        if match:
            return match.group(1).strip()
        return raw.strip()

    def _extract_tool_name(self, code: str) -> str:
        match = re.search(r'TOOL_NAME\s*=\s*["\']([^"\']+)["\']', code)
        return match.group(1) if match else "unnamed_tool"
