from __future__ import annotations

import json

from backend.llm.base import LLMProvider


class MockProvider(LLMProvider):
    provider_name = "mock"

    @property
    def available(self) -> bool:
        return True

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> str:
        prompt = f"{system_prompt}\n{user_prompt}".lower()

        if "###task:plan###" in prompt:
            mode = "retrieval_only"
            tools: list[str] = []
            if any(token in prompt for token in ["resistance", "given", "check", "m_ed", "n_ed"]):
                mode = "hybrid"
                tools = ["section_classification_ec3", "member_resistance_ec3"]
            if "interaction" in prompt or "combined" in prompt:
                if "interaction_check_ec3" not in tools:
                    tools.append("interaction_check_ec3")
            return json.dumps(
                {
                    "mode": mode,
                    "tools": tools,
                    "rationale": "Mock deterministic plan.",
                }
            )

        if "###task:refine###" in prompt:
            return json.dumps(["ec3 section classification", "en 1993-1-1 bending resistance"])

        if "###task:answer###" in prompt:
            return "Grounded summary: The response is based only on retrieved EC3 clauses and tool outputs."

        return "Mock response."
