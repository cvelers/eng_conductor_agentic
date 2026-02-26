from __future__ import annotations

from backend.llm.openai_compat import OpenAICompatProvider


class GeminiProvider(OpenAICompatProvider):
    def __init__(self, *, api_key: str, model: str, base_url: str) -> None:
        super().__init__(
            provider_name="gemini",
            base_url=base_url,
            api_key=api_key,
            model=model,
        )
