from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    provider_name: str

    @property
    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        reasoning_effort: str | None = None,
    ) -> str:
        raise NotImplementedError

    def generate_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 8000,
        reasoning_effort: str | None = None,
    ) -> str:
        """Generate from a full messages list (multi-turn).

        Default implementation extracts system + user content and
        delegates to the single-turn ``generate()`` method.
        """
        system = next(
            (m["content"] for m in messages if m.get("role") == "system"), ""
        )
        user_parts = [
            m["content"]
            for m in messages
            if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
        ]
        return self.generate(
            system_prompt=system,
            user_prompt="\n".join(user_parts),
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

    def generate_multimodal(
        self,
        *,
        system_prompt: str,
        content_parts: list[dict[str, Any]],
        temperature: float = 0.3,
        max_tokens: int = 4000,
        reasoning_effort: str | None = None,
    ) -> str:
        """Generate with multimodal content (text + images).

        Default implementation falls back to text-only by extracting text parts.
        Providers with vision support should override this.
        """
        text_parts = [p["text"] for p in content_parts if p.get("type") == "text"]
        return self.generate(
            system_prompt=system_prompt,
            user_prompt="\n".join(text_parts),
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
