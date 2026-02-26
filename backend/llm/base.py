from __future__ import annotations

from abc import ABC, abstractmethod


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
        max_tokens: int = 800,
    ) -> str:
        raise NotImplementedError
