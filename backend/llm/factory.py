from __future__ import annotations

from backend.config import Settings
from backend.llm.base import LLMProvider
from backend.llm.gemini_provider import GeminiProvider
from backend.llm.mock_provider import MockProvider
from backend.llm.openrouter_provider import OpenRouterProvider


def get_orchestrator_provider(settings: Settings) -> LLMProvider:
    provider = settings.orchestrator_provider.lower()
    if provider == "gemini":
        return GeminiProvider(
            api_key=settings.orchestrator_api_key,
            model=settings.orchestrator_model,
            base_url=settings.orchestrator_base_url,
            default_reasoning_effort=settings.orchestrator_reasoning_effort or None,
        )
    if provider == "openrouter":
        return OpenRouterProvider(
            api_key=settings.orchestrator_api_key,
            model=settings.orchestrator_model,
            base_url=settings.orchestrator_base_url,
        )
    if provider == "mock":
        return MockProvider()
    raise ValueError(f"Unsupported orchestrator provider: {settings.orchestrator_provider}")


def get_fea_analyst_provider(settings: Settings) -> LLMProvider:
    """LLM for FEA analyst — uses orchestrator provider/model/key/url."""
    return get_orchestrator_provider(settings)


def get_search_provider(settings: Settings) -> LLMProvider:
    provider = settings.search_provider.lower()
    if provider == "openrouter":
        return OpenRouterProvider(
            api_key=settings.search_api_key,
            model=settings.search_model,
            base_url=settings.search_base_url,
        )
    if provider == "gemini":
        return GeminiProvider(
            api_key=settings.search_api_key,
            model=settings.search_model,
            base_url=settings.search_base_url,
            default_reasoning_effort=settings.search_reasoning_effort or None,
        )
    if provider == "mock":
        return MockProvider()
    raise ValueError(f"Unsupported search provider: {settings.search_provider}")
