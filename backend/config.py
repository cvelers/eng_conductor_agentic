from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _to_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    project_root: Path
    log_level: str = "INFO"

    orchestrator_provider: str = "gemini"
    orchestrator_model: str = "gemini-3.1-pro"
    orchestrator_api_key: str = ""
    orchestrator_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    orchestrator_reasoning_effort: str = ""  # Gemini only: low, medium, high

    search_provider: str = "openrouter"
    search_model: str = "moonshotai/kimi-k2.5"
    search_api_key: str = ""
    search_base_url: str = "https://openrouter.ai/api/v1"
    search_decompose_max_tokens: int = 1200
    search_reasoning_effort: str = ""  # Gemini only: low, medium, high

    tool_writer_provider: str = ""
    tool_writer_model: str = ""
    tool_writer_api_key: str = ""
    tool_writer_base_url: str = ""
    tool_writer_reasoning_effort: str = ""  # Gemini only: low, medium, high

    fea_analyst_provider: str = ""
    fea_analyst_model: str = ""
    fea_analyst_api_key: str = ""
    fea_analyst_base_url: str = ""
    fea_analyst_reasoning_effort: str = ""  # Gemini only: low, medium, high

    agentic_search_enabled: bool = True
    recursive_retrieval_enabled: bool = False
    embeddings_enabled: bool = False
    max_retrieval_iters: int = 3
    top_k_clauses: int = 20

    # Pipeline stage LLM parameters (temperature, max_tokens, reasoning_effort)
    intent_temperature: float = 0.0
    intent_max_tokens: int = 256
    intent_reasoning_effort: str = "low"

    decompose_temperature: float = 0.0
    decompose_max_tokens: int = 2048
    decompose_reasoning_effort: str = "low"

    rerank_temperature: float = 0.0
    rerank_max_tokens: int = 1200
    rerank_reasoning_effort: str = ""

    gap_analysis_temperature: float = 0.0
    gap_analysis_max_tokens: int = 600
    gap_analysis_reasoning_effort: str = ""

    chain_resolve_temperature: float = 0.0
    chain_resolve_max_tokens: int = 2000
    chain_resolve_reasoning_effort: str = ""

    input_resolve_temperature: float = 0.0
    input_resolve_max_tokens: int = 2000
    input_resolve_reasoning_effort: str = ""

    fix_inputs_temperature: float = 0.0
    fix_inputs_max_tokens: int = 1024
    fix_inputs_reasoning_effort: str = "low"

    upstream_resolve_temperature: float = 0.0
    upstream_resolve_max_tokens: int = 2000
    upstream_resolve_reasoning_effort: str = ""

    compose_temperature: float = 0.15
    compose_max_tokens: int = 8000
    compose_reasoning_effort: str = "low"

    document_registry_path: Path | None = None
    tool_registry_path: Path | None = None
    orchestrator_thread_log_path: Path | None = None

    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_jwt_secret: str = ""
    supabase_service_role_key: str = ""

    @property
    def auth_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_anon_key)

    @property
    def resolved_document_registry_path(self) -> Path:
        if self.document_registry_path:
            return self.document_registry_path
        return self.project_root / "data" / "document_registry.json"

    @property
    def resolved_tool_registry_path(self) -> Path:
        if self.tool_registry_path:
            return self.tool_registry_path
        return self.project_root / "tools" / "tool_registry.json"

    @property
    def resolved_orchestrator_thread_log_path(self) -> Path:
        if self.orchestrator_thread_log_path:
            return self.orchestrator_thread_log_path
        return self.project_root / "logs" / "orchestrator_threads.json"

    @classmethod
    def load(cls) -> "Settings":
        project_root = Path(__file__).resolve().parents[1]
        return cls(
            project_root=project_root,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            orchestrator_provider=os.getenv("ORCHESTRATOR_PROVIDER", "gemini"),
            orchestrator_model=os.getenv("ORCHESTRATOR_MODEL", "gemini-3.1-pro"),
            orchestrator_api_key=os.getenv("ORCHESTRATOR_API_KEY", ""),
            orchestrator_base_url=os.getenv(
                "ORCHESTRATOR_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai",
            ),
            orchestrator_reasoning_effort=(
                os.getenv("ORCHESTRATOR_REASONING_EFFORT", "").strip() or ""
            ),
            search_provider=os.getenv("SEARCH_PROVIDER", "openrouter"),
            search_model=os.getenv("SEARCH_MODEL", "moonshotai/kimi-k2.5"),
            search_api_key=os.getenv("SEARCH_API_KEY", ""),
            search_base_url=os.getenv("SEARCH_BASE_URL", "https://openrouter.ai/api/v1"),
            search_decompose_max_tokens=_to_int(
                os.getenv("SEARCH_DECOMPOSE_MAX_TOKENS"), 1200
            ),
            search_reasoning_effort=(
                os.getenv("SEARCH_REASONING_EFFORT", "").strip() or ""
            ),
            tool_writer_provider=os.getenv("TOOL_WRITER_PROVIDER", ""),
            tool_writer_model=os.getenv("TOOL_WRITER_MODEL", ""),
            tool_writer_api_key=os.getenv("TOOL_WRITER_API_KEY", ""),
            tool_writer_base_url=os.getenv("TOOL_WRITER_BASE_URL", ""),
            tool_writer_reasoning_effort=(
                os.getenv("TOOL_WRITER_REASONING_EFFORT", "").strip() or ""
            ),
            fea_analyst_provider=os.getenv("FEA_ANALYST_PROVIDER", ""),
            fea_analyst_model=os.getenv("FEA_ANALYST_MODEL", ""),
            fea_analyst_api_key=os.getenv("FEA_ANALYST_API_KEY", ""),
            fea_analyst_base_url=os.getenv("FEA_ANALYST_BASE_URL", ""),
            fea_analyst_reasoning_effort=(
                os.getenv("FEA_ANALYST_REASONING_EFFORT", "").strip() or ""
            ),
            agentic_search_enabled=_to_bool(os.getenv("AGENTIC_SEARCH_ENABLED"), True),
            recursive_retrieval_enabled=_to_bool(
                os.getenv("RECURSIVE_RETRIEVAL_ENABLED"), False
            ),
            embeddings_enabled=_to_bool(os.getenv("EMBEDDINGS_ENABLED"), False),
            max_retrieval_iters=_to_int(os.getenv("MAX_RETRIEVAL_ITERS"), 3),
            top_k_clauses=_to_int(os.getenv("TOP_K_CLAUSES"), 6),
            # Pipeline stage LLM parameters
            intent_temperature=_to_float(os.getenv("INTENT_TEMPERATURE"), 0.0),
            intent_max_tokens=_to_int(os.getenv("INTENT_MAX_TOKENS"), 256),
            intent_reasoning_effort=(
                os.getenv("INTENT_REASONING_EFFORT", "low").strip() or ""
            ),
            decompose_temperature=_to_float(os.getenv("DECOMPOSE_TEMPERATURE"), 0.0),
            decompose_max_tokens=_to_int(os.getenv("DECOMPOSE_MAX_TOKENS"), 2048),
            decompose_reasoning_effort=(
                os.getenv("DECOMPOSE_REASONING_EFFORT", "low").strip() or ""
            ),
            rerank_temperature=_to_float(os.getenv("RERANK_TEMPERATURE"), 0.0),
            rerank_max_tokens=_to_int(os.getenv("RERANK_MAX_TOKENS"), 1200),
            rerank_reasoning_effort=(
                os.getenv("RERANK_REASONING_EFFORT", "").strip() or ""
            ),
            gap_analysis_temperature=_to_float(os.getenv("GAP_ANALYSIS_TEMPERATURE"), 0.0),
            gap_analysis_max_tokens=_to_int(os.getenv("GAP_ANALYSIS_MAX_TOKENS"), 600),
            gap_analysis_reasoning_effort=(
                os.getenv("GAP_ANALYSIS_REASONING_EFFORT", "").strip() or ""
            ),
            chain_resolve_temperature=_to_float(os.getenv("CHAIN_RESOLVE_TEMPERATURE"), 0.0),
            chain_resolve_max_tokens=_to_int(os.getenv("CHAIN_RESOLVE_MAX_TOKENS"), 2000),
            chain_resolve_reasoning_effort=(
                os.getenv("CHAIN_RESOLVE_REASONING_EFFORT", "").strip() or ""
            ),
            input_resolve_temperature=_to_float(os.getenv("INPUT_RESOLVE_TEMPERATURE"), 0.0),
            input_resolve_max_tokens=_to_int(os.getenv("INPUT_RESOLVE_MAX_TOKENS"), 2000),
            input_resolve_reasoning_effort=(
                os.getenv("INPUT_RESOLVE_REASONING_EFFORT", "").strip() or ""
            ),
            fix_inputs_temperature=_to_float(os.getenv("FIX_INPUTS_TEMPERATURE"), 0.0),
            fix_inputs_max_tokens=_to_int(os.getenv("FIX_INPUTS_MAX_TOKENS"), 1024),
            fix_inputs_reasoning_effort=(
                os.getenv("FIX_INPUTS_REASONING_EFFORT", "low").strip() or ""
            ),
            upstream_resolve_temperature=_to_float(os.getenv("UPSTREAM_RESOLVE_TEMPERATURE"), 0.0),
            upstream_resolve_max_tokens=_to_int(os.getenv("UPSTREAM_RESOLVE_MAX_TOKENS"), 2000),
            upstream_resolve_reasoning_effort=(
                os.getenv("UPSTREAM_RESOLVE_REASONING_EFFORT", "").strip() or ""
            ),
            compose_temperature=_to_float(os.getenv("COMPOSE_TEMPERATURE"), 0.15),
            compose_max_tokens=_to_int(os.getenv("COMPOSE_MAX_TOKENS"), 8000),
            compose_reasoning_effort=(
                os.getenv("COMPOSE_REASONING_EFFORT", "low").strip() or ""
            ),
            document_registry_path=(
                Path(os.environ["DOCUMENT_REGISTRY_PATH"])
                if os.getenv("DOCUMENT_REGISTRY_PATH")
                else None
            ),
            tool_registry_path=(
                Path(os.environ["TOOL_REGISTRY_PATH"])
                if os.getenv("TOOL_REGISTRY_PATH")
                else None
            ),
            orchestrator_thread_log_path=(
                Path(os.environ["ORCHESTRATOR_THREAD_LOG_PATH"])
                if os.getenv("ORCHESTRATOR_THREAD_LOG_PATH")
                else None
            ),
            supabase_url=os.getenv("SUPABASE_URL", ""),
            supabase_anon_key=os.getenv("SUPABASE_ANON_KEY", ""),
            supabase_jwt_secret=os.getenv("SUPABASE_JWT_SECRET", ""),
            supabase_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
        )

    def with_overrides(self, **kwargs: object) -> "Settings":
        return replace(self, **kwargs)
