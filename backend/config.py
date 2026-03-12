from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path

logger = logging.getLogger(__name__)


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


def _load_cognitive_config(project_root: Path) -> dict:
    """Load LLM cognitive settings from cognitive_config.json."""
    config_path = project_root / "cognitive_config.json"
    if not config_path.exists():
        logger.warning("cognitive_config.json not found at %s, using defaults", config_path)
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to load cognitive_config.json")
        return {}


@dataclass(frozen=True)
class Settings:
    project_root: Path
    log_level: str = "INFO"

    # ── Main LLM (agent loop) ────────────────────────────────────────
    # Defaults match cognitive_config.json — that file is the single
    # source of truth; these are fallbacks if the file is missing.
    orchestrator_provider: str = "gemini"
    orchestrator_model: str = "gemini-3.1-flash-lite-preview"
    orchestrator_api_key: str = ""
    orchestrator_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    orchestrator_reasoning_effort: str = "high"

    # Agent loop parameters
    agent_temperature: float = 0.2
    agent_max_tokens: int = 32000
    agent_max_rounds: int = 25
    agent_context_window: int = 1_000_000

    # ── Search provider (for retriever reranking + eng tool selection)
    search_provider: str = "gemini"
    search_model: str = "gemini-3.1-flash-lite-preview"
    search_api_key: str = ""
    search_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    search_decompose_max_tokens: int = 4000
    search_decompose_reasoning_effort: str = "high"
    search_reasoning_effort: str = ""

    # ── Retrieval settings ───────────────────────────────────────────
    agentic_search_enabled: bool = True
    recursive_retrieval_enabled: bool = False
    embeddings_enabled: bool = False
    max_retrieval_iters: int = 3
    top_k_clauses: int = 20

    # Retriever LLM params (used internally by AgenticRetriever)
    rerank_temperature: float = 0.2
    rerank_max_tokens: int = 2000
    rerank_reasoning_effort: str = ""
    gap_analysis_temperature: float = 0.2
    gap_analysis_max_tokens: int = 2000
    gap_analysis_reasoning_effort: str = ""

    # ── FEA analyst (separate mode) ──────────────────────────────────
    fea_analyst_temperature: float = 0.2
    fea_analyst_max_tokens: int = 16000
    fea_analyst_reasoning_effort: str = ""

    # ── Grounding validator ───────────────────────────────────────────
    validator_enabled: bool = True
    validator_provider: str = "gemini"
    validator_model: str = "gemini-3.1-flash-lite-preview"
    validator_api_key: str = ""
    validator_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    validator_temperature: float = 0.0
    validator_max_tokens: int = 2000
    validator_reasoning_effort: str = ""

    # ── Paths ────────────────────────────────────────────────────────
    document_registry_path: Path | None = None
    orchestrator_thread_log_path: Path | None = None

    # ── Auth ─────────────────────────────────────────────────────────
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
    def resolved_orchestrator_thread_log_path(self) -> Path:
        if self.orchestrator_thread_log_path:
            return self.orchestrator_thread_log_path
        return self.project_root / "logs" / "orchestrator_threads.json"

    @classmethod
    def load(cls) -> "Settings":
        project_root = Path(__file__).resolve().parents[1]

        # ── Load cognitive config (single source of truth for LLM params) ──
        cc = _load_cognitive_config(project_root)
        orch = cc.get("orchestrator", {})
        search = cc.get("search", {})
        decompose = search.get("decompose", {})
        rerank = search.get("rerank", {})
        gap = search.get("gap_analysis", {})
        fea = cc.get("fea_analyst", {})
        validator = cc.get("validator", {})

        return cls(
            project_root=project_root,
            log_level=os.getenv("LOG_LEVEL", "INFO"),

            # ── Orchestrator (from cognitive_config.json) ──
            orchestrator_provider=orch.get("provider", "gemini"),
            orchestrator_model=orch.get("model", "gemini-3.1-flash-lite-preview"),
            orchestrator_api_key=os.getenv("ORCHESTRATOR_API_KEY", ""),
            orchestrator_base_url=orch.get(
                "base_url",
                "https://generativelanguage.googleapis.com/v1beta/openai",
            ),
            orchestrator_reasoning_effort=orch.get("reasoning_effort", "high"),
            agent_temperature=float(orch.get("temperature", 0.2)),
            agent_max_tokens=int(orch.get("max_tokens", 32000)),
            agent_max_rounds=int(orch.get("max_rounds", 25)),
            agent_context_window=int(orch.get("context_window", 1_000_000)),

            # ── Search (from cognitive_config.json) ──
            search_provider=search.get("provider", "gemini"),
            search_model=search.get("model", "gemini-3.1-flash-lite-preview"),
            search_api_key=os.getenv("SEARCH_API_KEY", ""),
            search_base_url=search.get(
                "base_url",
                "https://generativelanguage.googleapis.com/v1beta/openai",
            ),
            search_decompose_max_tokens=int(decompose.get("max_tokens", 4000)),
            search_decompose_reasoning_effort=decompose.get("reasoning_effort", "high"),
            search_reasoning_effort=decompose.get("reasoning_effort", ""),
            rerank_temperature=float(rerank.get("temperature", 0.2)),
            rerank_max_tokens=int(rerank.get("max_tokens", 2000)),
            rerank_reasoning_effort=rerank.get("reasoning_effort", ""),
            gap_analysis_temperature=float(gap.get("temperature", 0.2)),
            gap_analysis_max_tokens=int(gap.get("max_tokens", 2000)),
            gap_analysis_reasoning_effort=gap.get("reasoning_effort", ""),

            # ── Retrieval feature flags (still from env) ──
            agentic_search_enabled=_to_bool(os.getenv("AGENTIC_SEARCH_ENABLED"), True),
            recursive_retrieval_enabled=_to_bool(
                os.getenv("RECURSIVE_RETRIEVAL_ENABLED"), False
            ),
            embeddings_enabled=_to_bool(os.getenv("EMBEDDINGS_ENABLED"), False),
            max_retrieval_iters=_to_int(os.getenv("MAX_RETRIEVAL_ITERS"), 3),
            top_k_clauses=_to_int(os.getenv("TOP_K_CLAUSES"), 8),

            # ── FEA analyst (from cognitive_config.json) ──
            fea_analyst_temperature=float(fea.get("temperature", 0.0)),
            fea_analyst_max_tokens=int(fea.get("max_tokens", 16000)),
            fea_analyst_reasoning_effort=fea.get("reasoning_effort", ""),

            # ── Grounding validator (from cognitive_config.json) ──
            validator_enabled=bool(validator.get("enabled", True)),
            validator_provider=validator.get("provider", "gemini"),
            validator_model=validator.get("model", "gemini-3.1-flash-lite-preview"),
            validator_api_key=os.getenv("VALIDATOR_API_KEY", ""),
            validator_base_url=validator.get(
                "base_url",
                "https://generativelanguage.googleapis.com/v1beta/openai",
            ),
            validator_temperature=float(validator.get("temperature", 0.0)),
            validator_max_tokens=int(validator.get("max_tokens", 2000)),
            validator_reasoning_effort=validator.get("reasoning_effort", ""),

            # ── Paths (from env) ──
            document_registry_path=(
                Path(os.environ["DOCUMENT_REGISTRY_PATH"])
                if os.getenv("DOCUMENT_REGISTRY_PATH")
                else None
            ),
            orchestrator_thread_log_path=(
                Path(os.environ["ORCHESTRATOR_THREAD_LOG_PATH"])
                if os.getenv("ORCHESTRATOR_THREAD_LOG_PATH")
                else None
            ),

            # ── Auth (from env) ──
            supabase_url=os.getenv("SUPABASE_URL", ""),
            supabase_anon_key=os.getenv("SUPABASE_ANON_KEY", ""),
            supabase_jwt_secret=os.getenv("SUPABASE_JWT_SECRET", ""),
            supabase_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
        )

    def with_overrides(self, **kwargs: object) -> "Settings":
        return replace(self, **kwargs)
