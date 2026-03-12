"""LLM-scored search over the engineering tool registry.

Primary path: shows the full catalogue to an LLM and asks it to score
each tool for relevance to the query (0-10).

Fallback path: keyword-based token overlap scoring (used when no LLM
provider is available or the LLM call fails).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from backend.eurocodepy.registry import ENGINEERING_TOOL_REGISTRY, EngToolEntry

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────


def search_engineering_tools(
    query: str,
    category: str | None = None,
    max_results: int = 8,
    llm_provider: Any | None = None,
) -> list[dict[str, Any]]:
    """Search the registry for tools relevant to *query*.

    If *llm_provider* is supplied (and available), uses LLM-scored
    selection.  Falls back to keyword scoring on failure or when no
    provider is given.
    """
    if not query or not query.strip():
        return []

    # Filter by category first (applies to both paths)
    entries = [
        e for e in ENGINEERING_TOOL_REGISTRY
        if not category or e.category.upper() == category.upper()
    ]
    if not entries:
        return []

    # ── Primary: LLM-scored selection ─────────────────────────────
    if llm_provider is not None:
        try:
            scored = _llm_score_tools(query, entries, llm_provider)
            if scored:
                return _format_results(scored, max_results)
        except Exception:
            logger.warning(
                "LLM tool scoring failed, falling back to keyword search",
                exc_info=True,
            )

    # ── Fallback: keyword scoring ─────────────────────────────────
    scored = _keyword_score_tools(query, entries)
    return _format_results(scored, max_results)


def list_categories() -> list[dict[str, Any]]:
    """Return available categories with tool counts."""
    counts: dict[str, int] = {}
    for entry in ENGINEERING_TOOL_REGISTRY:
        counts[entry.category] = counts.get(entry.category, 0) + 1
    return [{"category": k, "tool_count": v} for k, v in sorted(counts.items())]


# ── LLM scoring ──────────────────────────────────────────────────────

_TOOL_SCORE_SYSTEM = (
    "You are selecting engineering calculation tools for a structural "
    "engineer's query. Score each tool's relevance to the query (0-10).\n"
    "10 = directly answers the query with the right calculation.\n"
    "7-9 = highly relevant, likely needed as part of the workflow.\n"
    "4-6 = possibly useful as supporting data (e.g. a lookup the "
    "calculation needs).\n"
    "1-3 = tangentially related.\n"
    "0 = not relevant.\n\n"
    "Return JSON only: {\"tool_name\": score, ...}  No other text."
)


def _build_catalogue(entries: list[EngToolEntry]) -> str:
    """Build a compact catalogue string for the LLM prompt."""
    lines: list[str] = []
    for i, e in enumerate(entries, 1):
        refs = f" (refs: {', '.join(e.clause_references)})" if e.clause_references else ""
        lines.append(f"{i}. {e.name} — {e.description}{refs}")
    return "\n".join(lines)


def _llm_score_tools(
    query: str,
    entries: list[EngToolEntry],
    llm_provider: Any,
) -> list[tuple[float, EngToolEntry]]:
    """Score tools via LLM and return entries with score > 3."""
    from backend.utils.json_utils import parse_json_loose

    catalogue = _build_catalogue(entries)
    user_prompt = (
        "###TASK:TOOL_SELECTION###\n"
        f"Query: {query}\n\n"
        f"Available tools:\n{catalogue}\n\n"
        "Score each tool. Return JSON only."
    )

    raw = llm_provider.generate(
        system_prompt=_TOOL_SCORE_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=400,
    )

    data = parse_json_loose(raw)
    if not isinstance(data, dict):
        logger.warning("LLM returned non-dict for tool scores: %s", type(data))
        return []

    # Map scores back to entries
    entry_map = {e.name: e for e in entries}
    scored: list[tuple[float, EngToolEntry]] = []
    for tool_name, score in data.items():
        entry = entry_map.get(tool_name)
        if entry is None:
            continue
        try:
            s = float(score)
        except (TypeError, ValueError):
            continue
        if s > 3:
            scored.append((s, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


# ── Keyword scoring (fallback) ───────────────────────────────────────


def _keyword_score_tools(
    query: str,
    entries: list[EngToolEntry],
) -> list[tuple[float, EngToolEntry]]:
    """Score tools by keyword token overlap (fallback path)."""
    query_tokens = set(re.findall(r"\w+", query.lower()))
    if not query_tokens:
        return []

    scored: list[tuple[float, EngToolEntry]] = []
    for entry in entries:
        keyword_set = {kw.lower() for kw in entry.keywords}
        desc_tokens = set(re.findall(r"\w+", entry.description.lower()))
        name_tokens = set(re.findall(r"\w+", entry.name.lower()))

        keyword_hits = len(query_tokens & keyword_set)
        desc_hits = len(query_tokens & desc_tokens)
        name_hits = len(query_tokens & name_tokens)

        score = keyword_hits * 3 + name_hits * 2 + desc_hits
        if score > 0:
            scored.append((float(score), entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


# ── Shared formatting ────────────────────────────────────────────────


def _format_results(
    scored: list[tuple[float, EngToolEntry]],
    max_results: int,
) -> list[dict[str, Any]]:
    """Convert scored entries to the result dict format."""
    results: list[dict[str, Any]] = []
    for score, entry in scored[:max_results]:
        results.append({
            "tool_name": entry.name,
            "category": entry.category,
            "subcategory": entry.subcategory,
            "description": entry.description,
            "parameters": entry.parameters,
            "clause_references": entry.clause_references,
        })
    return results
