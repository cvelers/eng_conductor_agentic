from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterator

from backend.config import Settings
from backend.llm.base import LLMProvider
from backend.registries.document_registry import ClauseRecord

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[a-z0-9_\-\.]+")
QUERY_SANITIZE_RE = re.compile(r"[^A-Za-z0-9\s_\-\./]")


@dataclass
class RetrievedClause:
    clause: ClauseRecord
    score: float
    matched_terms: list[str]


class AgenticRetriever:
    def __init__(
        self,
        *,
        settings: Settings,
        search_provider: LLMProvider,
        clauses: list[ClauseRecord],
    ) -> None:
        self.settings = settings
        self.search_provider = search_provider
        self.clauses = clauses

    def retrieve(self, query: str, top_k: int | None = None) -> tuple[list[RetrievedClause], list[dict[str, object]]]:
        final_results: list[RetrievedClause] = []
        trace: list[dict[str, object]] = []
        for event in self.iter_retrieve(query, top_k=top_k):
            if event.get("type") == "final":
                final_results = event.get("results", [])
                trace = event.get("trace", [])
        return final_results, trace

    def iter_retrieve(
        self, query: str, top_k: int | None = None
    ) -> Iterator[dict[str, Any]]:
        limit = top_k or self.settings.top_k_clauses
        safe_query = self._sanitize_query(query)

        queries = [safe_query]
        seen_queries = {safe_query}
        trace: list[dict[str, object]] = []
        aggregated: dict[str, RetrievedClause] = {}

        max_iters = self.settings.max_retrieval_iters if self.settings.agentic_search_enabled else 1
        max_iters = max(1, max_iters)

        for iteration in range(1, max_iters + 1):
            current = queries[-1]
            ranked = self._search_once(current)
            top = ranked[:limit]

            for item in top:
                key = item.clause.citation_address
                if key not in aggregated or item.score > aggregated[key].score:
                    aggregated[key] = item

            step = {
                "iteration": iteration,
                "query": current,
                "top_clause_ids": [f"{x.clause.doc_id}:{x.clause.clause_id}" for x in top],
            }
            trace.append(step)

            logger.info(
                "retrieval_iteration",
                extra={
                    "iteration": iteration,
                    "query": current,
                    "hits": len(top),
                    "top_clause_ids": [x.clause.clause_id for x in top[:3]],
                },
            )
            yield {
                "type": "iteration",
                "step": step,
                "top": [
                    {
                        "doc_id": x.clause.doc_id,
                        "clause_id": x.clause.clause_id,
                        "title": x.clause.clause_title,
                        "score": x.score,
                    }
                    for x in top[:3]
                ],
            }

            if iteration == max_iters:
                break

            refined = self._refine_query(current, top)
            if not refined:
                break
            if refined in seen_queries:
                break
            queries.append(refined)
            seen_queries.add(refined)

        merged = sorted(
            aggregated.values(),
            key=lambda item: (-item.score, item.clause.doc_id, item.clause.clause_id),
        )

        if self.settings.recursive_retrieval_enabled:
            merged = self._recursive_expand(merged, limit)
            yield {"type": "recursive", "detail": "Recursive retrieval expansion applied."}

        final_results = merged[:limit]
        yield {"type": "final", "results": final_results, "trace": trace}

    def _sanitize_query(self, query: str) -> str:
        clean = QUERY_SANITIZE_RE.sub(" ", query).strip().lower()
        return re.sub(r"\s+", " ", clean)

    def _tokenize(self, value: str) -> list[str]:
        tokens = TOKEN_RE.findall(value.lower())
        return [token for token in tokens if len(token) > 1]

    def _search_once(self, query: str) -> list[RetrievedClause]:
        tokens = self._tokenize(query)
        if not tokens:
            return []

        ranked: list[RetrievedClause] = []
        for clause in self.clauses:
            haystack = " ".join(
                [
                    clause.standard,
                    clause.clause_id,
                    clause.clause_title,
                    clause.text,
                    " ".join(clause.keywords),
                ]
            ).lower()

            matched = [token for token in tokens if token in haystack]
            if not matched:
                continue

            overlap = len(matched)
            title_bonus = sum(1 for token in tokens if token in clause.clause_title.lower())
            exact_bonus = 3 if query and query in haystack else 0
            score = float(overlap + title_bonus + exact_bonus)

            ranked.append(
                RetrievedClause(clause=clause, score=score, matched_terms=sorted(set(matched)))
            )

        ranked.sort(key=lambda item: (-item.score, item.clause.doc_id, item.clause.clause_id))
        return ranked

    def _refine_query(self, query: str, top: list[RetrievedClause]) -> str | None:
        if not top:
            return None

        if self.search_provider.available and self.settings.agentic_search_enabled:
            try:
                prompt = (
                    "###TASK:REFINE###\n"
                    "You are a deterministic EC3 retrieval refiner. Return JSON array with 1 concise refined query.\n"
                    f"Original query: {query}\n"
                    "Top clause titles:\n"
                    + "\n".join(
                        f"- {item.clause.clause_id} {item.clause.clause_title}" for item in top[:4]
                    )
                )
                raw = self.search_provider.generate(
                    system_prompt="Refine retrieval query only. JSON output only.",
                    user_prompt=prompt,
                    temperature=0,
                    max_tokens=120,
                )
                data = json.loads(raw)
                if isinstance(data, list) and data:
                    refined = self._sanitize_query(str(data[0]))
                    if refined:
                        return refined
            except Exception as exc:  # noqa: BLE001
                logger.warning("search_refine_failed", extra={"error": str(exc)})

        # Heuristic fallback: append highest-signal clause title keywords.
        top_title = top[0].clause.clause_title.lower()
        tokens = self._tokenize(top_title)
        suffix = " ".join(tokens[:4])
        refined = self._sanitize_query(f"{query} {suffix}")
        return refined if refined != query else None

    def _recursive_expand(
        self, ranked: list[RetrievedClause], limit: int
    ) -> list[RetrievedClause]:
        # Experimental recursive mode: follow simple textual references like "6.2.5".
        if not ranked:
            return ranked

        by_id = {f"{c.clause.doc_id}:{c.clause.clause_id}": c for c in ranked}
        clause_lookup = {
            f"{c.doc_id}:{c.clause_id}": c
            for c in [item.clause for item in ranked] + [clause for clause in self.clauses]
        }

        reference_re = re.compile(r"\b\d+\.\d+(?:\.\d+)?(?:\([^)]+\))?\b")
        for item in list(ranked):
            refs = reference_re.findall(item.clause.text)
            for ref in refs:
                for key, clause in clause_lookup.items():
                    if key in by_id:
                        continue
                    if clause.doc_id != item.clause.doc_id:
                        continue
                    if clause.clause_id == ref:
                        by_id[key] = RetrievedClause(
                            clause=clause,
                            score=max(0.1, item.score - 0.25),
                            matched_terms=["recursive_ref"],
                        )

        expanded = sorted(
            by_id.values(), key=lambda entry: (-entry.score, entry.clause.doc_id, entry.clause.clause_id)
        )
        return expanded[:limit]
