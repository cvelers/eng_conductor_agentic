"""Dynamic iterative search — intent-aware retrieval with sufficiency evaluation.

Two retrieval modes:
  - **Blanket search**: broad topic queries decomposed into sub-queries,
    scored via TF-IDF + LLM reranking, then iteratively evaluated for gaps.
  - **Targeted fetch**: direct clause/table/equation lookup by ID pattern,
    used when the query (or a gap analysis result) asks for a specific item.

The key innovation over the previous linear pipeline is the **sufficiency
evaluation loop**: after initial retrieval the LLM reads actual clause
*content* and identifies specific missing items (tables, referenced clauses,
equations), then the retriever fetches those items directly before returning.

This means a query like "bending resistance check for IPE300" will:
  1. Find clauses about bending resistance (6.2.5, 6.2.6, etc.)
  2. Notice that Table 6.2 (cross-section classification) is referenced but missing
  3. Fetch Table 6.2 directly
  4. Return the complete set of clauses needed for the check.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterator

from backend.config import Settings
from backend.llm.base import LLMProvider
from backend.registries.document_registry import ClauseRecord
from backend.retrieval.semantic_scorer import SemanticScorer
from backend.utils.json_utils import parse_json_loose, strip_code_fences

logger = logging.getLogger(__name__)

QUERY_SANITIZE_RE = re.compile(r"[^A-Za-z0-9\s_\-\./]")
TOKEN_RE = re.compile(r"[a-z0-9_\-\.]+")
CLAUSE_ID_RE = re.compile(r"\b(\d+\.\d+(?:\.\d+)?(?:\([^)]+\))?)\b")
TABLE_ID_RE = re.compile(r"\b[Tt]able\s+(\d+\.\d+(?:\.\d+)?)\b")
FIGURE_ID_RE = re.compile(r"\b[Ff]igure?\s+(\d+\.\d+(?:\.\d+)?)\b")
# Match "EN 1993-1-X" or "EC3" style standard references
STANDARD_REF_RE = re.compile(r"\bEN\s*1993[- ]1[- ](\d+)\b", re.IGNORECASE)

STOPWORDS = frozenset({
    "the", "is", "in", "of", "to", "and", "for", "with", "that", "this",
    "are", "from", "be", "as", "by", "on", "it", "an", "or", "at", "if",
    "shall", "may", "should", "can", "which", "where", "when", "how",
    "what", "does", "has", "been", "its", "not", "but", "than", "into",
})


@dataclass
class RetrievedClause:
    clause: ClauseRecord
    score: float
    matched_terms: list[str]


@dataclass
class _IndexedClause:
    """Pre-processed clause data for fast search."""
    clause: ClauseRecord
    title_tokens: set[str]
    text_tokens: set[str]
    keyword_tokens: set[str]
    all_tokens: set[str]
    full_text_lower: str


@dataclass
class _SearchIntent:
    """Classified query intent."""
    mode: str  # "blanket" | "targeted" | "hybrid"
    original_query: str
    # For targeted mode: specific IDs to look up
    clause_ids: list[str] = field(default_factory=list)
    table_ids: list[str] = field(default_factory=list)
    # For blanket mode: the search terms
    search_query: str = ""
    # Optional standard filter
    standard_filter: str = ""


@dataclass
class _SufficiencyResult:
    """Result of LLM sufficiency evaluation."""
    score: int  # 1-10
    sufficient: bool
    missing_clauses: list[str] = field(default_factory=list)
    missing_tables: list[str] = field(default_factory=list)
    follow_up_query: str = ""
    reasoning: str = ""


class AgenticRetriever:
    def __init__(
        self,
        *,
        settings: Settings,
        search_provider: LLMProvider,
        clauses: list[ClauseRecord],
        semantic_scorer: SemanticScorer | None = None,
    ) -> None:
        self.settings = settings
        self.search_provider = search_provider
        self.clauses = clauses
        self.semantic_scorer = semantic_scorer
        self._entries: list[_IndexedClause] = []
        self._inverted_index: dict[str, set[int]] = defaultdict(set)
        self._doc_freq: dict[str, int] = {}
        self._total_docs = max(len(clauses), 1)
        # Direct lookup indices
        self._by_clause_id: dict[str, list[ClauseRecord]] = defaultdict(list)
        self._by_title_lower: dict[str, list[ClauseRecord]] = defaultdict(list)
        self._build_index()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        for idx, clause in enumerate(self.clauses):
            title_tokens = self._tokenize(clause.clause_title)
            text_tokens = self._tokenize(clause.text)
            keyword_tokens = self._tokenize(" ".join(clause.keywords))
            all_tokens = title_tokens | text_tokens | keyword_tokens

            full_text = " ".join([
                clause.standard, clause.clause_id, clause.clause_title,
                clause.text, " ".join(clause.keywords),
            ]).lower()

            entry = _IndexedClause(
                clause=clause,
                title_tokens=title_tokens,
                text_tokens=text_tokens,
                keyword_tokens=keyword_tokens,
                all_tokens=all_tokens,
                full_text_lower=full_text,
            )
            self._entries.append(entry)

            for token in all_tokens:
                self._inverted_index[token].add(idx)

            # Build direct lookup indices
            cid = clause.clause_id.lower().strip()
            self._by_clause_id[cid].append(clause)
            # Also index by bare ID for tables: "table 6.2" → "6.2"
            if cid.startswith("table"):
                bare = cid.replace("table", "").strip()
                self._by_clause_id[f"table {bare}"].append(clause)
                self._by_clause_id[bare].append(clause)
            self._by_title_lower[clause.clause_title.lower().strip()].append(clause)

        self._doc_freq = {
            token: len(indices) for token, indices in self._inverted_index.items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        *,
        agentic: bool | None = None,
        recursive: bool | None = None,
    ) -> tuple[list[RetrievedClause], list[dict[str, object]]]:
        final_results: list[RetrievedClause] = []
        trace: list[dict[str, object]] = []
        for event in self.iter_retrieve(
            query,
            top_k=top_k,
            agentic=agentic,
            recursive=recursive,
        ):
            if event.get("type") == "final":
                final_results = event.get("results", [])
                trace = event.get("trace", [])
        return final_results, trace

    def iter_retrieve(
        self,
        query: str,
        top_k: int | None = None,
        *,
        agentic: bool | None = None,
        recursive: bool | None = None,
    ) -> Iterator[dict[str, Any]]:
        limit = top_k or self.settings.top_k_clauses
        trace: list[dict[str, object]] = []
        aggregated: dict[str, RetrievedClause] = {}

        agentic_enabled = (
            self.settings.agentic_search_enabled if agentic is None else bool(agentic)
        ) and self.search_provider.available
        recursive_enabled = (
            self.settings.recursive_retrieval_enabled
            if recursive is None
            else bool(recursive)
        )
        max_sufficiency_iters = (
            max(0, self.settings.max_retrieval_iters) if agentic_enabled else 0
        )

        # ---- Phase 1: Intent Classification ----
        intent = self._classify_intent(query)
        trace.append({
            "iteration": 0, "phase": "classify",
            "mode": intent.mode, "query": query,
            "clause_ids": intent.clause_ids,
            "table_ids": intent.table_ids,
        })

        # ---- Phase 2A: Targeted fetch (if any direct IDs found) ----
        if intent.clause_ids or intent.table_ids:
            targeted_results = self._targeted_fetch(
                clause_ids=intent.clause_ids,
                table_ids=intent.table_ids,
                standard_filter=intent.standard_filter,
            )
            for item in targeted_results:
                key = item.clause.citation_address
                if key not in aggregated or item.score > aggregated[key].score:
                    aggregated[key] = item

            trace.append({
                "iteration": 0, "phase": "targeted_fetch",
                "fetched": len(targeted_results),
                "ids": [r.clause.clause_id for r in targeted_results],
            })

            if intent.mode == "targeted" and targeted_results:
                # Pure targeted query — return directly
                final_results = sorted(
                    aggregated.values(),
                    key=lambda x: (-x.score, x.clause.doc_id, x.clause.clause_id),
                )[:limit]
                yield {"type": "final", "results": final_results, "trace": trace}
                return

        # ---- Phase 2B: Blanket search ----
        use_semantic = (
            self.semantic_scorer is not None
            and self.semantic_scorer.available
        )

        sub_queries = (
            self._decompose_query(query) if agentic_enabled
            else [self._sanitize(query)]
        )
        trace.append({
            "iteration": 1, "phase": "decompose",
            "query": query, "sub_queries": sub_queries,
        })

        if use_semantic:
            # ── Semantic retrieval (bi-encoder) ──────────────────────
            candidates = self._search_semantic(
                query, limit=limit * 3, sub_queries=sub_queries,
            )
            search_mode = "semantic"
        else:
            # ── TF-IDF fallback (wire cut — only used if semantic unavailable)
            candidate_pool: dict[str, RetrievedClause] = {}
            for sq in sub_queries:
                for hit in self._search_lexical(sq, limit=limit * 4):
                    key = hit.clause.citation_address
                    if key not in candidate_pool or hit.score > candidate_pool[key].score:
                        candidate_pool[key] = hit
            candidates = sorted(candidate_pool.values(), key=lambda x: -x.score)[: limit * 3]
            search_mode = "lexical"

        step_retrieval = {
            "iteration": 1, "phase": search_mode, "query": query,
            "top_clause_ids": [
                f"{c.clause.doc_id}:{c.clause.clause_id}" for c in candidates[:limit]
            ],
        }
        trace.append(step_retrieval)
        logger.info(
            "retrieval_%s", search_mode,
            extra={"hits": len(candidates), "top": [c.clause.clause_id for c in candidates[:3]]},
        )
        yield {
            "type": "iteration", "step": step_retrieval,
            "top": [
                {"doc_id": c.clause.doc_id, "clause_id": c.clause.clause_id,
                 "title": c.clause.clause_title, "score": c.score}
                for c in candidates[:3]
            ],
        }

        # ---- Phase 3: LLM relevance re-ranking ----
        if agentic_enabled and candidates:
            scored = self._llm_score_relevance(query, candidates[:30])
            for item in scored:
                key = item.clause.citation_address
                if key not in aggregated or item.score > aggregated[key].score:
                    aggregated[key] = item

            step_rerank = {
                "iteration": 2, "phase": "rerank", "query": query,
                "top_clause_ids": [
                    f"{c.clause.doc_id}:{c.clause.clause_id}"
                    for c in sorted(aggregated.values(), key=lambda x: -x.score)[:limit]
                ],
            }
            trace.append(step_rerank)
            logger.info(
                "retrieval_reranked",
                extra={"hits": len(aggregated), "top": [
                    c.clause.clause_id
                    for c in sorted(aggregated.values(), key=lambda x: -x.score)[:3]
                ]},
            )
            yield {
                "type": "iteration", "step": step_rerank,
                "top": [
                    {"doc_id": c.clause.doc_id, "clause_id": c.clause.clause_id,
                     "title": c.clause.clause_title, "score": c.score}
                    for c in sorted(aggregated.values(), key=lambda x: -x.score)[:3]
                ],
            }
        else:
            for item in candidates[:limit]:
                aggregated[item.clause.citation_address] = item

        # ---- Phase 4: Sufficiency evaluation loop ----
        seen_queries: set[str] = set()
        fetched_ids: set[str] = set()

        for suff_iter in range(max_sufficiency_iters):
            current_top = sorted(
                aggregated.values(), key=lambda x: -x.score,
            )[:limit]

            evaluation = self._evaluate_sufficiency(query, current_top)

            step_eval = {
                "iteration": 3 + suff_iter, "phase": "sufficiency_eval",
                "score": evaluation.score,
                "sufficient": evaluation.sufficient,
                "missing_clauses": evaluation.missing_clauses,
                "missing_tables": evaluation.missing_tables,
                "follow_up_query": evaluation.follow_up_query,
            }
            trace.append(step_eval)
            logger.info(
                "sufficiency_eval",
                extra={
                    "score": evaluation.score,
                    "sufficient": evaluation.sufficient,
                    "missing": evaluation.missing_clauses + evaluation.missing_tables,
                },
            )

            if evaluation.sufficient:
                break

            made_progress = False

            # 4A: Targeted fetches for specific missing items
            new_clause_ids = [
                cid for cid in evaluation.missing_clauses if cid not in fetched_ids
            ]
            new_table_ids = [
                tid for tid in evaluation.missing_tables if tid not in fetched_ids
            ]

            if new_clause_ids or new_table_ids:
                targeted_hits = self._targeted_fetch(
                    clause_ids=new_clause_ids,
                    table_ids=new_table_ids,
                    standard_filter=intent.standard_filter,
                )
                for item in targeted_hits:
                    key = item.clause.citation_address
                    if key not in aggregated or item.score > aggregated[key].score:
                        aggregated[key] = item
                        made_progress = True
                fetched_ids.update(new_clause_ids)
                fetched_ids.update(new_table_ids)

                if targeted_hits:
                    step_targeted = {
                        "iteration": 3 + suff_iter, "phase": "targeted_follow_up",
                        "fetched": [r.clause.clause_id for r in targeted_hits],
                    }
                    trace.append(step_targeted)
                    yield {
                        "type": "iteration", "step": step_targeted,
                        "top": [
                            {"doc_id": r.clause.doc_id, "clause_id": r.clause.clause_id,
                             "title": r.clause.clause_title, "score": r.score}
                            for r in targeted_hits[:3]
                        ],
                    }

            # 4B: Blanket search for topic gaps
            gap_query = evaluation.follow_up_query
            if gap_query and gap_query not in seen_queries:
                seen_queries.add(gap_query)
                if use_semantic:
                    gap_hits = self._search_semantic(gap_query, limit=limit * 2)
                else:
                    gap_hits = self._search_lexical(gap_query, limit=limit * 2)
                new_candidates = [
                    h for h in gap_hits if h.clause.citation_address not in aggregated
                ][:15]

                if new_candidates:
                    if self.search_provider.available:
                        scored_new = self._llm_score_relevance(query, new_candidates)
                        for item in scored_new:
                            key = item.clause.citation_address
                            if key not in aggregated or item.score > aggregated[key].score:
                                aggregated[key] = item
                                made_progress = True
                    else:
                        for item in new_candidates:
                            aggregated[item.clause.citation_address] = item
                            made_progress = True

                    step_gap = {
                        "iteration": 3 + suff_iter, "phase": "gap_search",
                        "query": gap_query,
                        "new_hits": len(new_candidates),
                    }
                    trace.append(step_gap)
                    yield {
                        "type": "iteration", "step": step_gap,
                        "top": [
                            {"doc_id": c.clause.doc_id, "clause_id": c.clause.clause_id,
                             "title": c.clause.clause_title, "score": c.score}
                            for c in sorted(aggregated.values(), key=lambda x: -x.score)[:3]
                        ],
                    }

            if not made_progress:
                break

        # ---- Phase 5: Recursive expansion (optional) ----
        merged = sorted(
            aggregated.values(),
            key=lambda item: (-item.score, item.clause.doc_id, item.clause.clause_id),
        )

        if recursive_enabled:
            merged = self._recursive_expand(merged, limit)
            yield {"type": "recursive", "detail": "Recursive retrieval expansion applied."}

        final_results = merged[:limit]
        yield {"type": "final", "results": final_results, "trace": trace}

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    def _classify_intent(self, query: str) -> _SearchIntent:
        """Classify query as blanket search, targeted fetch, or hybrid."""
        clean = query.strip()
        lower = clean.lower()

        clause_ids: list[str] = []
        table_ids: list[str] = []
        standard_filter = ""

        # Extract standard references
        std_match = STANDARD_REF_RE.search(clean)
        if std_match:
            part_num = std_match.group(1)
            standard_filter = f"EN 1993-1-{part_num}"

        # Extract explicit table references: "Table 6.2", "table 3.1"
        for m in TABLE_ID_RE.finditer(clean):
            tid = f"table {m.group(1)}"
            if tid not in table_ids:
                table_ids.append(tid)

        # Extract clause ID patterns: "6.3.2.3", "3.1(2)"
        for m in CLAUSE_ID_RE.finditer(clean):
            cid = m.group(1)
            # Don't treat numbers that are part of "EN 1993-1-1" as clause IDs
            start = m.start()
            prefix = clean[:start].rstrip()
            if prefix.endswith("-") or prefix.lower().endswith("en 1993"):
                continue
            if cid not in clause_ids:
                clause_ids.append(cid)

        # Detect "read clause X" / "show me clause X" / "what does clause X say"
        direct_patterns = [
            r"(?:read|show|get|fetch|look\s*up|find)\s+(?:me\s+)?(?:clause|table|section)\s+",
            r"what\s+does?\s+(?:clause|table|section)\s+",
            r"(?:clause|table|section)\s+\d+\.\d+",
        ]
        is_direct = any(re.search(p, lower) for p in direct_patterns)

        # Determine mode
        if (clause_ids or table_ids) and is_direct:
            mode = "targeted"
        elif clause_ids or table_ids:
            mode = "hybrid"  # Has IDs but also has search context
        else:
            mode = "blanket"

        return _SearchIntent(
            mode=mode,
            original_query=clean,
            clause_ids=clause_ids,
            table_ids=table_ids,
            search_query=self._sanitize(clean),
            standard_filter=standard_filter,
        )

    # ------------------------------------------------------------------
    # Targeted fetch — direct clause/table/equation lookup
    # ------------------------------------------------------------------

    def _targeted_fetch(
        self,
        *,
        clause_ids: list[str] | None = None,
        table_ids: list[str] | None = None,
        standard_filter: str = "",
    ) -> list[RetrievedClause]:
        """Directly look up clauses by ID. Fast, no LLM call."""
        results: list[RetrievedClause] = []
        seen: set[str] = set()

        all_targets = []
        for cid in (clause_ids or []):
            all_targets.append(("clause", cid))
        for tid in (table_ids or []):
            all_targets.append(("table", tid))

        for target_type, target_id in all_targets:
            target_lower = target_id.lower().strip()

            # Strategy 1: Exact match in clause_id index
            candidates = list(self._by_clause_id.get(target_lower, []))

            # Strategy 2: Prefix match (e.g., "6.3" matches "6.3.1", "6.3.2")
            if not candidates:
                for key, clauses in self._by_clause_id.items():
                    if key.startswith(target_lower) or target_lower.startswith(key):
                        candidates.extend(clauses)

            # Strategy 3: Title search for tables
            if target_type == "table" and not candidates:
                # Extract the numeric part
                num_part = target_lower.replace("table", "").strip()
                for entry in self._entries:
                    title_lower = entry.clause.clause_title.lower()
                    cid_lower = entry.clause.clause_id.lower()
                    if (f"table {num_part}" in title_lower
                            or f"table {num_part}" in cid_lower
                            or cid_lower == f"table {num_part}"):
                        candidates.append(entry.clause)

            # Strategy 4: Full-text search as fallback
            if not candidates:
                for entry in self._entries:
                    if target_lower in entry.full_text_lower:
                        candidates.append(entry.clause)
                        if len(candidates) >= 3:
                            break

            # Apply standard filter
            if standard_filter:
                std_lower = standard_filter.lower()
                filtered = [c for c in candidates if c.standard.lower() == std_lower]
                if filtered:
                    candidates = filtered

            # Deduplicate and score
            for clause in candidates[:5]:
                key = clause.citation_address
                if key in seen:
                    continue
                seen.add(key)
                # High score for direct fetches — they were explicitly requested
                results.append(RetrievedClause(
                    clause=clause,
                    score=9.0,
                    matched_terms=[f"targeted:{target_id}"],
                ))

        return results

    # ------------------------------------------------------------------
    # Sufficiency evaluation — LLM reads actual content to find gaps
    # ------------------------------------------------------------------

    def _evaluate_sufficiency(
        self, query: str, current_results: list[RetrievedClause],
    ) -> _SufficiencyResult:
        """LLM evaluates whether results are sufficient, with content awareness."""
        if not current_results or not self.search_provider.available:
            return _SufficiencyResult(score=5, sufficient=True)

        # Build content summaries — show actual text, not just titles
        result_summaries: list[str] = []
        for i, r in enumerate(current_results[:10]):
            # Show enough text to see cross-references
            text_preview = r.clause.text[:500].replace("\n", " ")
            result_summaries.append(
                f"{i + 1}. [{r.clause.standard} {r.clause.clause_id}] "
                f"{r.clause.clause_title}\n"
                f"   Content preview: {text_preview}"
            )

        try:
            raw = self.search_provider.generate(
                system_prompt=(
                    "You evaluate Eurocode search results for completeness.\n"
                    "Given a query and the retrieved clauses (with content previews), assess:\n"
                    "1. Are the key formulas/rules/tables needed to answer the query present?\n"
                    "2. Are there tables, clauses, or equations REFERENCED in the results "
                    "that are NOT in the results but would be needed?\n\n"
                    "Return JSON:\n"
                    "{\n"
                    '  "score": 1-10,  // 10 = fully sufficient, 1 = mostly missing\n'
                    '  "sufficient": true/false,  // true if score >= 7\n'
                    '  "missing_clauses": ["6.3.2.3", "6.1(1)"],  // specific clause IDs referenced but not present\n'
                    '  "missing_tables": ["Table 6.2", "Table 3.1"],  // specific tables referenced but not present\n'
                    '  "follow_up_query": "string or null",  // topic search if there is a conceptual gap\n'
                    '  "reasoning": "brief explanation"\n'
                    "}\n\n"
                    "IMPORTANT: Only list SPECIFIC clause/table IDs that are explicitly "
                    "referenced in the retrieved text but not present in the results. "
                    "Do not guess or invent IDs."
                ),
                user_prompt=(
                    "###TASK:SUFFICIENCY###\n"
                    f"Query: {query}\n\n"
                    f"Retrieved clauses:\n"
                    + "\n".join(result_summaries)
                    + "\n\nEvaluate completeness. Return JSON only."
                ),
                temperature=0,
                max_tokens=self.settings.gap_analysis_max_tokens,
                **({"reasoning_effort": self.settings.gap_analysis_reasoning_effort}
                   if self.settings.gap_analysis_reasoning_effort else {}),
            )
            data = parse_json_loose(raw)
            if isinstance(data, dict):
                score = int(data.get("score", 5))
                missing_clauses = [
                    str(c) for c in data.get("missing_clauses", [])
                    if isinstance(c, str) and c.strip()
                ]
                missing_tables = [
                    str(t) for t in data.get("missing_tables", [])
                    if isinstance(t, str) and t.strip()
                ]
                follow_up = data.get("follow_up_query")
                if isinstance(follow_up, str) and follow_up.strip():
                    follow_up = self._sanitize(follow_up)
                else:
                    follow_up = ""

                return _SufficiencyResult(
                    score=score,
                    sufficient=score >= 7,
                    missing_clauses=missing_clauses,
                    missing_tables=missing_tables,
                    follow_up_query=follow_up,
                    reasoning=str(data.get("reasoning", "")),
                )
        except Exception as exc:
            logger.warning("sufficiency_eval_failed", extra={"error": str(exc)})

        return _SufficiencyResult(score=5, sufficient=True)

    # ------------------------------------------------------------------
    # Lexical search with TF-IDF field-weighted scoring
    # ------------------------------------------------------------------

    def _sanitize(self, query: str) -> str:
        clean = QUERY_SANITIZE_RE.sub(" ", query).strip().lower()
        return re.sub(r"\s+", " ", clean)

    def _tokenize(self, value: str) -> set[str]:
        tokens = TOKEN_RE.findall(value.lower())
        return {t for t in tokens if len(t) > 1 and t not in STOPWORDS}

    def _idf(self, token: str) -> float:
        df = self._doc_freq.get(token, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + self._total_docs / df)

    def _search_lexical(self, query: str, limit: int) -> list[RetrievedClause]:
        safe_query = self._sanitize(query)
        tokens = self._tokenize(safe_query)
        if not tokens:
            return []

        candidate_indices: dict[int, float] = {}
        for token in tokens:
            idf = self._idf(token)
            for idx in self._inverted_index.get(token, set()):
                candidate_indices[idx] = candidate_indices.get(idx, 0) + idf

        clause_id_patterns = set(CLAUSE_ID_RE.findall(safe_query))
        for idx, entry in enumerate(self._entries):
            cid = entry.clause.clause_id.lower()
            for qid in clause_id_patterns:
                if cid == qid or cid.startswith(qid):
                    candidate_indices[idx] = candidate_indices.get(idx, 0) + 5.0

        ranked: list[RetrievedClause] = []
        for idx in candidate_indices:
            entry = self._entries[idx]

            title_score = sum(self._idf(t) * 3.0 for t in tokens if t in entry.title_tokens)
            keyword_score = sum(self._idf(t) * 2.0 for t in tokens if t in entry.keyword_tokens)
            text_score = sum(self._idf(t) for t in tokens if t in entry.text_tokens)

            phrase_bonus = 4.0 if (len(safe_query) > 5 and safe_query in entry.full_text_lower) else 0.0

            matched = tokens & entry.all_tokens
            coverage_bonus = (len(matched) / len(tokens)) * 2.0 if tokens else 0.0

            total = title_score + keyword_score + text_score + phrase_bonus + coverage_bonus

            ranked.append(RetrievedClause(
                clause=entry.clause, score=total, matched_terms=sorted(matched),
            ))

        ranked.sort(key=lambda x: (-x.score, x.clause.doc_id, x.clause.clause_id))
        return ranked[:limit]

    # ------------------------------------------------------------------
    # Semantic search (bi-encoder)
    # ------------------------------------------------------------------

    def _search_semantic(
        self,
        query: str,
        limit: int,
        sub_queries: list[str] | None = None,
    ) -> list[RetrievedClause]:
        """Retrieve clauses using the bi-encoder semantic scorer.

        If sub_queries are provided, searches with all of them and merges
        results by taking the max similarity per clause.

        Scores are scaled from cosine [0,1] to [0,10] range for downstream
        compatibility with the LLM reranker and sufficiency loop.
        """
        if not self.semantic_scorer or not self.semantic_scorer.available:
            return []

        queries = sub_queries if sub_queries and len(sub_queries) > 1 else None
        if queries:
            hits = self.semantic_scorer.search_multi(queries, top_k=limit)
        else:
            hits = self.semantic_scorer.search(query, top_k=limit)

        results: list[RetrievedClause] = []
        for hit in hits:
            clause = self.clauses[hit.index]
            entry = self._entries[hit.index]
            # Scale cosine similarity [0,1] → [0,10]
            scaled_score = round(hit.score * 10.0, 2)
            # Identify which query tokens matched (for diagnostics)
            q_tokens = self._tokenize(query)
            matched = q_tokens & entry.all_tokens
            results.append(RetrievedClause(
                clause=clause,
                score=scaled_score,
                matched_terms=sorted(matched) + ["semantic"],
            ))

        return results

    # ------------------------------------------------------------------
    # LLM-powered search operations
    # ------------------------------------------------------------------

    def _decompose_query(self, query: str) -> list[str]:
        try:
            raw = self.search_provider.generate(
                system_prompt=(
                    "You decompose engineering queries into focused search sub-queries "
                    "for a Eurocode clause database.\n"
                    "Return a JSON array of 1-4 concise search strings. "
                    "Each should target a different aspect of the question.\n"
                    "Always include the original query (possibly simplified) as the first element.\n"
                    "Think about what Eurocode clauses, tables, and formulas would be needed."
                ),
                user_prompt=(
                    "###TASK:DECOMPOSE###\n"
                    f"Query: {query}\n\n"
                    "Return JSON array of search strings only."
                ),
                temperature=0,
                max_tokens=self.settings.search_decompose_max_tokens,
                **({"reasoning_effort": self.settings.search_decompose_reasoning_effort}
                   if self.settings.search_decompose_reasoning_effort else {}),
            )
            data = parse_json_loose(raw)
            if isinstance(data, list) and data:
                queries = [self._sanitize(str(q)) for q in data[:4] if str(q).strip()]
                if queries:
                    return queries
        except Exception as exc:
            logger.warning("query_decomposition_failed", extra={"error": str(exc)})

        return [self._sanitize(query)]

    def _llm_score_relevance(
        self, query: str, candidates: list[RetrievedClause],
    ) -> list[RetrievedClause]:
        if not candidates:
            return []
        if not self.search_provider.available:
            return candidates

        descriptions: list[str] = []
        for i, c in enumerate(candidates):
            snippet = c.clause.text[:250].replace("\n", " ")
            descriptions.append(
                f"{i + 1}. [{c.clause.standard} — {c.clause.clause_id}] "
                f"{c.clause.clause_title}: {snippet}"
            )

        try:
            raw = self.search_provider.generate(
                system_prompt=(
                    "You are a Eurocode relevance scorer. "
                    "Score each clause for relevance to the engineering query.\n"
                    "Each clause shows [STANDARD — ClauseID]. The standard matters: "
                    "if the query mentions a specific standard (e.g. EN 1993-1-1), "
                    "clauses from that standard should score higher than equivalent "
                    "clauses from other parts (EN 1993-1-3, 1-4, 1-5, etc.).\n"
                    "A clause that IS the primary source (e.g. 6.2.6 in EN 1993-1-1 "
                    "for shear) should score higher than clauses that merely reference it.\n\n"
                    "Return a JSON array: [{\"idx\": 1, \"score\": 0-10}, ...]\n"
                    "10 = directly answers the query with key formulas/rules, from the right standard.\n"
                    "7-9 = highly relevant, contains needed information.\n"
                    "4-6 = related context or from a secondary standard.\n"
                    "0-3 = tangentially related or irrelevant."
                ),
                user_prompt=(
                    "###TASK:RELEVANCE###\n"
                    f"Query: {query}\n\n"
                    "Clauses:\n" + "\n".join(descriptions) + "\n\n"
                    "Score each clause. Return JSON array only."
                ),
                temperature=self.settings.rerank_temperature,
                max_tokens=self.settings.rerank_max_tokens,
                **({"reasoning_effort": self.settings.rerank_reasoning_effort}
                   if self.settings.rerank_reasoning_effort else {}),
            )
            data = parse_json_loose(raw)
            if isinstance(data, list):
                score_map: dict[int, float] = {}
                for item in data:
                    if isinstance(item, dict):
                        idx = int(item.get("idx", 0)) - 1
                        score = float(item.get("score", 0))
                        if 0 <= idx < len(candidates):
                            score_map[idx] = score

                result: list[RetrievedClause] = []
                for i, c in enumerate(candidates):
                    llm_score = score_map.get(i, c.score)
                    result.append(RetrievedClause(
                        clause=c.clause,
                        score=llm_score,
                        matched_terms=c.matched_terms + ["llm_scored"],
                    ))
                result.sort(key=lambda x: (-x.score, x.clause.doc_id, x.clause.clause_id))
                return result
        except Exception as exc:
            logger.warning("llm_relevance_scoring_failed", extra={"error": str(exc)})

        return candidates

    # ------------------------------------------------------------------
    # Recursive cross-reference expansion
    # ------------------------------------------------------------------

    def _recursive_expand(
        self, ranked: list[RetrievedClause], limit: int,
    ) -> list[RetrievedClause]:
        if not ranked:
            return ranked

        by_id = {f"{c.clause.doc_id}:{c.clause.clause_id}": c for c in ranked}
        clause_lookup = {
            f"{c.doc_id}:{c.clause_id}": c
            for c in [item.clause for item in ranked] + self.clauses
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
            by_id.values(),
            key=lambda entry: (-entry.score, entry.clause.doc_id, entry.clause.clause_id),
        )
        return expanded[:limit]


def _strip_code_fences(text: str) -> str:
    # Backward-compat shim for existing call sites/tests.
    return strip_code_fences(text)
