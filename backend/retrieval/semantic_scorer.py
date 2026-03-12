"""Bi-encoder semantic scorers for clause retrieval.

Two scorer implementations:
  - SimpleSemanticScorer: MiniLM with 1200-char truncation per clause.
    Fast, lightweight, currently wired in.
  - SemanticScorer: Chunked approach with overlapping chunks.
    More thorough but heavier. Wire cut for now.

Falls back gracefully if sentence-transformers is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from backend.registries.document_registry import ClauseRecord

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Chunking parameters
_CHUNK_CHARS = 800          # ~200 tokens for MiniLM
_CHUNK_OVERLAP_CHARS = 200  # ~50 tokens overlap between chunks
_MIN_CHUNK_CHARS = 100      # don't create tiny trailing chunks

# Aggregation parameters
_TOP_K_CHUNKS = 3           # mean of top-k chunk scores per clause
_COVERAGE_THRESHOLD = 0.15  # chunk sim must exceed this to count as "relevant"
_COVERAGE_WEIGHT = 0.15     # how much coverage fraction boosts the final score


@dataclass(frozen=True)
class ScoredHit:
    """A clause index + aggregated similarity score."""
    index: int
    score: float


class SemanticScorer:
    """Chunked bi-encoder semantic similarity scorer.

    Each clause is split into overlapping chunks, each prefixed with
    the clause's standard, ID and title. All chunks are embedded at init.

    At query time, cosine similarities are computed for all chunks, then
    aggregated per clause using: mean(top-3 chunk sims) * (1 + coverage_bonus).

    This produces a unique per-clause score that reflects both peak relevance
    and breadth of match across the clause content.
    """

    def __init__(
        self,
        clauses: list[ClauseRecord],
        model_name: str = _DEFAULT_MODEL,
    ) -> None:
        self.clauses = clauses
        self._model_name = model_name
        self._model = None
        self._chunk_embeddings: np.ndarray | None = None
        # Mapping: chunk index → clause index
        self._chunk_to_clause: np.ndarray | None = None
        # Number of chunks per clause (for coverage calculation)
        self._clause_chunk_counts: np.ndarray | None = None
        self._n_clauses: int = len(clauses)
        self._build_index()

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    @staticmethod
    def _make_prefix(c: ClauseRecord) -> str:
        """Short prefix prepended to every chunk so the model sees the source."""
        parts = [c.standard, f"Clause {c.clause_id}"]
        if c.clause_title:
            parts.append(c.clause_title)
        return " — ".join(parts) + ". "

    @staticmethod
    def _split_text(text: str, chunk_chars: int, overlap: int) -> list[str]:
        """Split text into overlapping chunks by character count."""
        if len(text) <= chunk_chars:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_chars
            chunk = text[start:end]
            if len(chunk) < _MIN_CHUNK_CHARS and chunks:
                # Append tiny trailing text to last chunk instead
                chunks[-1] = chunks[-1] + " " + chunk
                break
            chunks.append(chunk)
            start = end - overlap
        return chunks

    def _clause_to_chunks(self, c: ClauseRecord) -> list[str]:
        """Build chunk texts for a single clause."""
        prefix = self._make_prefix(c)

        # Build full text: keywords + body
        parts = []
        if c.keywords:
            parts.append("Keywords: " + ", ".join(c.keywords[:10]))
        body = c.text.replace("\n", " ").strip()
        if body:
            parts.append(body)
        full_text = " ".join(parts)

        if not full_text.strip():
            # Clause has no body — just embed the prefix (title/standard)
            return [prefix]

        raw_chunks = self._split_text(full_text, _CHUNK_CHARS, _CHUNK_OVERLAP_CHARS)
        return [prefix + chunk for chunk in raw_chunks]

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — semantic scorer disabled. "
                "Install with: pip install sentence-transformers"
            )
            return

        try:
            logger.info("Loading semantic model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)

            # Build all chunks with clause-index mapping
            all_chunks: list[str] = []
            chunk_to_clause: list[int] = []
            clause_chunk_counts: list[int] = []

            for idx, clause in enumerate(self.clauses):
                chunks = self._clause_to_chunks(clause)
                clause_chunk_counts.append(len(chunks))
                for chunk in chunks:
                    all_chunks.append(chunk)
                    chunk_to_clause.append(idx)

            self._chunk_to_clause = np.array(chunk_to_clause, dtype=np.int32)
            self._clause_chunk_counts = np.array(clause_chunk_counts, dtype=np.int32)

            logger.info(
                "Encoding %d chunks from %d clauses …",
                len(all_chunks), len(self.clauses),
            )

            self._chunk_embeddings = self._model.encode(
                all_chunks,
                show_progress_bar=False,
                batch_size=64,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

            logger.info(
                "Semantic index ready — %d clauses, %d chunks, embedding dim %d",
                len(self.clauses),
                len(all_chunks),
                self._chunk_embeddings.shape[1],
            )
        except Exception:
            logger.exception("Failed to build semantic index")
            self._model = None
            self._chunk_embeddings = None

    # ------------------------------------------------------------------
    # Score aggregation
    # ------------------------------------------------------------------

    def _aggregate_clause_scores(self, chunk_sims: np.ndarray) -> np.ndarray:
        """Aggregate per-chunk similarities into per-clause scores.

        For each clause:
          base = mean of top-K chunk similarities (K = _TOP_K_CHUNKS)
          coverage = fraction of chunks with sim > _COVERAGE_THRESHOLD
          score = base * (1 + _COVERAGE_WEIGHT * coverage)

        This rewards clauses where multiple chunks are relevant (broad match)
        over clauses where only one chunk happens to match well.
        """
        n_clauses = self._n_clauses
        scores = np.zeros(n_clauses, dtype=np.float64)

        # Pre-compute chunk offset ranges per clause
        offsets = np.zeros(n_clauses + 1, dtype=np.int64)
        np.cumsum(self._clause_chunk_counts, out=offsets[1:])

        for ci in range(n_clauses):
            start, end = offsets[ci], offsets[ci + 1]
            if start == end:
                continue
            clause_sims = chunk_sims[start:end]
            n_chunks = end - start

            # Top-K mean (or all chunks if fewer than K)
            k = min(_TOP_K_CHUNKS, n_chunks)
            top_k_vals = np.partition(clause_sims, -k)[-k:]
            base = float(np.mean(top_k_vals))

            # Coverage bonus: fraction of chunks above threshold
            above = int(np.sum(clause_sims > _COVERAGE_THRESHOLD))
            coverage = above / n_chunks

            scores[ci] = base * (1.0 + _COVERAGE_WEIGHT * coverage)

        return scores

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._model is not None and self._chunk_embeddings is not None

    def _encode_queries(self, queries: list[str]) -> np.ndarray:
        """Encode query strings."""
        return self._model.encode(
            queries,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def search(self, query: str, top_k: int = 50) -> list[ScoredHit]:
        """Return top-k clause hits sorted by aggregated score (desc).

        Scores are in [0, ~1.15] range (base cosine * coverage bonus).
        """
        if not self.available:
            return []

        q_emb = self._encode_queries([query])

        # Cosine sim of query against all chunks
        chunk_sims = (self._chunk_embeddings @ q_emb.T).flatten()

        # Aggregate to per-clause scores
        clause_scores = self._aggregate_clause_scores(chunk_sims)

        top_idx = np.argsort(clause_scores)[::-1][:top_k]

        return [
            ScoredHit(index=int(i), score=max(0.0, float(clause_scores[i])))
            for i in top_idx
        ]

    def search_multi(
        self,
        queries: list[str],
        top_k: int = 50,
    ) -> list[ScoredHit]:
        """Search with multiple queries, aggregate per clause.

        For each query, computes per-clause aggregated scores, then takes
        the max across queries for each clause.
        """
        if not self.available or not queries:
            return []

        q_embs = self._encode_queries(queries)

        # (n_chunks, n_queries) chunk-level sims
        all_chunk_sims = self._chunk_embeddings @ q_embs.T

        # Aggregate per clause for each query, then take max across queries
        n_queries = all_chunk_sims.shape[1]
        all_clause_scores = np.zeros((self._n_clauses, n_queries), dtype=np.float64)

        for qi in range(n_queries):
            all_clause_scores[:, qi] = self._aggregate_clause_scores(
                all_chunk_sims[:, qi]
            )

        # Max across queries for each clause
        max_scores = all_clause_scores.max(axis=1)

        top_idx = np.argsort(max_scores)[::-1][:top_k]

        return [
            ScoredHit(index=int(i), score=max(0.0, float(max_scores[i])))
            for i in top_idx
        ]


# ======================================================================
# Simple scorer — MiniLM, 1200 chars, no chunking (currently active)
# ======================================================================

_MAX_DOC_CHARS = 1200
_MIN_BODY_CHARS = 50  # clauses shorter than this are section headings — skip


class SimpleSemanticScorer:
    """Simple bi-encoder scorer using all-MiniLM-L6-v2.

    Embeds each clause as: "standard — clause_id — title. body[:1200]".
    Clauses with less than 50 chars of body text (section headings, empty
    stubs) are excluded from the index entirely.

    At query time, computes cosine similarity via a single matrix multiply.
    """

    def __init__(
        self,
        clauses: list[ClauseRecord],
        model_name: str = _DEFAULT_MODEL,
    ) -> None:
        self.clauses = clauses
        self._model_name = model_name
        self._model = None
        self._embeddings: np.ndarray | None = None
        # Mapping from embedding row → original clause index
        # (skipped clauses are not embedded)
        self._emb_to_clause: np.ndarray | None = None
        self._build_index()

    @staticmethod
    def _clause_to_text(c: ClauseRecord) -> str | None:
        """Build embedding text. Returns None for empty/stub clauses."""
        body = c.text.replace("\n", " ").strip()
        if len(body) < _MIN_BODY_CHARS:
            return None
        parts = [c.standard, f"Clause {c.clause_id}", c.clause_title]
        if c.keywords:
            parts.append("Keywords: " + ", ".join(c.keywords[:10]))
        parts.append(body[:_MAX_DOC_CHARS])
        return ". ".join(parts)

    def _build_index(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — semantic scorer disabled."
            )
            return

        try:
            logger.info("Loading semantic model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)

            docs: list[str] = []
            emb_to_clause: list[int] = []
            skipped = 0

            for idx, clause in enumerate(self.clauses):
                text = self._clause_to_text(clause)
                if text is None:
                    skipped += 1
                    continue
                docs.append(text)
                emb_to_clause.append(idx)

            self._emb_to_clause = np.array(emb_to_clause, dtype=np.int32)

            logger.info(
                "Encoding %d clauses (%d empty/stubs skipped) …",
                len(docs), skipped,
            )

            self._embeddings = self._model.encode(
                docs,
                show_progress_bar=False,
                batch_size=64,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

            logger.info(
                "Simple semantic index ready — %d clauses, dim %d",
                len(docs), self._embeddings.shape[1],
            )
        except Exception:
            logger.exception("Failed to build simple semantic index")
            self._model = None
            self._embeddings = None

    @property
    def available(self) -> bool:
        return self._model is not None and self._embeddings is not None

    def _encode_queries(self, queries: list[str]) -> np.ndarray:
        return self._model.encode(
            queries, normalize_embeddings=True, convert_to_numpy=True,
        )

    def search(self, query: str, top_k: int = 50) -> list[ScoredHit]:
        if not self.available:
            return []
        q_emb = self._encode_queries([query])
        sims = (self._embeddings @ q_emb.T).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [
            ScoredHit(
                index=int(self._emb_to_clause[i]),
                score=max(0.0, float(sims[i])),
            )
            for i in top_idx
        ]

    def search_multi(self, queries: list[str], top_k: int = 50) -> list[ScoredHit]:
        if not self.available or not queries:
            return []
        q_embs = self._encode_queries(queries)
        sims_matrix = self._embeddings @ q_embs.T
        max_sims = sims_matrix.max(axis=1)
        top_idx = np.argsort(max_sims)[::-1][:top_k]
        return [
            ScoredHit(
                index=int(self._emb_to_clause[i]),
                score=max(0.0, float(max_sims[i])),
            )
            for i in top_idx
        ]
