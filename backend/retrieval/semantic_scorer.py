"""Bi-encoder semantic scorer for clause retrieval.

Uses sentence-transformers to embed all clauses at startup, then scores
queries against the pre-computed embeddings via cosine similarity.

Each clause embedding includes the standard name, clause ID, title,
keywords, and body text — so the model captures which standard a clause
belongs to, not just its content.

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

# Max chars of clause body to include in the embedding.
# MiniLM has a 256-token window (~1200 chars); we include a bit more
# because the tokenizer will truncate gracefully.
_MAX_DOC_CHARS = 1200


@dataclass(frozen=True)
class ScoredHit:
    """A clause index + cosine similarity score."""
    index: int
    score: float


class SemanticScorer:
    """Bi-encoder semantic similarity scorer.

    Encodes every clause (with standard context) at init time.
    At query time, encodes the query and computes cosine similarity
    against all clause embeddings via a single matrix multiply.
    """

    def __init__(
        self,
        clauses: list[ClauseRecord],
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.clauses = clauses
        self._model_name = model_name
        self._model = None
        self._embeddings: np.ndarray | None = None
        self._build_index()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    @staticmethod
    def _clause_to_text(c: ClauseRecord) -> str:
        """Build a rich text representation that front-loads the most
        discriminating information: standard, ID, title, keywords, then body."""
        parts = [
            c.standard,
            f"Clause {c.clause_id}",
            c.clause_title,
        ]
        if c.keywords:
            parts.append("Keywords: " + ", ".join(c.keywords[:10]))
        body = c.text[:_MAX_DOC_CHARS].replace("\n", " ")
        parts.append(body)
        return ". ".join(parts)

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

            docs = [self._clause_to_text(c) for c in self.clauses]
            logger.info("Encoding %d clauses …", len(docs))

            self._embeddings = self._model.encode(
                docs,
                show_progress_bar=False,
                batch_size=128,
                normalize_embeddings=True,   # L2-normalized → dot = cosine
                convert_to_numpy=True,
            )

            logger.info(
                "Semantic index ready — %d clauses, embedding dim %d",
                len(docs),
                self._embeddings.shape[1],
            )
        except Exception:
            logger.exception("Failed to build semantic index")
            self._model = None
            self._embeddings = None

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._model is not None and self._embeddings is not None

    def search(self, query: str, top_k: int = 50) -> list[ScoredHit]:
        """Return top-k clause hits sorted by cosine similarity (desc).

        Scores are in [0, 1] range (negative similarities clamped to 0).
        """
        if not self.available:
            return []

        q_emb = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Dot product of L2-normalised vectors = cosine similarity
        sims = (self._embeddings @ q_emb.T).flatten()

        # Top-k by descending similarity
        top_idx = np.argsort(sims)[::-1][:top_k]

        return [
            ScoredHit(index=int(i), score=max(0.0, float(sims[i])))
            for i in top_idx
        ]

    def search_multi(
        self,
        queries: list[str],
        top_k: int = 50,
    ) -> list[ScoredHit]:
        """Search with multiple queries and merge results.

        For each clause, the final score is the max similarity across
        all queries — this captures the best-matching aspect.
        """
        if not self.available or not queries:
            return []

        q_embs = self._model.encode(
            queries,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # (n_clauses, n_queries) similarities
        sims_matrix = self._embeddings @ q_embs.T

        # Max similarity across queries for each clause
        max_sims = sims_matrix.max(axis=1)

        top_idx = np.argsort(max_sims)[::-1][:top_k]

        return [
            ScoredHit(index=int(i), score=max(0.0, float(max_sims[i])))
            for i in top_idx
        ]
