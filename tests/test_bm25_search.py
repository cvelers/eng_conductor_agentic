"""Tests for BM25F lexical search in AgenticRetriever."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.config import Settings
from backend.registries.document_registry import ClauseRecord
from backend.retrieval.agentic_search import AgenticRetriever, _IndexedClause

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_clause(
    *,
    doc_id: str = "doc1",
    standard: str = "EN 1993-1-1",
    clause_id: str = "1.1",
    clause_title: str = "General",
    text: str = "some clause text",
    keywords: list[str] | None = None,
) -> ClauseRecord:
    return ClauseRecord(
        doc_id=doc_id,
        doc_title="Test Document",
        standard=standard,
        clause_id=clause_id,
        clause_title=clause_title,
        text=text,
        keywords=keywords or [],
        pointer="p1",
    )


def _make_retriever(clauses: list[ClauseRecord]) -> AgenticRetriever:
    settings = Settings(project_root=PROJECT_ROOT)
    mock_provider = MagicMock()
    mock_provider.available = False
    return AgenticRetriever(
        settings=settings,
        search_provider=mock_provider,
        clauses=clauses,
    )


# ---- IDF tests ----


class TestBM25IDF:
    def test_idf_unknown_token_returns_zero(self):
        r = _make_retriever([_make_clause()])
        assert r._idf("nonexistent") == 0.0

    def test_idf_formula_matches_bm25(self):
        """Verify IDF uses Robertson-Sparck Jones: log((N-df+0.5)/(df+0.5)+1)."""
        clauses = [
            _make_clause(clause_id=str(i), text=f"word{i} common")
            for i in range(10)
        ]
        r = _make_retriever(clauses)
        n = 10
        # "common" appears in all 10 docs
        df_common = r._doc_freq.get("common", 0)
        expected = math.log((n - df_common + 0.5) / (df_common + 0.5) + 1.0)
        assert r._idf("common") == pytest.approx(expected)

    def test_idf_rare_term_higher_than_common(self):
        clauses = [
            _make_clause(clause_id="1", text="rare common"),
            _make_clause(clause_id="2", text="common everyday"),
            _make_clause(clause_id="3", text="common everyday"),
        ]
        r = _make_retriever(clauses)
        assert r._idf("rare") > r._idf("common")


# ---- TF saturation tests ----


class TestTFSaturation:
    def test_higher_tf_scores_higher(self):
        """A term appearing 5 times should score higher than appearing once."""
        clauses = [
            _make_clause(clause_id="1", text="steel"),
            _make_clause(clause_id="2", text="steel steel steel steel steel"),
        ]
        r = _make_retriever(clauses)
        results = r._search_lexical("steel", limit=10)
        scores = {rc.clause.clause_id: rc.score for rc in results}
        assert scores["2"] > scores["1"]

    def test_tf_saturation_not_linear(self):
        """tf=10 should not score 10x more than tf=1 (saturation)."""
        single = "steel"
        repeated = " ".join(["steel"] * 10)
        clauses = [
            _make_clause(clause_id="1", text=single),
            _make_clause(clause_id="2", text=repeated),
        ]
        r = _make_retriever(clauses)
        entry_1 = r._entries[0]
        entry_2 = r._entries[1]
        tf_1 = r._bm25f_tf("steel", entry_1)
        tf_10 = r._bm25f_tf("steel", entry_2)
        assert tf_10 > tf_1
        assert tf_10 < tf_1 * 5  # well below 10x due to saturation


# ---- Length normalization tests ----


class TestLengthNormalization:
    def test_short_doc_scores_higher_for_same_term(self):
        """With b>0, a short doc mentioning 'steel' should score higher
        than a long doc mentioning 'steel' once (length normalization)."""
        short_text = "steel beams"
        long_text = "steel " + " ".join(f"word{i}" for i in range(200))
        clauses = [
            _make_clause(clause_id="short", text=short_text),
            _make_clause(clause_id="long", text=long_text),
        ]
        r = _make_retriever(clauses)
        results = r._search_lexical("steel", limit=10)
        scores = {rc.clause.clause_id: rc.score for rc in results}
        assert scores["short"] > scores["long"]


# ---- Field weighting tests ----


class TestFieldWeighting:
    def test_title_match_scores_higher_than_text_match(self):
        """A term in the title (weight 3.0) should score higher than in text only."""
        clauses = [
            _make_clause(
                clause_id="title_match",
                clause_title="Steel resistance",
                text="general provisions for design",
            ),
            _make_clause(
                clause_id="text_match",
                clause_title="General provisions",
                text="steel resistance calculation method",
            ),
        ]
        r = _make_retriever(clauses)
        results = r._search_lexical("steel resistance", limit=10)
        scores = {rc.clause.clause_id: rc.score for rc in results}
        assert scores["title_match"] > scores["text_match"]

    def test_standard_field_contributes_to_score(self):
        """Querying for the standard name should boost matching documents."""
        clauses = [
            _make_clause(clause_id="1", standard="EN 1993-1-1", text="beams"),
            _make_clause(clause_id="2", standard="EN 1993-1-8", text="beams"),
        ]
        r = _make_retriever(clauses)
        results = r._search_lexical("EN 1993-1-1 beams", limit=10)
        scores = {rc.clause.clause_id: rc.score for rc in results}
        assert scores["1"] > scores["2"]


# ---- Score normalization tests ----


class TestScoreNormalization:
    def test_scores_in_zero_to_ten_range(self):
        clauses = [
            _make_clause(clause_id=str(i), text=f"word{i} common term")
            for i in range(20)
        ]
        r = _make_retriever(clauses)
        results = r._search_lexical("common term", limit=50)
        for rc in results:
            assert 0.0 <= rc.score <= 10.0, f"Score {rc.score} out of [0,10]"

    def test_top_result_gets_max_score(self):
        """With min-max normalization, the best result should get 10.0."""
        clauses = [
            _make_clause(clause_id="1", clause_title="Bending resistance", text="bending"),
            _make_clause(clause_id="2", text="some other text about bending"),
            _make_clause(clause_id="3", text="unrelated content"),
        ]
        r = _make_retriever(clauses)
        results = r._search_lexical("bending resistance", limit=10)
        assert len(results) >= 2
        assert results[0].score == 10.0

    def test_single_result_gets_neutral_score(self):
        """When only one result exists, score_range=0 so score should be 5.0."""
        clauses = [
            _make_clause(clause_id="1", text="unique_xyz_token"),
            _make_clause(clause_id="2", text="other content"),
        ]
        r = _make_retriever(clauses)
        results = r._search_lexical("unique_xyz_token", limit=10)
        assert len(results) == 1
        assert results[0].score == 5.0


# ---- Edge cases ----


class TestEdgeCases:
    def test_empty_query_returns_empty(self):
        r = _make_retriever([_make_clause()])
        assert r._search_lexical("", limit=10) == []

    def test_stopword_only_query_returns_empty(self):
        r = _make_retriever([_make_clause()])
        assert r._search_lexical("the is in of", limit=10) == []

    def test_no_matching_tokens_returns_empty(self):
        r = _make_retriever([_make_clause(text="steel beams")])
        assert r._search_lexical("concrete slabs", limit=10) == []

    def test_empty_corpus(self):
        r = _make_retriever([])
        assert r._search_lexical("anything", limit=10) == []

    def test_matched_terms_populated(self):
        clauses = [_make_clause(text="steel beam resistance")]
        r = _make_retriever(clauses)
        results = r._search_lexical("steel beam", limit=10)
        assert len(results) == 1
        assert set(results[0].matched_terms) == {"steel", "beam"}


# ---- Index construction tests ----


class TestIndexConstruction:
    def test_avg_field_lengths_computed(self):
        clauses = [
            _make_clause(clause_id="1", clause_title="Short", text="one two three"),
            _make_clause(clause_id="2", clause_title="Longer title here", text="four five"),
        ]
        r = _make_retriever(clauses)
        assert r._avg_title_len > 0
        assert r._avg_text_len > 0

    def test_entries_have_counter_fields(self):
        clauses = [_make_clause(text="steel steel beam")]
        r = _make_retriever(clauses)
        entry = r._entries[0]
        assert isinstance(entry.text_tf, Counter)
        assert entry.text_tf["steel"] == 2
        assert entry.text_tf["beam"] == 1

    def test_field_lengths_correct(self):
        clauses = [_make_clause(text="steel steel beam")]
        r = _make_retriever(clauses)
        entry = r._entries[0]
        assert entry.text_len == 3  # 3 tokens total (steel*2 + beam*1)


# ---- BM25F integration ----


class TestBM25FIntegration:
    def test_multi_term_query_additive(self):
        """Matching more query terms should score higher than fewer."""
        clauses = [
            _make_clause(clause_id="both", text="steel beam design"),
            _make_clause(clause_id="one", text="steel column design"),
        ]
        r = _make_retriever(clauses)
        results = r._search_lexical("steel beam", limit=10)
        scores = {rc.clause.clause_id: rc.score for rc in results}
        assert scores["both"] > scores["one"]

    def test_limit_respected(self):
        clauses = [
            _make_clause(clause_id=str(i), text=f"common word{i}")
            for i in range(50)
        ]
        r = _make_retriever(clauses)
        results = r._search_lexical("common", limit=5)
        assert len(results) <= 5
