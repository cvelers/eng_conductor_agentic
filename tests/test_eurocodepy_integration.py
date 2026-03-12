"""Tests for the eurocodepy integration: registry, search, dispatcher, and adapters."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from backend.eurocodepy.registry import ENGINEERING_TOOL_REGISTRY, TOOL_INDEX
from backend.eurocodepy.search import search_engineering_tools, list_categories
from backend.eurocodepy.dispatcher import execute_engineering_tool


# ── Registry tests ────────────────────────────────────────────────────


class TestRegistry:
    def test_has_entries(self):
        assert len(ENGINEERING_TOOL_REGISTRY) >= 10

    def test_entries_have_required_fields(self):
        for entry in ENGINEERING_TOOL_REGISTRY:
            assert entry.name, f"Missing name in entry"
            assert entry.category == "EC3", (
                f"Invalid category {entry.category} for {entry.name}"
            )
            assert entry.description, f"Missing description for {entry.name}"
            assert entry.parameters, f"Missing parameters for {entry.name}"
            assert entry.keywords, f"Missing keywords for {entry.name}"
            assert entry.handler_module, f"Missing handler_module for {entry.name}"
            assert entry.handler_function, f"Missing handler_function for {entry.name}"

    def test_no_duplicate_names(self):
        names = [e.name for e in ENGINEERING_TOOL_REGISTRY]
        assert len(names) == len(set(names)), f"Duplicate tool names: {names}"

    def test_tool_index_matches_registry(self):
        assert len(TOOL_INDEX) == len(ENGINEERING_TOOL_REGISTRY)
        for entry in ENGINEERING_TOOL_REGISTRY:
            assert TOOL_INDEX[entry.name] is entry

    def test_all_categories_present(self):
        categories = {e.category for e in ENGINEERING_TOOL_REGISTRY}
        assert categories == {"EC3"}


# ── Search tests ──────────────────────────────────────────────────────


class TestSearch:
    def test_finds_steel_bending(self):
        results = search_engineering_tools("steel bending capacity check")
        assert len(results) > 0
        names = [r["tool_name"] for r in results]
        assert any("ec3" in n for n in names)

    def test_finds_ipe_profile(self):
        results = search_engineering_tools("IPE profile section properties")
        assert len(results) > 0
        names = [r["tool_name"] for r in results]
        assert any("profile" in n for n in names)

    def test_category_filter(self):
        results = search_engineering_tools("buckling", category="EC3")
        assert all(r["category"] == "EC3" for r in results)

    def test_no_results_for_nonsense(self):
        results = search_engineering_tools("quantum chromodynamics laser sword")
        assert len(results) == 0

    def test_results_have_parameters(self):
        results = search_engineering_tools("steel buckling")
        for r in results:
            assert "parameters" in r
            assert "description" in r
            assert "tool_name" in r

    def test_max_results_limit(self):
        results = search_engineering_tools("steel", max_results=3)
        assert len(results) <= 3

    def test_list_categories(self):
        cats = list_categories()
        assert len(cats) == 1
        assert cats[0]["category"] == "EC3"


# ── Dispatcher tests ──────────────────────────────────────────────────


class TestDispatcher:
    def test_unknown_tool_returns_error(self):
        result = json.loads(execute_engineering_tool("nonexistent_tool", {}))
        assert "error" in result
        assert "available_tools" in result

    def test_ec3_steel_grade_lookup(self):
        result = json.loads(execute_engineering_tool(
            "ec3_steel_grade_lookup",
            {"grade": "S355"},
        ))
        assert "outputs" in result
        assert result["outputs"]["fy_MPa"] == 355

    def test_ec3_profile_i_lookup(self):
        result = json.loads(execute_engineering_tool(
            "ec3_profile_i_lookup",
            {"profile_name": "IPE300"},
        ))
        assert "outputs" in result
        assert result["outputs"]["Section"] == "IPE300"
        assert result["outputs"]["h"] == 30.0  # cm

    def test_ec3_bolt_lookup(self):
        result = json.loads(execute_engineering_tool(
            "ec3_bolt_lookup",
            {"diameter": "M20", "grade": "8.8"},
        ))
        assert "outputs" in result
        assert result["outputs"]["fub_MPa"] == 800.0

    def test_ec3_combined_check(self):
        result = json.loads(execute_engineering_tool(
            "ec3_combined_section_check",
            {
                "N_Ed": 100,
                "M_Ed": 50,
                "V_Ed": 30,
                "area": 53.81,
                "area_v": 25.68,
                "W_el": 557.1,
                "fy": 355,
            },
        ))
        assert "outputs" in result
        assert result["outputs"]["passed"] is True

    def test_ec3_ltb_check(self):
        result = json.loads(execute_engineering_tool(
            "ec3_ltb_check",
            {
                "f_y": 355,
                "E": 210000,
                "G": 81000,
                "gamma_M1": 1.0,
                "I_y": 8356e4,
                "I_z": 603.8e4,
                "W_el_z": 80.5e3,
                "I_w": 124260e6,
                "I_t": 19.75e4,
                "L": 6000,
                "M_Ed": 100,
            },
        ))
        assert "outputs" in result
        assert "Status" in result["outputs"]

    def test_ec3_euler_critical_force(self):
        result = json.loads(execute_engineering_tool(
            "ec3_elastic_critical_force",
            {"E": 210000, "I": 8356e4, "L": 6000},
        ))
        assert "outputs" in result
        assert result["outputs"]["result"] > 0

    def test_bad_params_returns_schema(self):
        result = json.loads(execute_engineering_tool(
            "ec3_steel_grade_lookup",
            {"wrong_param": "S355"},
        ))
        assert "error" in result
        assert "expected_parameters" in result

    def test_result_format_consistent(self):
        """All successful results should have inputs_used, outputs, clause_references, notes."""
        result = json.loads(execute_engineering_tool(
            "ec3_steel_grade_lookup",
            {"grade": "S355"},
        ))
        assert "inputs_used" in result
        assert "outputs" in result
        assert "clause_references" in result
        assert "notes" in result


# ── LLM-scored search tests ──────────────────────────────────────────


def _mock_llm_provider(response_json: dict) -> MagicMock:
    """Create a mock LLM provider that returns a fixed JSON response."""
    provider = MagicMock()
    provider.generate.return_value = json.dumps(response_json)
    return provider


class TestLLMSearch:
    def test_llm_scores_combined_check(self):
        """LLM scoring should find the right tool for a natural query."""
        scores = {
            "ec3_combined_section_check": 9,
            "ec3_ltb_check": 3,
            "ec3_flexural_buckling_check": 2,
            "ec3_elastic_critical_force": 1,
            "ec3_profile_i_lookup": 5,
            "ec3_profile_chs_lookup": 0,
            "ec3_profile_rhs_lookup": 0,
            "ec3_profile_shs_lookup": 0,
            "ec3_steel_grade_lookup": 4,
            "ec3_bolt_lookup": 0,
        }
        provider = _mock_llm_provider(scores)
        results = search_engineering_tools(
            "check my beam for bending", llm_provider=provider,
        )
        assert len(results) > 0
        # Top result should be combined check (score 9)
        assert results[0]["tool_name"] == "ec3_combined_section_check"
        # Only tools with score > 3 should appear
        names = [r["tool_name"] for r in results]
        assert "ec3_ltb_check" not in names  # score 3, not > 3
        assert "ec3_bolt_lookup" not in names  # score 0

    def test_llm_provider_called_correctly(self):
        """Verify the LLM provider is called with the right prompt structure."""
        provider = _mock_llm_provider({"ec3_combined_section_check": 8})
        search_engineering_tools("steel buckling", llm_provider=provider)
        provider.generate.assert_called_once()
        call_kwargs = provider.generate.call_args.kwargs
        assert "system_prompt" in call_kwargs
        assert "user_prompt" in call_kwargs
        assert call_kwargs["temperature"] == 0.0
        assert "ec3_combined_section_check" in call_kwargs["user_prompt"]

    def test_llm_fallback_on_failure(self):
        """If LLM fails, keyword search should kick in."""
        provider = MagicMock()
        provider.generate.side_effect = RuntimeError("API timeout")
        # Should not raise — falls back to keyword scoring
        results = search_engineering_tools(
            "steel buckling column", llm_provider=provider,
        )
        # Keyword fallback should still find buckling-related tools
        assert len(results) > 0
        names = [r["tool_name"] for r in results]
        assert any("buckling" in n for n in names)

    def test_llm_fallback_on_bad_json(self):
        """If LLM returns unparseable output, keyword fallback works."""
        provider = MagicMock()
        provider.generate.return_value = "This is not JSON at all"
        results = search_engineering_tools(
            "IPE profile lookup", llm_provider=provider,
        )
        assert len(results) > 0

    def test_no_llm_uses_keyword_search(self):
        """Without llm_provider, keyword search is used (backwards compatible)."""
        results = search_engineering_tools("steel buckling")
        assert len(results) > 0
        names = [r["tool_name"] for r in results]
        assert any("buckling" in n for n in names)

    def test_category_filter_with_llm(self):
        """Category filter should apply before LLM scoring."""
        scores = {"ec3_combined_section_check": 9}
        provider = _mock_llm_provider(scores)
        results = search_engineering_tools(
            "section check", category="EC3", llm_provider=provider,
        )
        assert all(r["category"] == "EC3" for r in results)
