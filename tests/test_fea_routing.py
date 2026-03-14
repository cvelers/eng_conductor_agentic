from __future__ import annotations

from backend.llm.base import LLMProvider
from backend.orchestrator.fea_routing import FEA_ROUTER_SYSTEM, classify_fea_route, should_route_to_fea


class StaticRouterLLM(LLMProvider):
    provider_name = "static-router"

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_user_prompt = ""

    @property
    def available(self) -> bool:
        return True

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        reasoning_effort: str | None = None,
    ) -> str:
        self.last_user_prompt = user_prompt
        return self.response


def test_fea_router_routes_frame_analysis_prompt_to_fea() -> None:
    llm = StaticRouterLLM('{"route":"fea","reason":"The user wants a global frame analysis model."}')

    assert should_route_to_fea(
        llm,
        "ok create analyze 2x2 bays frame on selfweight",
        [],
    ) is True


def test_fea_router_relies_on_classifier_even_for_explicit_fea_prompt() -> None:
    llm = StaticRouterLLM('{"route":"chat","reason":"Incorrect classifier response for test."}')

    assert should_route_to_fea(
        llm,
        "ok create analyze fea model 2x2 bays frame on selfweight",
        [],
    ) is False


def test_fea_router_routes_eurocode_lookup_to_chat() -> None:
    llm = StaticRouterLLM('{"route":"chat","reason":"This is a clause lookup and resistance check request."}')

    decision = classify_fea_route(
        llm,
        "what clause governs lateral torsional buckling in EN 1993-1-1?",
        [],
    )

    assert decision["route"] == "chat"


def test_fea_router_relies_on_classifier_for_prior_fea_history() -> None:
    llm = StaticRouterLLM('{"route":"chat","reason":"Classifier remains authoritative for follow-ups."}')
    history = [
        {
            "role": "assistant",
            "content": "FEA analysis complete.",
            "response_payload": {
                "answer": "FEA analysis complete.",
                "assumptions": ["2D frame idealised in the XY plane."],
                "tool_trace": [{"tool_name": "fea_solve", "status": "ok"}],
            },
        },
    ]

    assert should_route_to_fea(llm, "show the deformed shape", history) is False


def test_fea_router_passes_plain_history_without_prior_fea_hint() -> None:
    llm = StaticRouterLLM('{"route":"chat","reason":"Prompt inspection test."}')
    history = [
        {
            "role": "assistant",
            "content": "Frame solved.",
            "response_payload": {
                "answer": "FEA analysis complete.",
                "assumptions": ["2D frame idealised in the XY plane."],
                "tool_trace": [{"tool_name": "fea_solve", "status": "ok"}],
            },
        },
    ]

    classify_fea_route(llm, "show the deformed shape", history)

    assert "[prior_fea_turn]" not in llm.last_user_prompt
    assert "ASSISTANT: Frame solved." in llm.last_user_prompt


def test_fea_router_falls_back_to_chat_on_malformed_output() -> None:
    llm = StaticRouterLLM("not json")

    decision = classify_fea_route(
        llm,
        "analyse a portal frame under self-weight",
        [],
    )

    assert decision["route"] == "chat"


def test_fea_router_recovers_route_from_truncated_json_prefix() -> None:
    llm = StaticRouterLLM('{"route":"fea')

    decision = classify_fea_route(
        llm,
        "create fea model of simply supported beam",
        [],
    )

    assert decision["route"] == "fea"


def test_fea_router_prompt_explicitly_allows_switch_from_design_check_to_fea() -> None:
    assert "overrides prior member-level chat context" in FEA_ROUTER_SYSTEM
    assert "explicitly ask to build, create, or analyse a structural model" in FEA_ROUTER_SYSTEM
