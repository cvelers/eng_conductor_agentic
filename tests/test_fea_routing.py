from __future__ import annotations

from backend.llm.base import LLMProvider
from backend.orchestrator.fea_routing import classify_fea_route, should_route_to_fea


class StaticRouterLLM(LLMProvider):
    provider_name = "static-router"

    def __init__(self, response: str) -> None:
        self.response = response

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
        return self.response


def test_fea_router_routes_frame_analysis_prompt_to_fea() -> None:
    llm = StaticRouterLLM('{"route":"fea","reason":"The user wants a global frame analysis model."}')

    assert should_route_to_fea(
        llm,
        "ok create analyze 2x2 bays frame on selfweight",
        [],
    ) is True


def test_fea_router_explicit_fea_override_beats_chat_classifier() -> None:
    llm = StaticRouterLLM('{"route":"chat","reason":"Incorrect classifier response for test."}')

    assert should_route_to_fea(
        llm,
        "ok create analyze fea model 2x2 bays frame on selfweight",
        [],
    ) is True


def test_fea_router_routes_eurocode_lookup_to_chat() -> None:
    llm = StaticRouterLLM('{"route":"chat","reason":"This is a clause lookup and resistance check request."}')

    decision = classify_fea_route(
        llm,
        "what clause governs lateral torsional buckling in EN 1993-1-1?",
        [],
    )

    assert decision["route"] == "chat"


def test_fea_router_marks_followup_after_prior_fea_turn() -> None:
    llm = StaticRouterLLM('{"route":"fea","reason":"This is a follow-up to an existing FEA thread."}')
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

    assert should_route_to_fea(llm, "show the deformed shape", history) is True


def test_fea_router_does_not_treat_generic_chat_plan_as_prior_fea_turn() -> None:
    llm = StaticRouterLLM('{"route":"chat","reason":"This is an LTB follow-up to a member-level chat answer."}')
    history = [
        {
            "role": "assistant",
            "content": "The bending resistance is 223.08 kNm.",
            "response_payload": {
                "answer": "The bending resistance is 223.08 kNm.",
                "assumptions": ["The cross-section is Class 1 or 2."],
                "tool_trace": [
                    {"tool_name": "todo_write", "status": "ok"},
                    {"tool_name": "eurocode_search", "status": "ok"},
                    {"tool_name": "math_calculator", "status": "ok"},
                ],
            },
        },
    ]

    assert should_route_to_fea(llm, "what about ltb", history) is False


def test_fea_router_uses_structured_fea_session_memory_for_followups() -> None:
    llm = StaticRouterLLM('{"route":"chat","reason":"Override should detect prior FEA session."}')
    history = [
        {
            "role": "assistant",
            "content": "Frame solved.",
            "response_payload": {
                "answer": "Frame solved.",
                "session_memory": {
                    "state": "final",
                    "fea_session": {
                        "model_summary": "2D portal frame",
                    },
                },
            },
        },
    ]

    assert should_route_to_fea(llm, "show the moment diagram", history) is True


def test_fea_router_falls_back_to_chat_on_malformed_output() -> None:
    llm = StaticRouterLLM("not json")

    decision = classify_fea_route(
        llm,
        "analyse a portal frame under self-weight",
        [],
    )

    assert decision["route"] == "chat"
