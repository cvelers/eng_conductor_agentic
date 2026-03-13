from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace
import types

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _OpenAIStub:
        pass

    openai_stub.OpenAI = _OpenAIStub
    sys.modules["openai"] = openai_stub

from backend.agent.context import (
    convert_frontend_history,
    last_assistant_message_waiting_for_user,
    should_continue_from_ask_user,
    split_visible_and_tool_context,
)
from backend.agent.prompt import SYSTEM_PROMPT
from backend.agent.loop import (
    _GROUNDING_VALIDATOR_PROMPT,
    _build_conversation_history_for_validator,
    _build_tool_results_for_validator,
    _build_tool_context,
    run_agent_loop,
)


def test_convert_frontend_history_restores_structured_tool_context() -> None:
    tool_context = (
        "<tool-context>\n"
        '[tool_call] engineering_calculator({"tool_name":"ec3_profile_i_lookup"})\n'
        "[tool_result] engineering_calculator:\n"
        '{"outputs":{"h_mm":300}}\n'
        "</tool-context>"
    )
    history = [
        {
            "role": "assistant",
            "content": "IPE300 properties loaded.",
            "response_payload": {
                "answer": "IPE300 properties loaded.",
                "session_memory": {
                    "tool_context": tool_context,
                    "state": "final",
                },
            },
        }
    ]

    converted = convert_frontend_history(history)

    assert converted == [{
        "role": "assistant",
        "content": "IPE300 properties loaded.\n\n" + tool_context,
    }]


def test_waiting_for_user_detected_from_structured_session_memory() -> None:
    history = [
        {
            "role": "assistant",
            "content": "I need more information to continue.",
            "response_payload": {
                "session_memory": {
                    "state": "waiting_for_user",
                    "tool_context": (
                        "<tool-context>\n"
                        '[tool_call] ask_user({"question":"What is the unbraced length?"})\n'
                        "[tool_result] ask_user:\n"
                        '{"status":"waiting_for_user"}\n'
                        "</tool-context>"
                    ),
                },
            },
        }
    ]

    assert last_assistant_message_waiting_for_user(history) is True


def test_ask_user_continuation_requires_explicit_reply_flag() -> None:
    history = [
        {
            "role": "assistant",
            "content": "I need more information to continue.",
            "response_payload": {
                "session_memory": {
                    "state": "waiting_for_user",
                    "tool_context": (
                        "<tool-context>\n"
                        '[tool_call] ask_user({"question":"What is the unbraced length?"})\n'
                        "[tool_result] ask_user:\n"
                        '{"status":"waiting_for_user"}\n'
                        "</tool-context>"
                    ),
                },
            },
        }
    ]

    assert should_continue_from_ask_user(history, False) is False
    assert should_continue_from_ask_user(history, True) is True


def test_split_visible_and_tool_context_strips_raw_tool_blocks() -> None:
    visible, tool_context = split_visible_and_tool_context(
        "Classification summary.\n[tool_call] todo_write({\"todos\":[]})"
    )

    assert visible == "Classification summary."
    assert tool_context.startswith("[tool_call] todo_write(")


def test_build_tool_context_keeps_only_selected_search_clauses() -> None:
    all_messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tc_search",
                    "function": {
                        "name": "eurocode_search",
                        "arguments": json.dumps({"query": "ltb"}),
                    },
                },
                {
                    "id": "tc_calc",
                    "function": {
                        "name": "engineering_calculator",
                        "arguments": json.dumps({"tool_name": "ec3_ltb_check"}),
                    },
                },
                {
                    "id": "tc_ask",
                    "function": {
                        "name": "ask_user",
                        "arguments": json.dumps({"question": "What is the unbraced length?"}),
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tc_search",
            "content": json.dumps({
                "clauses": [
                    {
                        "clause_id": "6.3.2",
                        "standard": "EN 1993-1-1",
                        "text": "Selected clause",
                        "selected": True,
                        "score": 9.0,
                    },
                    {
                        "clause_id": "6.2.5",
                        "standard": "EN 1993-1-1",
                        "text": "Unselected clause",
                        "selected": False,
                        "score": 9.5,
                    },
                ],
                "total_found": 2,
            }),
        },
        {
            "role": "tool",
            "tool_call_id": "tc_calc",
            "content": json.dumps({
                "outputs": {"Mb_Rd_kNm": 94.1},
                "clause_references": ["EN 1993-1-1 §6.3.2"],
            }),
        },
        {
            "role": "tool",
            "tool_call_id": "tc_ask",
            "content": json.dumps({
                "status": "waiting_for_user",
                "question": "What is the unbraced length?",
            }),
        },
    ]

    tool_context = _build_tool_context(all_messages)

    assert "Selected clause" in tool_context
    assert "Unselected clause" not in tool_context
    assert "clause_references" not in tool_context
    assert "[tool_call] ask_user(" in tool_context
    assert "[tool_result] ask_user" in tool_context


def test_build_tool_context_includes_last_todo_write() -> None:
    """todo_write should appear in tool_context (only the last call)."""
    all_messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tc_todo1",
                    "function": {
                        "name": "todo_write",
                        "arguments": json.dumps({"todos": [
                            {"id": "search", "text": "Search clauses", "status": "in_progress"},
                            {"id": "calc", "text": "Calculate Mb,Rd", "status": "pending"},
                        ]}),
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tc_todo1",
            "content": json.dumps({"status": "ok", "plan": []}),
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tc_search",
                    "function": {
                        "name": "eurocode_search",
                        "arguments": json.dumps({"query": "bending"}),
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tc_search",
            "content": json.dumps({"clauses": [], "total_found": 0}),
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tc_todo2",
                    "function": {
                        "name": "todo_write",
                        "arguments": json.dumps({"todos": [
                            {"id": "search", "text": "Search clauses", "status": "done"},
                            {"id": "calc", "text": "Calculate Mb,Rd", "status": "in_progress"},
                        ]}),
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tc_todo2",
            "content": json.dumps({"status": "ok", "plan": []}),
        },
    ]

    tool_context = _build_tool_context(all_messages)

    # Only the LAST todo_write should appear (with "done" + "in_progress")
    assert "[tool_call] todo_write(" in tool_context
    assert "[tool_result] todo_write:" in tool_context
    # The last plan has "done" for search and "in_progress" for calc
    assert '"status": "done"' in tool_context or '"status":"done"' in tool_context
    # Should NOT have two separate todo_write call blocks
    assert tool_context.count("[tool_call] todo_write(") == 1


def test_build_tool_context_strips_clause_references_from_search_tools() -> None:
    """search_engineering_tools clause_references should be stripped too."""
    all_messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tc_st",
                    "function": {
                        "name": "search_engineering_tools",
                        "arguments": json.dumps({"query": "ltb"}),
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tc_st",
            "content": json.dumps({
                "results": [
                    {
                        "name": "ec3_ltb_check",
                        "description": "LTB check",
                        "clause_references": ["EN 1993-1-1 §6.3.2"],
                    },
                ],
                "total_found": 1,
            }),
        },
    ]

    tool_context = _build_tool_context(all_messages)

    assert "clause_references" not in tool_context
    assert "ec3_ltb_check" in tool_context


def test_validator_view_strips_clause_references_from_engineering_tools() -> None:
    all_messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tc_calc",
                    "function": {
                        "name": "engineering_calculator",
                        "arguments": json.dumps({"tool_name": "ec3_ltb_check"}),
                    },
                },
                {
                    "id": "tc_search_tools",
                    "function": {
                        "name": "search_engineering_tools",
                        "arguments": json.dumps({"query": "ltb"}),
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tc_calc",
            "content": json.dumps({
                "outputs": {"M_b,Rd [kNm]": 94.1},
                "clause_references": ["EN 1993-1-1 §6.3.2"],
            }),
        },
        {
            "role": "tool",
            "tool_call_id": "tc_search_tools",
            "content": json.dumps({
                "results": [
                    {
                        "name": "ec3_ltb_check",
                        "clause_references": ["EN 1993-1-1 §6.3.2"],
                    },
                ],
            }),
        },
    ]

    validator_view = _build_tool_results_for_validator(all_messages)

    assert "clause_references" not in validator_view
    assert "ec3_ltb_check" in validator_view


def test_prompts_require_preserving_demand_vs_resistance_semantics() -> None:
    assert "Never reuse a previously computed resistance or capacity value" in SYSTEM_PROMPT
    assert "M_c,Rd = 223.08 kNm" in SYSTEM_PROMPT
    assert "Flag demand/resistance swaps" in _GROUNDING_VALIDATOR_PROMPT
    assert "Do NOT treat a previously validated resistance as evidence for a demand value" in _GROUNDING_VALIDATOR_PROMPT


def test_validator_view_preserves_full_tool_output_when_validator_content_exists() -> None:
    long_value = "X" * 7000
    all_messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tc_calc",
                    "function": {
                        "name": "engineering_calculator",
                        "arguments": json.dumps({"tool_name": "ec3_ltb_check"}),
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tc_calc",
            "content": '{"outputs":{"preview":"short"}}<system-reminder>ignored</system-reminder>',
            "validator_content": json.dumps({"outputs": {"full": long_value}}),
        },
    ]

    validator_view = _build_tool_results_for_validator(all_messages)

    assert long_value in validator_view
    assert "... (truncated)" not in validator_view


def test_validator_conversation_history_preserves_full_prior_messages() -> None:
    user_text = "U" * 900
    assistant_text = "A" * 3500
    history = _build_conversation_history_for_validator([
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ])

    assert user_text in history
    assert assistant_text in history


def test_run_agent_loop_stops_after_ask_user_even_if_more_tools_were_emitted() -> None:
    class FakeCompletions:
        def __init__(self, response: object) -> None:
            self._response = response

        def create(self, **_: object) -> object:
            return self._response

    class FakeClient:
        def __init__(self, response: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions(response))

    def tool_call(name: str, args: dict[str, object], call_id: str) -> object:
        return SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(name=name, arguments=json.dumps(args)),
        )

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        tool_call(
                            "ask_user",
                            {"question": "What is the unbraced length?"},
                            "tc_ask",
                        ),
                        tool_call(
                            "engineering_calculator",
                            {"tool_name": "ec3_ltb_check", "params": {"L_mm": 5000}},
                            "tc_calc",
                        ),
                    ],
                )
            )
        ]
    )

    dispatched: list[str] = []

    def dispatcher(tool_name: str, args: dict[str, object]) -> str:
        dispatched.append(tool_name)
        if tool_name == "ask_user":
            return json.dumps({"question": args["question"], "status": "waiting_for_user"})
        return json.dumps({"unexpected": tool_name})

    async def collect() -> list[dict]:
        events: list[dict] = []
        async for event in run_agent_loop(
            client=FakeClient(response),
            model="fake",
            system_prompt="system",
            messages=[{"role": "user", "content": "what about ltb"}],
            tools=[],
            tool_dispatcher=dispatcher,
            max_rounds=1,
            grounding_validation=False,
        ):
            events.append(event)
        return events

    events = asyncio.run(collect())

    assert dispatched == ["ask_user"]
    assert any(event.get("type") == "ask_user" for event in events)


def test_run_agent_loop_discards_assistant_text_from_ask_user_round() -> None:
    class FakeCompletions:
        def __init__(self, response: object) -> None:
            self._response = response

        def create(self, **_: object) -> object:
            return self._response

    class FakeClient:
        def __init__(self, response: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions(response))

    def tool_call(name: str, args: dict[str, object], call_id: str) -> object:
        return SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(name=name, arguments=json.dumps(args)),
        )

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="I will assume a 5 m length and continue with LTB.",
                    tool_calls=[
                        tool_call(
                            "ask_user",
                            {"question": "What is the unbraced length?"},
                            "tc_ask",
                        ),
                    ],
                )
            )
        ]
    )

    def dispatcher(tool_name: str, args: dict[str, object]) -> str:
        if tool_name == "ask_user":
            return json.dumps({"question": args["question"], "status": "waiting_for_user"})
        raise AssertionError(f"Unexpected tool call: {tool_name}")

    async def collect() -> list[dict]:
        events: list[dict] = []
        async for event in run_agent_loop(
            client=FakeClient(response),
            model="fake",
            system_prompt="system",
            messages=[{"role": "user", "content": "what about ltb"}],
            tools=[],
            tool_dispatcher=dispatcher,
            max_rounds=1,
            grounding_validation=False,
        ):
            events.append(event)
        return events

    events = asyncio.run(collect())

    assert not any(event.get("type") == "delta" for event in events)
    assert any(event.get("type") == "ask_user" for event in events)
    assert events[-1] == {"type": "done", "content": ""}


def test_run_agent_loop_final_answer_excludes_intermediate_tool_round_text() -> None:
    class FakeCompletions:
        def __init__(self, responses: list[object]) -> None:
            self._responses = list(responses)

        def create(self, **_: object) -> object:
            if not self._responses:
                raise AssertionError("No more fake responses configured")
            return self._responses.pop(0)

    class FakeClient:
        def __init__(self, responses: list[object]) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions(responses))

    def tool_call(name: str, args: dict[str, object], call_id: str) -> object:
        return SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(name=name, arguments=json.dumps(args)),
        )

    first_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="I am checking the section properties now.",
                    tool_calls=[
                        tool_call(
                            "engineering_calculator",
                            {"tool_name": "ec3_profile_i_lookup", "params": {"section": "IPE300"}},
                            "tc_calc",
                        ),
                    ],
                )
            )
        ]
    )
    second_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Final grounded answer.",
                    tool_calls=[],
                )
            )
        ]
    )

    def dispatcher(tool_name: str, _: dict[str, object]) -> str:
        if tool_name == "engineering_calculator":
            return json.dumps({"outputs": {"h_mm": 300}})
        raise AssertionError(f"Unexpected tool call: {tool_name}")

    async def collect() -> list[dict]:
        events: list[dict] = []
        async for event in run_agent_loop(
            client=FakeClient([first_response, second_response]),
            model="fake",
            system_prompt="system",
            messages=[{"role": "user", "content": "check bending"}],
            tools=[],
            tool_dispatcher=dispatcher,
            max_rounds=2,
            grounding_validation=False,
        ):
            events.append(event)
        return events

    events = asyncio.run(collect())

    deltas = [event["content"] for event in events if event.get("type") == "delta"]
    assert deltas == ["Final grounded answer."]
    assert events[-1] == {"type": "done", "content": "Final grounded answer."}
