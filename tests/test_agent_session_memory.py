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
    compact_if_needed,
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

    assert converted == [
        {
            "role": "assistant",
            "content": "IPE300 properties loaded.",
        },
        {
            "role": "system",
            "content": tool_context,
        },
    ]


def test_convert_frontend_history_only_injects_latest_structured_memory() -> None:
    history = [
        {
            "role": "assistant",
            "content": "Earlier answer.",
            "response_payload": {
                "session_memory": {
                    "tool_context": "<tool-context>\nold tool context\n</tool-context>",
                    "state": "final",
                },
            },
        },
        {
            "role": "user",
            "content": "what about ltb",
        },
        {
            "role": "assistant",
            "content": "LTB depends on restraint.",
            "response_payload": {
                "session_memory": {
                    "state": "final",
                    "task_anchor": "Given IPE300 and S355, check bending resistance.",
                    "preferred_standard": "EN 1993-1-1",
                    "selected_clauses": [
                        {
                            "standard": "EN 1993-1-1",
                            "clause_id": "6.3.2",
                            "title": "Lateral torsional buckling",
                        },
                    ],
                    "recent_tool_results": [
                        {
                            "tool": "engineering_calculator",
                            "summary": "Mb_Rd_kNm=94.1",
                        },
                    ],
                },
            },
        },
    ]

    converted = convert_frontend_history(history)

    assert converted[0] == {"role": "assistant", "content": "Earlier answer."}
    assert not any(
        "old tool context" in msg.get("content", "")
        for msg in converted
    )
    assert converted[-1]["role"] == "system"
    assert "Preferred standard: EN 1993-1-1" in converted[-1]["content"]
    assert "engineering_calculator: Mb_Rd_kNm=94.1" in converted[-1]["content"]


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


def test_compact_if_needed_inserts_system_summary_message() -> None:
    messages = [
        {"role": "user", "content": "Question 1"},
        {"role": "assistant", "content": "Answer 1"},
        {"role": "user", "content": "Question 2"},
        {"role": "assistant", "content": "Answer 2"},
        {"role": "user", "content": "Question 3"},
        {"role": "assistant", "content": "Answer 3"},
        {"role": "user", "content": "Question 4"},
        {"role": "assistant", "content": "Answer 4"},
    ]

    compacted = compact_if_needed(messages, "system prompt", context_window=50)

    assert compacted[0]["role"] == "system"
    assert "Conversation memory (auto-compacted):" in compacted[0]["content"]


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
    assert "Topic continuity is not evidence." in _GROUNDING_VALIDATOR_PROMPT


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


def test_validator_conversation_history_includes_structured_session_memory() -> None:
    history = _build_conversation_history_for_validator([
        {
            "role": "system",
            "content": (
                "Continuation memory:\n"
                "Preferred standard: EN 1993-1-1\n"
                "Recent tool results:\n"
                "- math_calculator: Mcr_kNm=731.79"
            ),
        },
        {"role": "assistant", "content": "Previous grounded answer."},
    ])

    assert "[SESSION MEMORY]" in history
    assert "Preferred standard: EN 1993-1-1" in history


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


def test_run_agent_loop_does_not_replay_hidden_tool_round_text_in_next_request() -> None:
    class FakeCompletions:
        def __init__(self, responses: list[object]) -> None:
            self._responses = list(responses)
            self.requests: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.requests.append(kwargs)
            if not self._responses:
                raise AssertionError("No more fake responses configured")
            return self._responses.pop(0)

    class FakeClient:
        def __init__(self, completions: FakeCompletions) -> None:
            self.chat = SimpleNamespace(completions=completions)

    def tool_call(name: str, args: dict[str, object], call_id: str) -> object:
        return SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(name=name, arguments=json.dumps(args)),
        )

    first_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="I think the answer is probably around 220 kNm.",
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

    completions = FakeCompletions([first_response, second_response])
    client = FakeClient(completions)

    def dispatcher(tool_name: str, _: dict[str, object]) -> str:
        if tool_name == "engineering_calculator":
            return json.dumps({"outputs": {"h_mm": 300}})
        raise AssertionError(f"Unexpected tool call: {tool_name}")

    async def collect() -> None:
        async for _ in run_agent_loop(
            client=client,
            model="fake",
            system_prompt="system",
            messages=[{"role": "user", "content": "check bending"}],
            tools=[],
            tool_dispatcher=dispatcher,
            max_rounds=2,
            grounding_validation=False,
        ):
            pass

    asyncio.run(collect())

    second_request_messages = completions.requests[1]["messages"]
    assistant_with_tool_calls = next(
        msg for msg in second_request_messages
        if isinstance(msg, dict) and msg.get("tool_calls")
    )

    assert "content" not in assistant_with_tool_calls


def test_run_agent_loop_self_review_rejects_ungrounded_no_tool_answer_and_forces_evidence() -> None:
    class FakeCompletions:
        def __init__(self, responses: list[object]) -> None:
            self._responses = list(responses)
            self.requests: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.requests.append(kwargs)
            if not self._responses:
                raise AssertionError("No more fake responses configured")
            return self._responses.pop(0)

    class FakeClient:
        def __init__(self, completions: FakeCompletions) -> None:
            self.chat = SimpleNamespace(completions=completions)

    class FakeValidatorCompletions:
        def __init__(self, verdicts: list[str]) -> None:
            self._verdicts = list(verdicts)
            self.requests: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.requests.append(kwargs)
            if not self._verdicts:
                raise AssertionError("No more fake validator verdicts configured")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=self._verdicts.pop(0))
                    )
                ]
            )

    class FakeValidatorClient:
        def __init__(self, completions: FakeValidatorCompletions) -> None:
            self.chat = SimpleNamespace(completions=completions)

    def tool_call(name: str, args: dict[str, object], call_id: str) -> object:
        return SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(name=name, arguments=json.dumps(args)),
        )

    draft_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="In Eurocode 3, shear is covered by the usual web shear checks.",
                    tool_calls=[],
                )
            )
        ]
    )
    self_review_reject = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps({
                        "answer_type": "technical_or_factual",
                        "requires_validation": True,
                        "requires_tools_before_answering": True,
                        "reason": "The draft makes technical claims without retrieved evidence.",
                    }),
                )
            )
        ]
    )
    search_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        tool_call(
                            "eurocode_search",
                            {"query": "shear resistance EN 1993-1-1"},
                            "tc_search",
                        ),
                    ],
                )
            )
        ]
    )
    grounded_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Grounded shear answer.",
                    tool_calls=[],
                )
            )
        ]
    )
    self_review_accept = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps({
                        "answer_type": "technical_or_factual",
                        "requires_validation": True,
                        "requires_tools_before_answering": False,
                        "reason": "This is a technical answer and should be validated.",
                    }),
                )
            )
        ]
    )

    main_completions = FakeCompletions([
        draft_response,
        self_review_reject,
        search_response,
        grounded_response,
        self_review_accept,
    ])
    client = FakeClient(main_completions)
    validator_completions = FakeValidatorCompletions(['{"valid": true}'])
    validator_client = FakeValidatorClient(validator_completions)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "eurocode_search",
                "description": "Search Eurocode clauses.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
    ]

    def dispatcher(tool_name: str, _: dict[str, object]) -> str:
        if tool_name == "eurocode_search":
            return json.dumps({
                "clauses": [
                    {
                        "standard": "EN 1993-1-1",
                        "clause_id": "6.2.6",
                        "title": "Shear",
                        "selected": True,
                    },
                ],
                "total_found": 1,
            })
        raise AssertionError(f"Unexpected tool call: {tool_name}")

    async def collect() -> list[dict]:
        events: list[dict] = []
        async for event in run_agent_loop(
            client=client,
            model="fake",
            system_prompt="system",
            messages=[{"role": "user", "content": "ok now give me everything about sher in ec 3"}],
            tools=tools,
            tool_dispatcher=dispatcher,
            max_rounds=3,
            grounding_validation=True,
            validator_client=validator_client,
            validator_model="validator-fake",
        ):
            events.append(event)
        return events

    events = asyncio.run(collect())

    deltas = [event["content"] for event in events if event.get("type") == "delta"]

    assert main_completions.requests[0]["tool_choice"] == "auto"
    assert main_completions.requests[2]["tool_choice"] == "required"
    assert len(validator_completions.requests) == 1
    assert any(
        event.get("type") == "tool_start" and event.get("tool") == "grounding_validator"
        for event in events
    )
    assert any(
        event.get("type") == "tool_start" and event.get("tool") == "eurocode_search"
        for event in events
    )
    assert deltas == ["Grounded shear answer."]


def test_run_agent_loop_allows_history_only_followup_without_forced_research() -> None:
    class FakeCompletions:
        def __init__(self, responses: list[object]) -> None:
            self._responses = list(responses)
            self.requests: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.requests.append(kwargs)
            if not self._responses:
                raise AssertionError("No more fake responses configured")
            return self._responses.pop(0)

    class FakeClient:
        def __init__(self, completions: FakeCompletions) -> None:
            self.chat = SimpleNamespace(completions=completions)

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Using the previous calculation, M_cr = 731.79 kNm and λ̄_LT = 0.746.",
                    tool_calls=[],
                )
            )
        ]
    )

    self_review = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps({
                        "answer_type": "technical_or_factual",
                        "requires_validation": False,
                        "requires_tools_before_answering": False,
                        "reason": "This technical follow-up only restates previously validated context.",
                    }),
                )
            )
        ]
    )

    completions = FakeCompletions([response, self_review])
    client = FakeClient(completions)

    async def collect() -> list[dict]:
        events: list[dict] = []
        async for event in run_agent_loop(
            client=client,
            model="fake",
            system_prompt="system",
            messages=[
                {"role": "assistant", "content": "Previous grounded answer."},
                {
                    "role": "system",
                    "content": (
                        "Continuation memory:\n"
                        "Preferred standard: EN 1993-1-1\n"
                        "Recent tool results:\n"
                        "- math_calculator: Mcr_kNm=731.79, lambda_LT_bar=0.746"
                    ),
                },
                {"role": "user", "content": "Can you show your M_cr calculation from your previous calcs?"},
            ],
            tools=[],
            tool_dispatcher=lambda *_: "{}",
            max_rounds=1,
            grounding_validation=True,
        ):
            events.append(event)
        return events

    events = asyncio.run(collect())

    assert len(completions.requests) == 2
    assert completions.requests[0]["tool_choice"] == "auto"
    assert not any(event.get("type") == "tool_start" for event in events)
    assert events[-1] == {
        "type": "done",
        "content": "Using the previous calculation, M_cr = 731.79 kNm and λ̄_LT = 0.746.",
    }


def test_run_agent_loop_skips_validator_for_conversation_meta_question() -> None:
    class FakeCompletions:
        def __init__(self, responses: list[object]) -> None:
            self._responses = list(responses)
            self.requests: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.requests.append(kwargs)
            if not self._responses:
                raise AssertionError("No more fake responses configured")
            return self._responses.pop(0)

    class FakeClient:
        def __init__(self, completions: FakeCompletions) -> None:
            self.chat = SimpleNamespace(completions=completions)

    class FakeValidatorCompletions:
        def __init__(self) -> None:
            self.requests: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.requests.append(kwargs)
            raise AssertionError("Validator should not be called for conversation meta questions")

    class FakeValidatorClient:
        def __init__(self, completions: FakeValidatorCompletions) -> None:
            self.chat = SimpleNamespace(completions=completions)

    final_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='The first thing you asked me was: "Given IPE300, S355, what is the bending resistance? Assume typical parameters if missing."',
                    tool_calls=[],
                )
            )
        ]
    )
    self_review = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps({
                        "answer_type": "conversation_meta",
                        "requires_validation": False,
                        "requires_tools_before_answering": False,
                        "reason": "This only recalls the chat history.",
                    }),
                )
            )
        ]
    )

    completions = FakeCompletions([final_response, self_review])
    client = FakeClient(completions)
    validator_completions = FakeValidatorCompletions()
    validator_client = FakeValidatorClient(validator_completions)

    async def collect() -> list[dict]:
        events: list[dict] = []
        async for event in run_agent_loop(
            client=client,
            model="fake",
            system_prompt="system",
            messages=[{"role": "user", "content": "what was the first thing i asked you"}],
            tools=[],
            tool_dispatcher=lambda *_: "{}",
            max_rounds=1,
            grounding_validation=True,
            validator_client=validator_client,
            validator_model="validator-fake",
        ):
            events.append(event)
        return events

    events = asyncio.run(collect())

    assert len(completions.requests) == 2
    assert not any(event.get("tool") == "grounding_validator" for event in events if event.get("type") == "tool_start")
    assert events[-1]["type"] == "done"
