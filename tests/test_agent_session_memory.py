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
)
from backend.agent.loop import (
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
