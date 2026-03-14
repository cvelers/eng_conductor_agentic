from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.config import Settings
from backend.llm.base import LLMProvider
from backend.orchestrator.fea_analyst import FEAAnalystLoop


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ScriptedLLM(LLMProvider):
    provider_name = "scripted"

    def __init__(self, responses: list[dict | str]) -> None:
        self._responses = list(responses)

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
        raise AssertionError("generate() should not be called in these tests")

    def generate_messages(
        self,
        *,
        messages: list[dict[str, object]],
        temperature: float = 0.0,
        max_tokens: int = 8000,
        reasoning_effort: str | None = None,
        tools: list[dict[str, object]] | None = None,
    ) -> dict | str:
        if not self._responses:
            raise AssertionError("No scripted FEA response left")
        return self._responses.pop(0)


class RecordingScriptedLLM(ScriptedLLM):
    def __init__(self, responses: list[dict | str]) -> None:
        super().__init__(responses)
        self.calls: list[list[dict[str, object]]] = []
        self.tool_defs: list[list[dict[str, object]] | None] = []

    def generate_messages(
        self,
        *,
        messages: list[dict[str, object]],
        temperature: float = 0.0,
        max_tokens: int = 8000,
        reasoning_effort: str | None = None,
        tools: list[dict[str, object]] | None = None,
    ) -> dict | str:
        self.calls.append([dict(m) for m in messages])
        self.tool_defs.append(tools)
        return super().generate_messages(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            tools=tools,
        )


def _tool_call(name: str, args: dict[str, object]) -> dict[str, object]:
    return {
        "function": {
            "name": name,
            "arguments": json.dumps(args),
        },
    }


def _fea_history_payload() -> dict[str, object]:
    return {
        "session_memory": {
            "state": "final",
            "answer_summary": "Previous FEA summary.",
            "plan": [{"id": "report", "text": "Report the results", "status": "done"}],
            "assumptions": ["2D frame idealised in the XY plane."],
            "fea_session": {
                "version": 2,
                "authoring_state": {
                    "analysis_type": "beam2d",
                    "nodes": {
                        "N1": {"x": 0, "y": 0, "z": 0},
                        "N2": {"x": 5000, "y": 0, "z": 0},
                    },
                    "elements": {
                        "E1": {"type": "beam", "nodeIds": ["N1", "N2"], "sectionId": "sec_IPE300", "materialId": "mat_S355"},
                    },
                    "sections": {
                        "sec_IPE300": {"profileName": "IPE300", "A": 5381, "Iy": 83560000},
                    },
                    "materials": {
                        "mat_S355": {"name": "S355"},
                    },
                    "restraints": {
                        "N1": {"type": "fixed_2d"},
                    },
                    "load_cases": {
                        "LC1": {
                            "loads": [
                                {"type": "self_weight", "factor": 1.0, "direction": {"x": 0, "y": -1, "z": 0}},
                            ],
                        },
                    },
                    "solved": True,
                    "results": {
                        "reactions": {"N1": {"fy": 12000}},
                        "displacements": {"N2": {"dx": 0.0, "dy": -5.0, "dz": 0.0}},
                        "elementForces": {"E1": {"M": [0.0, 1000000.0]}},
                    },
                    "plan": [{"id": "report", "text": "Report the results", "status": "done"}],
                    "assumptions": ["2D frame idealised in the XY plane."],
                    "semantic_model": {
                        "kind": "rectilinear_frame",
                        "dimension": "2d",
                        "geometry": {"spans_x": [5000.0], "spans_z": [], "storey_heights": [3000.0]},
                        "member_families": {
                            "columns": {"profile_name": "HEB200"},
                            "beams_x": {"profile_name": "IPE300"},
                            "beams_z": {"profile_name": "IPE300"},
                        },
                        "material": {"grade": "S355"},
                        "supports": {"base": "fixed"},
                        "load_cases": [
                            {
                                "id": "LC1",
                                "name": "LC1",
                                "loads": [{"type": "self_weight", "factor": 1.0, "direction": {"x": 0, "y": -1, "z": 0}}],
                            },
                        ],
                    },
                },
                "results_snapshot": {
                    "reactions": {"N1": {"fy": 12000}},
                    "displacements": {"N2": {"dx": 0.0, "dy": -5.0, "dz": 0.0}},
                    "elementForces": {"E1": {"M": [0.0, 1000000.0]}},
                },
                "model_summary": {
                    "analysis_type": "beam2d",
                    "node_count": 2,
                    "element_count": 1,
                    "load_case_ids": ["LC1"],
                    "solved": True,
                },
                "semantic_model": {
                    "kind": "rectilinear_frame",
                    "dimension": "2d",
                    "geometry": {"spans_x": [5000.0], "spans_z": [], "storey_heights": [3000.0]},
                    "member_families": {
                        "columns": {"profile_name": "HEB200"},
                        "beams_x": {"profile_name": "IPE300"},
                        "beams_z": {"profile_name": "IPE300"},
                    },
                    "material": {"grade": "S355"},
                    "supports": {"base": "fixed"},
                    "load_cases": [
                        {
                            "id": "LC1",
                            "name": "LC1",
                            "loads": [{"type": "self_weight", "factor": 1.0, "direction": {"x": 0, "y": -1, "z": 0}}],
                        },
                    ],
                },
                "model_snapshot": {
                    "nodes": {
                        "N1": {"x": 0, "y": 0, "z": 0},
                        "N2": {"x": 5000, "y": 0, "z": 0},
                    },
                    "elements": {
                        "E1": {"type": "beam2d", "nodeIds": ["N1", "N2"], "sectionId": "sec_IPE300", "materialId": "mat_S355"},
                    },
                    "materials": {
                        "mat_S355": {"name": "S355", "E": 210000, "rho": 7.85e-6},
                    },
                    "sections": {
                        "sec_IPE300": {"profileName": "IPE300", "A": 5381, "Iy": 83560000},
                    },
                    "supports": {
                        "SUP_N1": {"nodeId": "N1", "conditions": {"dx": True, "dy": True, "dz": False, "rx": False, "ry": False, "rz": True}},
                    },
                    "restraints": {
                        "N1": {"dx": True, "dy": True, "dz": False, "rx": False, "ry": False, "rz": True},
                    },
                    "loadCases": {
                        "LC1": {
                            "name": "LC1",
                            "loads": [{"type": "self_weight", "factor": 1.0, "direction": {"x": 0, "y": -1, "z": 0}}],
                        },
                    },
                    "analysisType": "beam2d",
                },
            },
        },
    }


@pytest.mark.asyncio
async def test_fea_analyst_filters_batch_to_ask_user() -> None:
    llm = ScriptedLLM([
        {
            "tool_calls": [
                _tool_call(
                    "ask_user",
                    {
                        "question": "Are the base supports fixed or pinned?",
                        "options": ["Fixed", "Pinned"],
                        "context": "Base fixity controls frame stability.",
                    },
                ),
                _tool_call(
                    "fea_add_nodes",
                    {
                        "nodes": [
                            {"id": "N1", "x": 0, "y": 0},
                            {"id": "N2", "x": 6000, "y": 0},
                        ],
                    },
                ),
            ],
        },
    ])
    analyst = FEAAnalystLoop(llm=llm, settings=Settings(project_root=PROJECT_ROOT))

    events: list[tuple[str, dict[str, object]]] = []
    agen = analyst.run_stream("Analyse a portal frame")

    async for event_type, payload in agen:
        events.append((event_type, payload))
        if event_type == "fea_user_query":
            break

    await agen.aclose()

    tool_calls = [payload["tool"] for event_type, payload in events if event_type == "fea_tool_call"]
    assert tool_calls == ["ask_user"]


@pytest.mark.asyncio
async def test_fea_analyst_emits_plan_updates_and_continues_after_answer() -> None:
    llm = ScriptedLLM([
        {
            "tool_calls": [
                _tool_call(
                    "todo_write",
                    {
                        "todos": [
                            {"id": "model", "text": "Define the structural model", "status": "in_progress"},
                            {"id": "solve", "text": "Solve and report", "status": "pending"},
                        ],
                    },
                ),
            ],
        },
        {
            "tool_calls": [
                _tool_call(
                    "ask_user",
                    {
                        "question": "Are the base supports fixed or pinned?",
                        "options": ["Fixed", "Pinned"],
                        "context": "Base fixity controls frame stability.",
                    },
                ),
            ],
        },
        {
            "tool_calls": [
                _tool_call(
                    "todo_write",
                    {
                        "todos": [
                            {"id": "model", "text": "Define the structural model", "status": "done"},
                            {"id": "solve", "text": "Solve and report", "status": "in_progress"},
                        ],
                    },
                ),
            ],
        },
    ])
    analyst = FEAAnalystLoop(llm=llm, settings=Settings(project_root=PROJECT_ROOT))

    events: list[tuple[str, dict[str, object]]] = []
    agen = analyst.run_stream("Analyse a portal frame")

    async for event_type, payload in agen:
        events.append((event_type, payload))
        if event_type == "fea_user_query":
            analyst.provide_answer("Fixed bases")
        if event_type == "plan_update":
            break

    await agen.aclose()

    assert any(event_type == "plan" for event_type, _payload in events)
    assert any(
        event_type == "plan_update"
        and payload["step_id"] == "model"
        and payload["status"] == "done"
        for event_type, payload in events
    )
    ask_results = [
        payload for event_type, payload in events
        if event_type == "fea_tool_result" and payload["tool"] == "ask_user"
    ]
    assert ask_results
    answered = json.loads(str(ask_results[-1]["result"]))
    assert answered["answer"] == "Fixed bases"


@pytest.mark.asyncio
async def test_fea_complete_carries_recorded_assumptions() -> None:
    llm = ScriptedLLM([
        {
            "tool_calls": [
                _tool_call(
                    "fea_record_assumptions",
                    {"assumptions": ["2D frame idealised in the XY plane."]},
                ),
            ],
        },
        {
            "tool_calls": [
                _tool_call("fea_solve", {"load_case_id": "LC1"}),
            ],
        },
        {
            "tool_calls": [
                _tool_call("fea_get_results", {"query": "summary"}),
            ],
        },
        "Final engineering summary.",
    ])
    analyst = FEAAnalystLoop(llm=llm, settings=Settings(project_root=PROJECT_ROOT))

    events: list[tuple[str, dict[str, object]]] = []
    agen = analyst.run_stream("Analyse a simple beam")

    async for event_type, payload in agen:
        events.append((event_type, payload))
        if event_type == "fea_solve_request":
            analyst.provide_results(
                {
                    "solverInfo": {"dofCount": 6, "elementCount": 1, "solveTimeMs": 3},
                    "maxValues": {},
                    "reactions": {},
                    "displacements": {},
                    "elementForces": {},
                }
            )
        if event_type == "fea_complete":
            break

    await agen.aclose()

    complete_events = [payload for event_type, payload in events if event_type == "fea_complete"]
    assert complete_events
    assert complete_events[-1]["assumptions"] == ["2D frame idealised in the XY plane."]


@pytest.mark.asyncio
async def test_fea_analyst_requires_plan_before_real_tool_calls() -> None:
    llm = ScriptedLLM([
        {
            "tool_calls": [
                _tool_call("fea_set_analysis_type", {"type": "beam2d"}),
            ],
        },
        {
            "tool_calls": [
                _tool_call(
                    "todo_write",
                    {
                        "todos": [
                            {"id": "model", "text": "Build the structural model", "status": "in_progress"},
                            {"id": "solve", "text": "Solve and report", "status": "pending"},
                        ],
                    },
                ),
            ],
        },
    ])
    analyst = FEAAnalystLoop(llm=llm, settings=Settings(project_root=PROJECT_ROOT))

    events: list[tuple[str, dict[str, object]]] = []
    agen = analyst.run_stream("Analyse a simple frame")

    async for event_type, payload in agen:
        events.append((event_type, payload))
        if event_type == "plan":
            break

    await agen.aclose()

    tool_calls = [payload["tool"] for event_type, payload in events if event_type == "fea_tool_call"]
    assert tool_calls == ["todo_write"]
    assert any(
        event_type == "fea_thinking" and "Planning the analysis workflow" in str(payload.get("content", ""))
        for event_type, payload in events
    )


@pytest.mark.asyncio
async def test_fea_analyst_requires_full_results_before_summary() -> None:
    llm = ScriptedLLM([
        {
            "tool_calls": [
                _tool_call(
                    "todo_write",
                    {
                        "todos": [
                            {"id": "model", "text": "Solve the model", "status": "in_progress"},
                            {"id": "report", "text": "Report the results", "status": "pending"},
                        ],
                    },
                ),
            ],
        },
        {
            "tool_calls": [
                _tool_call("fea_solve", {"load_case_id": "LC1"}),
            ],
        },
        {
            "tool_calls": [
                _tool_call("fea_get_results", {"query": "displacements"}),
            ],
        },
        "Premature summary.",
        {
            "tool_calls": [
                _tool_call("fea_get_results", {"query": "reactions"}),
            ],
        },
        {
            "tool_calls": [
                _tool_call("fea_get_results", {"query": "element_forces"}),
            ],
        },
        "Final engineering summary.",
    ])
    analyst = FEAAnalystLoop(llm=llm, settings=Settings(project_root=PROJECT_ROOT))

    events: list[tuple[str, dict[str, object]]] = []
    agen = analyst.run_stream("Analyse a frame")

    async for event_type, payload in agen:
        events.append((event_type, payload))
        if event_type == "fea_solve_request":
            analyst.provide_results(
                {
                    "solverInfo": {"dofCount": 6, "elementCount": 1, "solveTimeMs": 2},
                    "maxValues": {},
                    "reactions": {},
                    "displacements": {},
                    "elementForces": {},
                }
            )
        if event_type == "fea_complete":
            break

    await agen.aclose()

    result_queries = [
        payload["tool"]
        for event_type, payload in events
        if event_type == "fea_tool_call" and payload["tool"] == "fea_get_results"
    ]
    assert len(result_queries) == 3
    assert any(
        event_type == "fea_thinking" and "Gathering full FEA results before reporting" in str(payload.get("content", ""))
        for event_type, payload in events
    )
    complete_events = [payload for event_type, payload in events if event_type == "fea_complete"]
    assert complete_events[-1]["summary"] == "Final engineering summary."


@pytest.mark.asyncio
async def test_fea_analyst_accepts_displacement_alias_for_required_results() -> None:
    llm = ScriptedLLM([
        {
            "tool_calls": [
                _tool_call(
                    "todo_write",
                    {
                        "todos": [
                            {"id": "model", "text": "Solve the model", "status": "in_progress"},
                            {"id": "report", "text": "Report the results", "status": "pending"},
                        ],
                    },
                ),
            ],
        },
        {
            "tool_calls": [
                _tool_call("fea_solve", {"load_case_id": "LC1"}),
            ],
        },
        {
            "tool_calls": [
                _tool_call("fea_get_results", {"query": "max_displacement"}),
            ],
        },
        {
            "tool_calls": [
                _tool_call("fea_get_results", {"query": "reactions"}),
            ],
        },
        {
            "tool_calls": [
                _tool_call("fea_get_results", {"query": "element_forces"}),
            ],
        },
        "Premature summary.",
        {
            "tool_calls": [
                _tool_call("fea_get_results", {"query": "all_displacements"}),
            ],
        },
        "Final engineering summary.",
    ])
    analyst = FEAAnalystLoop(llm=llm, settings=Settings(project_root=PROJECT_ROOT))

    events: list[tuple[str, dict[str, object]]] = []
    agen = analyst.run_stream("Analyse a frame")

    async for event_type, payload in agen:
        events.append((event_type, payload))
        if event_type == "fea_solve_request":
            analyst.provide_results(
                {
                    "solverInfo": {"dofCount": 6, "elementCount": 1, "solveTimeMs": 2},
                    "maxValues": {},
                    "reactions": {},
                    "displacements": {},
                    "elementForces": {},
                }
            )
        if event_type == "fea_complete":
            break

    await agen.aclose()

    result_queries = [
        payload["args"]["query"]
        for event_type, payload in events
        if event_type == "fea_tool_call" and payload["tool"] == "fea_get_results"
    ]
    assert result_queries == ["max_displacement", "reactions", "element_forces", "all_displacements"]
    assert sum(
        1
        for event_type, payload in events
        if event_type == "fea_thinking"
        and "Gathering full FEA results before reporting" in str(payload.get("content", ""))
    ) == 1
    complete_events = [payload for event_type, payload in events if event_type == "fea_complete"]
    assert complete_events[-1]["summary"] == "Final engineering summary."


@pytest.mark.asyncio
async def test_fea_analyst_retries_after_malformed_function_call() -> None:
    llm = RecordingScriptedLLM([
        {
            "content": "",
            "tool_calls": [],
            "finish_reason": "function_call_filter: malformed_function_call",
        },
        {
            "tool_calls": [
                _tool_call(
                    "todo_write",
                    {
                        "todos": [
                            {"id": "model", "text": "Build the structural model", "status": "in_progress"},
                        ],
                    },
                ),
            ],
        },
    ])
    analyst = FEAAnalystLoop(llm=llm, settings=Settings(project_root=PROJECT_ROOT))

    events: list[tuple[str, dict[str, object]]] = []
    agen = analyst.run_stream("Analyse a simple frame")

    async for event_type, payload in agen:
        events.append((event_type, payload))
        if event_type == "plan":
            break

    await agen.aclose()

    assert any(
        event_type == "fea_thinking" and "Retrying after a malformed tool call" in str(payload.get("content", ""))
        for event_type, payload in events
    )
    assert len(llm.calls) >= 2
    repair_prompt = llm.calls[1][-1]["content"]
    assert isinstance(repair_prompt, str)
    assert "malformed function/tool call" in repair_prompt
    assert any(event_type == "plan" for event_type, _payload in events)


@pytest.mark.asyncio
async def test_fea_analyst_restores_saved_model_for_follow_up() -> None:
    llm = RecordingScriptedLLM([
        {
            "tool_calls": [
                _tool_call("fea_get_results", {"query": "reactions"}),
            ],
        },
        "Follow-up summary.",
    ])
    analyst = FEAAnalystLoop(llm=llm, settings=Settings(project_root=PROJECT_ROOT))
    history = [
        {"role": "assistant", "content": "Previous FEA summary.", "response_payload": _fea_history_payload()},
    ]

    events: list[tuple[str, dict[str, object]]] = []
    agen = analyst.run_stream("show the reactions", history=history)

    async for event_type, payload in agen:
        events.append((event_type, payload))
        if event_type == "fea_complete":
            break

    await agen.aclose()

    assert any(event_type == "fea_state_restored" for event_type, _payload in events)
    assert not any(event_type == "fea_solve_request" for event_type, _payload in events)
    assert any(
        event_type == "fea_tool_call" and payload["tool"] == "fea_get_results" and payload["args"]["query"] == "reactions"
        for event_type, payload in events
    )
    assert len(llm.calls) >= 1
    first_call = llm.calls[0]
    assert len(first_call) == 3
    assert first_call[1]["role"] == "system"
    assert "<restored-fea-session>" in str(first_call[1]["content"])
    complete_events = [payload for event_type, payload in events if event_type == "fea_complete"]
    assert complete_events[-1]["summary"] == "Follow-up summary."


@pytest.mark.asyncio
async def test_fea_analyst_batches_tool_call_history_per_round() -> None:
    llm = RecordingScriptedLLM([
        {
            "tool_calls": [
                _tool_call(
                    "todo_write",
                    {
                        "todos": [
                            {"id": "model", "text": "Build the model", "status": "in_progress"},
                        ],
                    },
                ),
                _tool_call(
                    "fea_record_assumptions",
                    {"assumptions": ["2D frame idealised in the XY plane."]},
                ),
            ],
        },
        "Final summary.",
    ])
    analyst = FEAAnalystLoop(llm=llm, settings=Settings(project_root=PROJECT_ROOT))

    agen = analyst.run_stream("Analyse a simple frame")

    async for event_type, _payload in agen:
        if event_type == "fea_complete":
            break

    await agen.aclose()

    assert len(llm.calls) >= 2
    second_round_messages = llm.calls[1]
    assistant_tool_messages = [
        msg for msg in second_round_messages
        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls")
    ]
    tool_messages = [
        msg for msg in second_round_messages
        if isinstance(msg, dict) and msg.get("role") == "tool"
    ]

    assert len(assistant_tool_messages) == 1
    assert len(assistant_tool_messages[0]["tool_calls"]) == 2
    assert "content" not in assistant_tool_messages[0]
    assert len(tool_messages) == 2
    assert all(msg.get("content") is not None for msg in tool_messages)


@pytest.mark.asyncio
async def test_fea_analyst_redirects_restored_semantic_follow_up_to_patch_tools() -> None:
    llm = RecordingScriptedLLM([
        {
            "tool_calls": [
                _tool_call(
                    "fea_add_nodes",
                    {"nodes": [{"id": "N99", "x": 0, "y": 0}]},
                ),
            ],
        },
        {
            "tool_calls": [
                _tool_call(
                    "fea_patch_frame_geometry",
                    {"operation": "extend", "additional_storeys": 1},
                ),
            ],
        },
    ])
    analyst = FEAAnalystLoop(llm=llm, settings=Settings(project_root=PROJECT_ROOT))
    history = [
        {"role": "assistant", "content": "Previous FEA summary.", "response_payload": _fea_history_payload()},
    ]

    events: list[tuple[str, dict[str, object]]] = []
    agen = analyst.run_stream("add one more floor", history=history)

    async for event_type, payload in agen:
        events.append((event_type, payload))
        if event_type == "fea_tool_call" and payload["tool"] == "fea_patch_frame_geometry":
            break

    await agen.aclose()

    tool_calls = [payload["tool"] for event_type, payload in events if event_type == "fea_tool_call"]
    assert tool_calls == ["fea_patch_frame_geometry"]
    assert any(
        event_type == "fea_thinking"
        and "Editing the restored frame semantically" in str(payload.get("content", ""))
        for event_type, payload in events
    )
    assert len(llm.calls) >= 2
    redirect_prompt = llm.calls[1][-1]["content"]
    assert isinstance(redirect_prompt, str)
    assert "semantic frame model is already restored" in redirect_prompt


@pytest.mark.asyncio
async def test_restored_semantic_sessions_hide_raw_fe_build_tools_from_llm() -> None:
    llm = RecordingScriptedLLM([
        {
            "tool_calls": [
                _tool_call("fea_query_model", {"scope": "geometry"}),
            ],
        },
        "Follow-up geometry summary.",
    ])
    analyst = FEAAnalystLoop(llm=llm, settings=Settings(project_root=PROJECT_ROOT))
    history = [
        {"role": "assistant", "content": "Previous FEA summary.", "response_payload": _fea_history_payload()},
    ]

    agen = analyst.run_stream("what is the current geometry", history=history)

    async for event_type, _payload in agen:
        if event_type == "fea_complete":
            break

    await agen.aclose()

    assert llm.tool_defs
    first_tools = llm.tool_defs[0] or []
    tool_names = {tool["function"]["name"] for tool in first_tools}
    assert "fea_query_model" in tool_names
    assert "fea_patch_frame_geometry" in tool_names
    assert "fea_check_model" in tool_names
    assert "fea_solve" in tool_names
    assert "fea_add_nodes" not in tool_names
    assert "fea_add_elements" not in tool_names
    assert "fea_assign_sections" not in tool_names
    assert "fea_set_analysis_type" not in tool_names
