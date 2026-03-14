from __future__ import annotations

from pathlib import Path

from backend.orchestrator.fea_prompts import FEA_TOOLS
from backend.orchestrator.fea_tools import (
    FEAModelState,
    execute_fea_tool,
    get_result_query_coverage_key,
    normalize_result_query,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_fea_add_elements_rejects_noncanonical_type() -> None:
    state = FEAModelState()

    _commands, result = execute_fea_tool(
        "fea_add_elements",
        {
            "elements": [
                {"id": "E1", "type": "beam_member", "node_ids": ["N1", "N2"]},
            ],
        },
        state,
        PROJECT_ROOT,
    )

    assert result.startswith("TOOL ERROR:")
    assert "Unsupported element type" in result


def test_fea_assign_material_rejects_unknown_grade() -> None:
    state = FEAModelState()
    state.elements["E1"] = {"type": "beam", "nodeIds": ["N1", "N2"]}

    _commands, result = execute_fea_tool(
        "fea_assign_material",
        {"element_ids": ["E1"], "grade": "S500"},
        state,
        PROJECT_ROOT,
    )

    assert result.startswith("TOOL ERROR:")
    assert "Unsupported steel grade" in result


def test_fea_record_assumptions_deduplicates_entries() -> None:
    state = FEAModelState()

    _commands, result = execute_fea_tool(
        "fea_record_assumptions",
        {"assumptions": ["2D idealisation in XY plane", "2D idealisation in XY plane"]},
        state,
        PROJECT_ROOT,
    )

    assert state.assumptions == ["2D idealisation in XY plane"]
    assert "2D idealisation in XY plane" in result


def test_fea_clear_resets_plan_and_assumptions() -> None:
    state = FEAModelState()
    state.nodes["N1"] = {"x": 0, "y": 0, "z": 0}
    state.plan = [{"id": "model", "text": "Build model", "status": "done"}]
    state.assumptions = ["Pinned base assumed"]

    commands, result = execute_fea_tool(
        "fea_clear",
        {},
        state,
        PROJECT_ROOT,
    )

    assert commands == [{"action": "clear"}]
    assert result == "Model cleared. Ready to rebuild."
    assert state.nodes == {}
    assert state.plan == []
    assert state.assumptions == []


def test_todo_write_rejects_blank_step_text() -> None:
    state = FEAModelState()

    _commands, result = execute_fea_tool(
        "todo_write",
        {"todos": [{"id": "model", "text": "   ", "status": "pending"}]},
        state,
        PROJECT_ROOT,
    )

    assert result.startswith("TOOL ERROR:")
    assert "non-empty 'text'" in result
    assert state.plan == []


def test_distributed_load_rejects_empty_components() -> None:
    state = FEAModelState()

    _commands, result = execute_fea_tool(
        "fea_add_loads",
        {"load_case_id": "LC1", "loads": [{"type": "distributed", "element_id": "B1"}]},
        state,
        PROJECT_ROOT,
    )

    assert result.startswith("TOOL ERROR:")
    assert "Use type='self_weight'" in result


def test_fea_add_loads_accepts_self_weight() -> None:
    state = FEAModelState()

    commands, result = execute_fea_tool(
        "fea_add_loads",
        {
            "load_case_id": "LC1",
            "loads": [{"type": "self_weight", "factor": 1.0, "direction": {"x": 0, "y": -1, "z": 0}}],
        },
        state,
        PROJECT_ROOT,
    )

    assert commands == [
        {
            "action": "add_loads",
            "load_case_id": "LC1",
            "loads": [{"type": "self_weight", "factor": 1.0, "direction": {"x": 0, "y": -1, "z": 0}}],
        }
    ]
    assert "Self-weight load" in result
    assert state.load_cases["LC1"]["loads"][0]["type"] == "self_weight"


def test_fea_define_rectilinear_frame_compiles_parametric_model() -> None:
    state = FEAModelState()

    commands, result = execute_fea_tool(
        "fea_define_rectilinear_frame",
        {
            "dimension": "2d",
            "spans_x": [5000, 5000],
            "storey_heights": [3000, 3000],
            "column_profile": "HEB200",
            "beam_x_profile": "IPE300",
            "material_grade": "S355",
            "base_support": "fixed",
            "load_cases": [
                {
                    "id": "LC1",
                    "loads": [
                        {"type": "self_weight", "factor": 1.0, "direction": {"x": 0, "y": -1, "z": 0}},
                    ],
                },
            ],
        },
        state,
        PROJECT_ROOT,
    )

    assert commands[0] == {"action": "clear"}
    assert commands[1] == {"action": "set_analysis_type", "type": "beam2d"}
    assert len(state.nodes) == 9
    assert len(state.elements) == 10
    assert state.analysis_type == "beam2d"
    assert state.semantic_model is not None
    assert state.semantic_model["geometry"]["spans_x"] == [5000.0, 5000.0]
    assert state.load_cases["LC1"]["loads"][0]["type"] == "self_weight"
    assert "Compiled 2D rectilinear frame" in result


def test_fea_patch_frame_geometry_extends_and_promotes_to_3d() -> None:
    state = FEAModelState()
    execute_fea_tool(
        "fea_define_rectilinear_frame",
        {
            "dimension": "2d",
            "spans_x": [5000, 5000],
            "storey_heights": [3000, 3000],
            "column_profile": "HEB200",
            "beam_x_profile": "IPE300",
            "material_grade": "S355",
            "base_support": "fixed",
            "load_cases": [],
        },
        state,
        PROJECT_ROOT,
    )

    commands, result = execute_fea_tool(
        "fea_patch_frame_geometry",
        {
            "operation": "extend",
            "additional_storeys": 3,
            "additional_bays_z": 2,
        },
        state,
        PROJECT_ROOT,
    )

    assert commands[1] == {"action": "set_analysis_type", "type": "frame3d"}
    assert state.semantic_model is not None
    assert state.semantic_model["dimension"] == "3d"
    assert state.semantic_model["geometry"]["storey_heights"] == [3000.0, 3000.0, 3000.0, 3000.0, 3000.0]
    assert state.semantic_model["geometry"]["spans_z"] == [5000.0, 5000.0]
    assert len(state.nodes) == 54
    assert state.analysis_type == "frame3d"
    assert "Promoted the frame to 3D" in result


def test_fea_patch_loads_replaces_semantic_load_cases() -> None:
    state = FEAModelState()
    execute_fea_tool(
        "fea_define_rectilinear_frame",
        {
            "dimension": "2d",
            "spans_x": [5000],
            "storey_heights": [3000],
            "column_profile": "HEB200",
            "beam_x_profile": "IPE300",
            "material_grade": "S355",
            "base_support": "fixed",
            "load_cases": [{"id": "LC1", "loads": []}],
        },
        state,
        PROJECT_ROOT,
    )

    _commands, _result = execute_fea_tool(
        "fea_patch_loads",
        {
            "mode": "replace_all",
            "load_cases": [
                {
                    "id": "LC_SELF",
                    "loads": [{"type": "self_weight", "factor": 1.0, "direction": {"x": 0, "y": -1, "z": 0}}],
                },
            ],
        },
        state,
        PROJECT_ROOT,
    )

    assert list(state.load_cases) == ["LC_SELF"]
    assert state.semantic_model is not None
    assert state.semantic_model["load_cases"][0]["id"] == "LC_SELF"
    assert state.load_cases["LC_SELF"]["loads"][0]["type"] == "self_weight"


def test_mutating_tool_invalidates_stale_results() -> None:
    state = FEAModelState()
    state.solved = True
    state.results = {"solverInfo": {"dofCount": 6}}

    commands, result = execute_fea_tool(
        "fea_set_analysis_type",
        {"type": "beam2d"},
        state,
        PROJECT_ROOT,
    )

    assert commands == [{"action": "set_analysis_type", "type": "beam2d"}]
    assert "Analysis type set to beam2d" in result
    assert state.solved is False
    assert state.results is None


def test_fea_check_model_flags_invalid_load_references() -> None:
    state = FEAModelState()
    execute_fea_tool(
        "fea_define_rectilinear_frame",
        {
            "dimension": "2d",
            "spans_x": [5000],
            "storey_heights": [3000],
            "column_profile": "HEB200",
            "beam_x_profile": "IPE300",
            "material_grade": "S355",
            "base_support": "fixed",
            "load_cases": [
                {
                    "id": "LC1",
                    "loads": [{"type": "distributed", "element_id": "MISSING", "qy": -1.0}],
                },
            ],
        },
        state,
        PROJECT_ROOT,
    )

    _commands, result = execute_fea_tool("fea_check_model", {}, state, PROJECT_ROOT)

    assert "undefined element MISSING" in result


def test_fea_load_schema_exposes_self_weight() -> None:
    tool = next(item["function"] for item in FEA_TOOLS if item["function"]["name"] == "fea_add_loads")
    enum = tool["parameters"]["properties"]["loads"]["items"]["properties"]["type"]["enum"]
    assert "self_weight" in enum


def test_result_query_contract_is_canonical() -> None:
    tool = next(item["function"] for item in FEA_TOOLS if item["function"]["name"] == "fea_get_results")
    enum = tool["parameters"]["properties"]["query"]["enum"]
    assert "displacements" in enum
    assert "all_displacements" not in enum
    assert normalize_result_query("all_displacements") == "displacements"
    assert normalize_result_query("support_reactions") == "reactions"
    assert get_result_query_coverage_key("all_displacements") == "displacements"
    assert get_result_query_coverage_key("max_displacement") is None


def test_fea_get_results_max_and_full_displacements_are_distinct() -> None:
    state = FEAModelState()
    state.results = {
        "maxValues": {
            "maxDisplacement": {"value": 12.5, "nodeId": "N3", "direction": "dy"},
        },
        "displacements": {
            "N1": {"dx": 0.0, "dy": 0.0, "dz": 0.0},
            "N3": {"dx": 0.0, "dy": -12.5, "dz": 0.0},
        },
    }

    _commands, max_result = execute_fea_tool(
        "fea_get_results",
        {"query": "max_displacement"},
        state,
        PROJECT_ROOT,
    )
    _commands, full_result = execute_fea_tool(
        "fea_get_results",
        {"query": "displacements"},
        state,
        PROJECT_ROOT,
    )

    assert "Maximum displacement: 12.500000 mm at node N3" in max_result
    assert "All nodal displacements (mm):" in full_result
    assert "Node N3: dx=0.000000, dy=-12.500000, dz=0.000000" in full_result
