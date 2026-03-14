"""FEA tool executor — processes FEA analyst tool calls into frontend commands.

Each tool call generates JSON commands that the frontend FEA engine executes.
No computation happens here; this is a command translator.
"""

from __future__ import annotations

from copy import deepcopy
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Load profile database for validation ──────────────────────────

_profile_db: dict | None = None


_RESULT_QUERY_ALIASES = {
    "all_displacements": "displacements",
    "nodal_displacements": "displacements",
    "displacement": "max_displacement",
    "max_disp": "max_displacement",
    "forces": "element_forces",
    "member_forces": "element_forces",
    "internal_forces": "element_forces",
    "reaction": "reactions",
    "support_reactions": "reactions",
}


def normalize_result_query(query: Any) -> str | None:
    """Map result-query aliases onto the canonical public query names."""
    if not isinstance(query, str):
        return None
    normalized = query.strip().lower()
    if not normalized:
        return None
    return _RESULT_QUERY_ALIASES.get(normalized, normalized)


def get_result_query_coverage_key(query: Any) -> str | None:
    """Return the completeness bucket satisfied by a result query, if any."""
    canonical = normalize_result_query(query)
    if canonical in {"displacements", "reactions", "element_forces"}:
        return canonical
    return None


def _load_profile_db(project_root: Path) -> dict:
    global _profile_db
    if _profile_db is not None:
        return _profile_db
    path = project_root / "data" / "profiles" / "european_sections.json"
    if path.exists():
        _profile_db = json.loads(path.read_text(encoding="utf-8"))
    else:
        _profile_db = {}
    return _profile_db


def _lookup_profile(project_root: Path, profile_name: str) -> dict | None:
    db = _load_profile_db(project_root)
    upper = profile_name.upper()
    for series in db.values():
        if upper in series:
            return series[upper]
    return None


# ── Model state tracker (mirrors what frontend has) ───────────────

class FEAModelState:
    """Tracks what has been sent to the frontend so the analyst can inspect it."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.elements: dict[str, dict] = {}
        self.sections: dict[str, dict] = {}
        self.materials: dict[str, dict] = {}
        self.restraints: dict[str, dict] = {}
        self.load_cases: dict[str, dict] = {}
        self.analysis_type: str = "beam2d"
        self.solved: bool = False
        self.results: dict | None = None
        self.plan: list[dict[str, Any]] = []
        self.assumptions: list[str] = []
        self.semantic_model: dict[str, Any] | None = None

    def to_authoring_snapshot(self) -> dict[str, Any]:
        return {
            "analysis_type": self.analysis_type,
            "nodes": deepcopy(self.nodes),
            "elements": deepcopy(self.elements),
            "sections": deepcopy(self.sections),
            "materials": deepcopy(self.materials),
            "restraints": deepcopy(self.restraints),
            "load_cases": deepcopy(self.load_cases),
            "solved": bool(self.solved),
            "results": deepcopy(self.results),
            "plan": deepcopy(self.plan),
            "assumptions": list(self.assumptions),
            "semantic_model": deepcopy(self.semantic_model),
        }

    @classmethod
    def from_authoring_snapshot(cls, snapshot: dict[str, Any]) -> "FEAModelState":
        state = cls()
        if not isinstance(snapshot, dict):
            return state

        state.analysis_type = str(snapshot.get("analysis_type") or state.analysis_type)
        if isinstance(snapshot.get("nodes"), dict):
            state.nodes = deepcopy(snapshot["nodes"])
        if isinstance(snapshot.get("elements"), dict):
            state.elements = deepcopy(snapshot["elements"])
        if isinstance(snapshot.get("sections"), dict):
            state.sections = deepcopy(snapshot["sections"])
        if isinstance(snapshot.get("materials"), dict):
            state.materials = deepcopy(snapshot["materials"])
        if isinstance(snapshot.get("restraints"), dict):
            state.restraints = deepcopy(snapshot["restraints"])
        if isinstance(snapshot.get("load_cases"), dict):
            state.load_cases = deepcopy(snapshot["load_cases"])
        if isinstance(snapshot.get("results"), dict):
            state.results = deepcopy(snapshot["results"])
        state.solved = bool(snapshot.get("solved")) and isinstance(state.results, dict)
        if isinstance(snapshot.get("plan"), list):
            state.plan = deepcopy(snapshot["plan"])
        if isinstance(snapshot.get("assumptions"), list):
            state.assumptions = [str(item).strip() for item in snapshot["assumptions"] if str(item).strip()]
        if isinstance(snapshot.get("semantic_model"), dict):
            state.semantic_model = deepcopy(snapshot["semantic_model"])
        return state


_SEMANTIC_MODEL_KIND = "rectilinear_frame"
_SEMANTIC_DIMENSIONS = {"2d": "beam2d", "3d": "frame3d"}
_SEMANTIC_BASE_SUPPORTS = {"fixed", "pinned"}
_SEMANTIC_MEMBER_FAMILIES = {"columns", "beams_x", "beams_z"}


def _clear_authoring_state(state: FEAModelState) -> None:
    state.nodes.clear()
    state.elements.clear()
    state.sections.clear()
    state.materials.clear()
    state.restraints.clear()
    state.load_cases.clear()
    _invalidate_results(state)


def _normalize_positive_lengths(raw: Any, field_name: str) -> tuple[list[float] | None, str | None]:
    if not isinstance(raw, list) or not raw:
        return None, f"'{field_name}' must be a non-empty array of positive lengths in mm."
    values: list[float] = []
    for index, item in enumerate(raw, start=1):
        if not _is_numeric(item):
            return None, f"'{field_name}[{index}]' must be numeric."
        value = float(item)
        if value <= 0:
            return None, f"'{field_name}[{index}]' must be positive."
        values.append(value)
    return values, None


def _normalize_load_cases(raw: Any) -> tuple[list[dict[str, Any]], str | None]:
    if raw is None:
        return [], None
    if not isinstance(raw, list):
        return [], "'load_cases' must be an array."

    normalized: list[dict[str, Any]] = []
    for index, case in enumerate(raw, start=1):
        if not isinstance(case, dict):
            return [], f"Load case #{index} must be an object."
        load_case_id = str(case.get("id", "")).strip()
        if not load_case_id:
            return [], f"Load case #{index} must include a non-empty 'id'."
        name = str(case.get("name", load_case_id) or load_case_id).strip() or load_case_id
        loads = case.get("loads", [])
        if not isinstance(loads, list):
            return [], f"Load case '{load_case_id}' must include a 'loads' array."
        normalized.append({
            "id": load_case_id,
            "name": name,
            "loads": deepcopy(loads),
        })
    return normalized, None


def _normalize_frame_semantic_model(args: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    dimension = str(args.get("dimension", "")).strip().lower()
    if dimension not in _SEMANTIC_DIMENSIONS:
        return None, "Rectilinear frames require dimension '2d' or '3d'."

    spans_x, error = _normalize_positive_lengths(args.get("spans_x"), "spans_x")
    if error:
        return None, error
    assert spans_x is not None

    storey_heights, error = _normalize_positive_lengths(args.get("storey_heights"), "storey_heights")
    if error:
        return None, error
    assert storey_heights is not None

    spans_z: list[float] = []
    if dimension == "3d":
        spans_z, error = _normalize_positive_lengths(args.get("spans_z"), "spans_z")
        if error:
            return None, error
        assert spans_z is not None

    column_profile = str(args.get("column_profile", "")).strip().upper()
    beam_x_profile = str(args.get("beam_x_profile", "")).strip().upper()
    beam_z_profile = str(args.get("beam_z_profile", beam_x_profile) or beam_x_profile).strip().upper()
    if not column_profile or not beam_x_profile:
        return None, "Rectilinear frames require 'column_profile' and 'beam_x_profile'."
    if dimension == "3d" and not beam_z_profile:
        return None, "3D rectilinear frames require 'beam_z_profile' or a valid 'beam_x_profile'."

    material_grade = str(args.get("material_grade", "")).strip().upper()
    if material_grade not in {"S235", "S275", "S355", "S420", "S460"}:
        return None, "Rectilinear frames require a supported 'material_grade' (S235/S275/S355/S420/S460)."

    base_support = str(args.get("base_support", "fixed")).strip().lower()
    if base_support not in _SEMANTIC_BASE_SUPPORTS:
        return None, "Rectilinear frames require 'base_support' to be 'fixed' or 'pinned'."

    load_cases, error = _normalize_load_cases(args.get("load_cases"))
    if error:
        return None, error

    semantic_model = {
        "kind": _SEMANTIC_MODEL_KIND,
        "dimension": dimension,
        "geometry": {
            "spans_x": spans_x,
            "spans_z": spans_z,
            "storey_heights": storey_heights,
        },
        "member_families": {
            "columns": {"profile_name": column_profile},
            "beams_x": {"profile_name": beam_x_profile},
            "beams_z": {"profile_name": beam_z_profile if dimension == "3d" else beam_x_profile},
        },
        "material": {"grade": material_grade},
        "supports": {"base": base_support},
        "load_cases": load_cases,
    }
    return semantic_model, None


def _cumulative_positions(lengths: list[float]) -> list[float]:
    positions = [0.0]
    total = 0.0
    for value in lengths:
        total += float(value)
        positions.append(total)
    return positions


def _node_id(dimension: str, x_idx: int, y_idx: int, z_idx: int = 0) -> str:
    if dimension == "2d":
        return f"N_X{x_idx}_Y{y_idx}"
    return f"N_X{x_idx}_Y{y_idx}_Z{z_idx}"


def _column_id(dimension: str, x_idx: int, y_idx: int, z_idx: int = 0) -> str:
    if dimension == "2d":
        return f"COL_X{x_idx}_Y{y_idx}"
    return f"COL_X{x_idx}_Y{y_idx}_Z{z_idx}"


def _beam_x_id(dimension: str, x_idx: int, y_idx: int, z_idx: int = 0) -> str:
    if dimension == "2d":
        return f"BMX_X{x_idx}_Y{y_idx}"
    return f"BMX_X{x_idx}_Y{y_idx}_Z{z_idx}"


def _beam_z_id(x_idx: int, y_idx: int, z_idx: int) -> str:
    return f"BMZ_X{x_idx}_Y{y_idx}_Z{z_idx}"


def _compiled_restraint_type(dimension: str, base_support: str) -> str:
    if dimension == "2d":
        return "fixed_2d" if base_support == "fixed" else "pin_2d"
    return "fixed" if base_support == "fixed" else "pin"


def _semantic_model_summary(semantic_model: dict[str, Any]) -> str:
    geometry = semantic_model.get("geometry", {}) if isinstance(semantic_model, dict) else {}
    member_families = semantic_model.get("member_families", {}) if isinstance(semantic_model, dict) else {}
    spans_x = geometry.get("spans_x", []) if isinstance(geometry.get("spans_x"), list) else []
    spans_z = geometry.get("spans_z", []) if isinstance(geometry.get("spans_z"), list) else []
    storey_heights = geometry.get("storey_heights", []) if isinstance(geometry.get("storey_heights"), list) else []
    return (
        f"{semantic_model.get('dimension', '2d').upper()} rectilinear frame: "
        f"{len(spans_x)} bay(s) in X, {len(storey_heights)} storey(s), "
        f"{len(spans_z)} bay(s) in Z; columns={member_families.get('columns', {}).get('profile_name', '?')}, "
        f"beams_x={member_families.get('beams_x', {}).get('profile_name', '?')}, "
        f"beams_z={member_families.get('beams_z', {}).get('profile_name', '?')}, "
        f"material={semantic_model.get('material', {}).get('grade', '?')}, "
        f"base_support={semantic_model.get('supports', {}).get('base', '?')}."
    )


def _build_semantic_query(scope: str, state: FEAModelState) -> str:
    semantic_model = state.semantic_model
    if not isinstance(semantic_model, dict):
        return (
            f"No semantic frame model is stored. Current authoring model: "
            f"analysis_type={state.analysis_type}, nodes={len(state.nodes)}, "
            f"elements={len(state.elements)}, load_cases={len(state.load_cases)}, solved={state.solved}."
        )

    geometry = semantic_model.get("geometry", {})
    member_families = semantic_model.get("member_families", {})
    load_cases = semantic_model.get("load_cases", [])
    dimension = semantic_model.get("dimension", "2d")
    if scope == "geometry":
        sample_node = _node_id(dimension, 0, min(len(geometry.get("storey_heights", [])), 1), 0)
        sample_column = _column_id(dimension, 0, 0, 0)
        sample_beam_x = _beam_x_id(dimension, 0, min(len(geometry.get("storey_heights", [])), 1), 0)
        parts = [
            _semantic_model_summary(semantic_model),
            f"spans_x={geometry.get('spans_x', [])}",
            f"storey_heights={geometry.get('storey_heights', [])}",
        ]
        if dimension == "3d":
            parts.append(f"spans_z={geometry.get('spans_z', [])}")
            parts.append(f"Stable IDs: nodes like {sample_node}, columns like {sample_column}, beams in X like {sample_beam_x}, beams in Z like { _beam_z_id(0, min(len(geometry.get('storey_heights', [])), 1), 0)}.")
        else:
            parts.append(f"Stable IDs: nodes like {sample_node}, columns like {sample_column}, beams like {sample_beam_x}.")
        return " ".join(parts)
    if scope == "loads":
        load_summaries = []
        for case in load_cases:
            if not isinstance(case, dict):
                continue
            load_summaries.append(
                f"{case.get('id', 'LC?')} ({len(case.get('loads', []))} load(s))"
            )
        return (
            _semantic_model_summary(semantic_model)
            + " Load cases: "
            + (", ".join(load_summaries) if load_summaries else "none.")
        )
    if scope == "supports":
        return (
            _semantic_model_summary(semantic_model)
            + f" Supports: base_support={semantic_model.get('supports', {}).get('base', '?')}."
        )
    if scope == "members":
        return (
            _semantic_model_summary(semantic_model)
            + " Member families: "
            + f"columns={member_families.get('columns', {}).get('profile_name', '?')}, "
            + f"beams_x={member_families.get('beams_x', {}).get('profile_name', '?')}, "
            + f"beams_z={member_families.get('beams_z', {}).get('profile_name', '?')}."
        )
    return _semantic_model_summary(semantic_model)


def _rebuild_authoring_state_from_semantic_model(
    state: FEAModelState,
    semantic_model: dict[str, Any],
    root: Path,
) -> tuple[list[dict], str]:
    if semantic_model.get("kind") != _SEMANTIC_MODEL_KIND:
        return _tool_error("Only rectilinear_frame semantic models are supported right now.")

    dimension = str(semantic_model.get("dimension", "")).strip().lower()
    if dimension not in _SEMANTIC_DIMENSIONS:
        return _tool_error("Semantic frame model must have dimension '2d' or '3d'.")

    geometry = semantic_model.get("geometry", {})
    spans_x = list(geometry.get("spans_x", []))
    storey_heights = list(geometry.get("storey_heights", []))
    spans_z = list(geometry.get("spans_z", [])) if dimension == "3d" else []
    if not spans_x or not storey_heights:
        return _tool_error("Semantic frame geometry is incomplete.")
    if dimension == "3d" and not spans_z:
        return _tool_error("3D semantic frame geometry requires spans_z.")

    member_families = semantic_model.get("member_families", {})
    column_profile = str(member_families.get("columns", {}).get("profile_name", "")).strip().upper()
    beam_x_profile = str(member_families.get("beams_x", {}).get("profile_name", "")).strip().upper()
    beam_z_profile = str(member_families.get("beams_z", {}).get("profile_name", beam_x_profile) or beam_x_profile).strip().upper()
    material_grade = str(semantic_model.get("material", {}).get("grade", "")).strip().upper()
    base_support = str(semantic_model.get("supports", {}).get("base", "fixed")).strip().lower()

    if not column_profile or not beam_x_profile or not material_grade:
        return _tool_error("Semantic frame members/material are incomplete.")
    if base_support not in _SEMANTIC_BASE_SUPPORTS:
        return _tool_error("Semantic frame supports must use base_support 'fixed' or 'pinned'.")

    profile_names = {column_profile, beam_x_profile}
    if dimension == "3d":
        profile_names.add(beam_z_profile)
    profile_props: dict[str, dict[str, Any]] = {}
    for profile_name in profile_names:
        props = _lookup_profile(root, profile_name)
        if not props:
            return _tool_error(
                f"Profile '{profile_name}' was not found in the section database. "
                "Choose a standard section that exists."
            )
        profile_props[profile_name] = props

    x_positions = _cumulative_positions([float(value) for value in spans_x])
    y_positions = _cumulative_positions([float(value) for value in storey_heights])
    z_positions = [0.0] if dimension == "2d" else _cumulative_positions([float(value) for value in spans_z])
    analysis_type = _SEMANTIC_DIMENSIONS[dimension]

    commands: list[dict[str, Any]] = [{"action": "clear"}, {"action": "set_analysis_type", "type": analysis_type}]
    _clear_authoring_state(state)
    state.analysis_type = analysis_type
    state.semantic_model = deepcopy(semantic_model)

    nodes: list[dict[str, Any]] = []
    for y_idx, y in enumerate(y_positions):
        for z_idx, z in enumerate(z_positions):
            for x_idx, x in enumerate(x_positions):
                node = {
                    "id": _node_id(dimension, x_idx, y_idx, z_idx),
                    "x": x,
                    "y": y,
                }
                if dimension == "3d":
                    node["z"] = z
                nodes.append(node)
                state.nodes[node["id"]] = {"x": x, "y": y, "z": z if dimension == "3d" else 0.0}
    commands.append({"action": "add_nodes", "nodes": nodes})

    elements: list[dict[str, Any]] = []
    column_ids: list[str] = []
    beam_x_ids: list[str] = []
    beam_z_ids: list[str] = []

    for y_idx in range(len(storey_heights)):
        for z_idx in range(len(z_positions)):
            for x_idx in range(len(x_positions)):
                elem_id = _column_id(dimension, x_idx, y_idx, z_idx)
                element = {
                    "id": elem_id,
                    "type": "column",
                    "node_ids": [
                        _node_id(dimension, x_idx, y_idx, z_idx),
                        _node_id(dimension, x_idx, y_idx + 1, z_idx),
                    ],
                }
                elements.append(element)
                column_ids.append(elem_id)
                state.elements[elem_id] = {
                    "type": "column",
                    "nodeIds": list(element["node_ids"]),
                    "sectionId": f"sec_{column_profile}",
                    "materialId": f"mat_{material_grade}",
                }

    for y_idx in range(1, len(y_positions)):
        for z_idx in range(len(z_positions)):
            for x_idx in range(len(spans_x)):
                elem_id = _beam_x_id(dimension, x_idx, y_idx, z_idx)
                element = {
                    "id": elem_id,
                    "type": "beam",
                    "node_ids": [
                        _node_id(dimension, x_idx, y_idx, z_idx),
                        _node_id(dimension, x_idx + 1, y_idx, z_idx),
                    ],
                }
                elements.append(element)
                beam_x_ids.append(elem_id)
                state.elements[elem_id] = {
                    "type": "beam",
                    "nodeIds": list(element["node_ids"]),
                    "sectionId": f"sec_{beam_x_profile}",
                    "materialId": f"mat_{material_grade}",
                }

    if dimension == "3d":
        for y_idx in range(1, len(y_positions)):
            for z_idx in range(len(spans_z)):
                for x_idx in range(len(x_positions)):
                    elem_id = _beam_z_id(x_idx, y_idx, z_idx)
                    element = {
                        "id": elem_id,
                        "type": "beam",
                        "node_ids": [
                            _node_id(dimension, x_idx, y_idx, z_idx),
                            _node_id(dimension, x_idx, y_idx, z_idx + 1),
                        ],
                    }
                    elements.append(element)
                    beam_z_ids.append(elem_id)
                    state.elements[elem_id] = {
                        "type": "beam",
                        "nodeIds": list(element["node_ids"]),
                        "sectionId": f"sec_{beam_z_profile}",
                        "materialId": f"mat_{material_grade}",
                    }

    commands.append({"action": "add_elements", "elements": elements})

    if column_ids:
        commands.append({
            "action": "assign_section",
            "element_ids": column_ids,
            "profile_name": column_profile,
            "properties": profile_props[column_profile],
        })
    if beam_x_ids:
        commands.append({
            "action": "assign_section",
            "element_ids": beam_x_ids,
            "profile_name": beam_x_profile,
            "properties": profile_props[beam_x_profile],
        })
    if beam_z_ids:
        commands.append({
            "action": "assign_section",
            "element_ids": beam_z_ids,
            "profile_name": beam_z_profile,
            "properties": profile_props[beam_z_profile],
        })

    material_command = {
        "action": "assign_material",
        "element_ids": list(state.elements.keys()),
        "grade": material_grade,
    }
    commands.append(material_command)

    state.sections = {
        f"sec_{column_profile}": deepcopy(profile_props[column_profile]),
        f"sec_{beam_x_profile}": deepcopy(profile_props[beam_x_profile]),
    }
    if dimension == "3d":
        state.sections[f"sec_{beam_z_profile}"] = deepcopy(profile_props[beam_z_profile])
    state.materials = {f"mat_{material_grade}": {"name": material_grade}}

    restraint_type = _compiled_restraint_type(dimension, base_support)
    restraints: list[dict[str, Any]] = []
    for z_idx in range(len(z_positions)):
        for x_idx in range(len(x_positions)):
            node_id = _node_id(dimension, x_idx, 0, z_idx)
            restraints.append({"node_id": node_id, "type": restraint_type})
            state.restraints[node_id] = {"type": restraint_type}
    commands.append({"action": "set_restraints", "restraints": restraints})

    state.load_cases = {}
    for load_case in semantic_model.get("load_cases", []):
        if not isinstance(load_case, dict):
            continue
        lc_id = str(load_case.get("id", "")).strip()
        if not lc_id:
            continue
        lc_loads = deepcopy(load_case.get("loads", [])) if isinstance(load_case.get("loads"), list) else []
        state.load_cases[lc_id] = {"name": str(load_case.get("name", lc_id) or lc_id), "loads": lc_loads}
        if lc_loads:
            commands.append({"action": "add_loads", "load_case_id": lc_id, "loads": lc_loads})

    return commands, (
        f"Compiled {_semantic_model_summary(semantic_model)} "
        f"Generated {len(state.nodes)} node(s), {len(state.elements)} element(s), "
        f"{len(state.load_cases)} load case(s)."
    )


# ── Tool executors ────────────────────────────────────────────────

def execute_fea_tool(
    tool_name: str,
    args: dict[str, Any],
    model_state: FEAModelState,
    project_root: Path,
) -> tuple[list[dict], str]:
    """Execute an FEA tool call.

    Returns (commands, result_text):
      - commands: list of JSON command objects for the frontend
      - result_text: human-readable result for the LLM context
    """
    handler = _TOOL_HANDLERS.get(tool_name)
    if not handler:
        return _tool_error(f"Unknown FEA tool: {tool_name}")
    return handler(args, model_state, project_root)


def _tool_error(message: str) -> tuple[list[dict], str]:
    return [], f"TOOL ERROR: {message}"


def _invalidate_results(state: FEAModelState) -> None:
    state.solved = False
    state.results = None


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _has_nonzero_components(load: dict[str, Any], keys: tuple[str, ...]) -> tuple[bool, str | None]:
    has_nonzero = False
    for key in keys:
        value = load.get(key)
        if value is None:
            continue
        if not _is_numeric(value):
            return False, f"Load component '{key}' must be numeric."
        if abs(float(value)) > 1e-12:
            has_nonzero = True
    return has_nonzero, None


def _normalize_todo(todo: Any, index: int) -> tuple[dict[str, str] | None, str | None]:
    if not isinstance(todo, dict):
        return None, f"Todo #{index} must be an object with 'id', 'text', and 'status'."

    todo_id = str(todo.get("id", "")).strip()
    text = str(todo.get("text", "")).strip()
    status = str(todo.get("status", "")).strip()

    if not todo_id:
        return None, f"Todo #{index} must include a non-empty 'id'."
    if not text:
        return None, f"Todo #{index} must include a non-empty 'text'."
    if status not in {"pending", "in_progress", "done"}:
        return None, (
            f"Todo #{index} has invalid status '{status}'. "
            "Use one of: pending, in_progress, done."
        )

    return {"id": todo_id, "text": text, "status": status}, None


def _handle_add_nodes(args: dict, state: FEAModelState, _root: Path) -> tuple[list[dict], str]:
    nodes = args.get("nodes", [])
    if not isinstance(nodes, list) or not nodes:
        return _tool_error("fea_add_nodes requires a non-empty 'nodes' array.")
    # Normalize IDs to strings (LLM may return ints)
    for n in nodes:
        if not all(key in n for key in ("id", "x", "y")):
            return _tool_error("Each node must include 'id', 'x', and 'y'.")
        n["id"] = str(n["id"])
    commands = [{"action": "add_nodes", "nodes": nodes}]
    for n in nodes:
        state.nodes[n["id"]] = {"x": n.get("x", 0), "y": n.get("y", 0), "z": n.get("z", 0)}
    _invalidate_results(state)
    return commands, f"Added {len(nodes)} nodes: {', '.join(n['id'] for n in nodes)}"


def _handle_add_elements(args: dict, state: FEAModelState, _root: Path) -> tuple[list[dict], str]:
    allowed_types = {"beam", "column", "truss"}
    elements = args.get("elements", [])
    if not isinstance(elements, list) or not elements:
        return _tool_error("fea_add_elements requires a non-empty 'elements' array.")
    for e in elements:
        if not all(key in e for key in ("id", "type", "node_ids")):
            return _tool_error("Each element must include 'id', 'type', and 'node_ids'.")
        e["id"] = str(e["id"])
        e["type"] = str(e["type"]).strip()
        if e["type"] not in allowed_types:
            return _tool_error(
                f"Unsupported element type '{e['type']}'. "
                f"Use one of: {', '.join(sorted(allowed_types))}."
            )
        nids = e.get("node_ids")
        if not isinstance(nids, list) or len(nids) != 2:
            return _tool_error(f"Element {e['id']} must have exactly two node_ids.")
        e["node_ids"] = [str(nid) for nid in nids]
    commands = [{"action": "add_elements", "elements": elements}]
    for e in elements:
        state.elements[e["id"]] = {"type": e["type"], "nodeIds": e.get("node_ids", [])}
    _invalidate_results(state)
    logger.info("fea_add_elements", extra={"elements": [{"id": e["id"], "node_ids": e.get("node_ids")} for e in elements]})
    return commands, f"Added {len(elements)} elements: {', '.join(e['id'] for e in elements)}"


def _handle_assign_sections(args: dict, state: FEAModelState, root: Path) -> tuple[list[dict], str]:
    elem_ids = [str(eid) for eid in args.get("element_ids", [])]
    profile_name = args.get("profile_name")
    if not elem_ids or not isinstance(profile_name, str) or not profile_name.strip():
        return _tool_error("fea_assign_sections requires 'element_ids' and 'profile_name'.")
    props = _lookup_profile(root, profile_name)
    if not props:
        return _tool_error(
            f"Profile '{profile_name}' was not found in the section database. "
            "Ask the user for a standard section or choose one that exists."
        )

    # Include properties so the frontend doesn't depend on fetching the JSON database
    cmd = {"action": "assign_section", "element_ids": elem_ids, "profile_name": profile_name}
    cmd["properties"] = props
    commands = [cmd]
    sec_info = f"Assigned {profile_name} (A={props.get('A',0)} mm², Iy={props.get('Iy',0)} mm⁴, h={props.get('h',0)} mm)"

    for eid in elem_ids:
        if eid in state.elements:
            state.elements[eid]["sectionId"] = f"sec_{profile_name}"
    state.sections[f"sec_{profile_name}"] = props
    _invalidate_results(state)

    return commands, sec_info


def _handle_assign_material(args: dict, state: FEAModelState, _root: Path) -> tuple[list[dict], str]:
    allowed_grades = {"S235", "S275", "S355", "S420", "S460"}
    elem_ids = [str(eid) for eid in args.get("element_ids", [])]
    grade = str(args.get("grade", "")).strip().upper()
    if not elem_ids or not isinstance(grade, str) or not grade.strip():
        return _tool_error("fea_assign_material requires 'element_ids' and 'grade'.")
    if grade not in allowed_grades:
        return _tool_error(
            f"Unsupported steel grade '{grade}'. "
            f"Use one of: {', '.join(sorted(allowed_grades))}."
        )
    commands = [{"action": "assign_material", "element_ids": elem_ids, "grade": grade}]

    for eid in elem_ids:
        if eid in state.elements:
            state.elements[eid]["materialId"] = f"mat_{grade}"
    state.materials[f"mat_{grade}"] = {"name": grade}
    _invalidate_results(state)

    return commands, f"Assigned material {grade} to elements {', '.join(str(e) for e in elem_ids)}"


def _handle_set_restraints(args: dict, state: FEAModelState, _root: Path) -> tuple[list[dict], str]:
    allowed = {"pin", "fixed", "roller_x", "roller_y", "pin_2d", "roller_2d", "fixed_2d"}
    restraints = args.get("restraints", [])
    if not isinstance(restraints, list) or not restraints:
        return _tool_error("fea_set_restraints requires a non-empty 'restraints' array.")
    for r in restraints:
        if not all(key in r for key in ("node_id", "type")):
            return _tool_error("Each restraint must include 'node_id' and 'type'.")
        r["node_id"] = str(r["node_id"])
        r["type"] = str(r["type"]).strip()
        if r["type"] not in allowed:
            return _tool_error(
                f"Unsupported restraint type '{r['type']}'. "
                f"Use one of: {', '.join(sorted(allowed))}."
            )
    commands = [{"action": "set_restraints", "restraints": restraints}]
    for r in restraints:
        nid = r["node_id"]
        state.restraints[nid] = {"type": r["type"]}
    _invalidate_results(state)
    descs = [f"{r['node_id']}: {r['type']}" for r in restraints]
    return commands, f"Set restraints: {'; '.join(descs)}"


def _handle_add_loads(args: dict, state: FEAModelState, _root: Path) -> tuple[list[dict], str]:
    lc_id = str(args.get("load_case_id", "LC1"))
    loads = args.get("loads", [])
    if not isinstance(loads, list) or not loads:
        return _tool_error("fea_add_loads requires a non-empty 'loads' array.")
    for load in loads:
        ltype = str(load.get("type", "")).strip()
        if ltype not in {"nodal", "distributed", "self_weight"}:
            return _tool_error("Load type must be 'nodal', 'distributed', or 'self_weight'.")
        load["type"] = ltype
        if ltype == "nodal":
            if "node_id" not in load:
                return _tool_error("Nodal loads must include 'node_id'.")
            load["node_id"] = str(load["node_id"])
            has_force, error = _has_nonzero_components(load, ("fx", "fy", "fz", "mx", "my", "mz"))
            if error:
                return _tool_error(error)
            if not has_force:
                return _tool_error(
                    "Nodal loads must include at least one non-zero force or moment component."
                )
        elif ltype == "distributed":
            if "element_id" not in load:
                return _tool_error("Distributed loads must include 'element_id'.")
            load["element_id"] = str(load["element_id"])
            has_load, error = _has_nonzero_components(load, ("qx", "qy", "qz"))
            if error:
                return _tool_error(error)
            if not has_load:
                return _tool_error(
                    "Distributed loads must include at least one non-zero 'qx', 'qy', or 'qz'. "
                    "Use type='self_weight' for gravity/self-weight loads."
                )
        elif "direction" in load and not isinstance(load["direction"], dict):
            return _tool_error("Self-weight 'direction' must be an object with x, y, z components.")
    commands = [{"action": "add_loads", "load_case_id": lc_id, "loads": loads}]

    if lc_id not in state.load_cases:
        state.load_cases[lc_id] = {"loads": []}
    state.load_cases[lc_id]["loads"].extend(loads)
    _invalidate_results(state)

    descs = []
    for load in loads:
        if load["type"] == "nodal":
            parts = []
            for key in ("fx", "fy", "fz", "mx", "my", "mz"):
                val = load.get(key, 0)
                if val: parts.append(f"{key}={val}")
            descs.append(f"Nodal at {load.get('node_id', '?')}: {', '.join(parts)}")
        elif load["type"] == "distributed":
            parts = []
            for key in ("qx", "qy", "qz"):
                val = load.get(key, 0)
                if val: parts.append(f"{key}={val} N/mm")
            descs.append(f"Distributed on {load.get('element_id', '?')}: {', '.join(parts)}")
        elif load["type"] == "self_weight":
            factor = load.get("factor", 1.0)
            direction = load.get("direction")
            if isinstance(direction, dict):
                descs.append(
                    "Self-weight load "
                    f"(factor={factor}, direction=({direction.get('x', 0)}, {direction.get('y', 0)}, {direction.get('z', 0)}))"
                )
            else:
                descs.append(f"Self-weight load (factor={factor})")

    return commands, f"Load case {lc_id}: {'; '.join(descs)}"


def _handle_set_analysis_type(args: dict, state: FEAModelState, _root: Path) -> tuple[list[dict], str]:
    atype = args.get("type")
    if atype not in {"beam2d", "frame3d", "truss2d", "truss3d"}:
        return _tool_error("fea_set_analysis_type requires one of: beam2d, frame3d, truss2d, truss3d.")
    state.analysis_type = atype
    commands = [{"action": "set_analysis_type", "type": atype}]
    _invalidate_results(state)
    return commands, f"Analysis type set to {atype}"


def _handle_solve(args: dict, state: FEAModelState, _root: Path) -> tuple[list[dict], str]:
    lc_id = args.get("load_case_id", "LC1")
    state.solved = False
    # This is a special tool — the caller (FEAAnalystLoop) handles the solve request specially
    return [{"action": "solve", "load_case_id": lc_id}], f"__SOLVE_REQUEST__|{lc_id}"


def _handle_get_results(args: dict, state: FEAModelState, _root: Path) -> tuple[list[dict], str]:
    query = normalize_result_query(args.get("query", "summary")) or "summary"
    results = state.results

    if not results:
        return [], "No results available. Please run fea_solve first."

    if query == "summary":
        mv = results.get("maxValues", {})
        si = results.get("solverInfo", {})
        max_d = mv.get("maxDisplacement", {})
        max_m = mv.get("maxMoment", {})
        max_v = mv.get("maxShear", {})

        text = (
            f"Solver info: {si.get('dofCount', 0)} DOF, {si.get('elementCount', 0)} elements, "
            f"solved in {si.get('solveTimeMs', 0)}ms.\n"
            f"Max displacement: {max_d.get('value', 0):.4f} mm at node {max_d.get('nodeId', '?')} "
            f"(direction: {max_d.get('direction', '?')})\n"
            f"Max bending moment: {max_m.get('value', 0):.0f} N·mm = {max_m.get('value', 0)/1e6:.3f} kN·m "
            f"at element {max_m.get('elementId', '?')}\n"
            f"Max shear force: {max_v.get('value', 0):.0f} N = {max_v.get('value', 0)/1e3:.3f} kN "
            f"at element {max_v.get('elementId', '?')}"
        )
        return [], text

    if query == "reactions":
        reactions = results.get("reactions", {})
        lines = ["Reactions:"]
        for nid, r in reactions.items():
            parts = []
            for key in ("fx", "fy", "fz", "mx", "my", "mz"):
                val = r.get(key, 0)
                if abs(val) > 0.01:
                    if key.startswith("f"):
                        parts.append(f"{key}={val/1e3:.3f} kN")
                    else:
                        parts.append(f"{key}={val/1e6:.3f} kN·m")
            lines.append(f"  Node {nid}: {', '.join(parts)}")
        return [], "\n".join(lines)

    if query == "max_displacement":
        max_disp = (results.get("maxValues", {}) or {}).get("maxDisplacement", {}) or {}
        value = max_disp.get("value")
        node_id = max_disp.get("nodeId", "?")
        direction = max_disp.get("direction", "?")

        if not _is_numeric(value):
            disps = results.get("displacements", {})
            best_node = "?"
            best_mag = 0.0
            for nid, d in disps.items():
                mag = (
                    float(d.get("dx", 0) or 0) ** 2
                    + float(d.get("dy", 0) or 0) ** 2
                    + float(d.get("dz", 0) or 0) ** 2
                ) ** 0.5
                if mag >= best_mag:
                    best_mag = mag
                    best_node = nid
            value = best_mag
            node_id = best_node

        return [], (
            f"Maximum displacement: {float(value or 0):.6f} mm "
            f"at node {node_id} (direction: {direction})"
        )

    if query == "element_forces":
        elem_id = args.get("element_id")
        ef = results.get("elementForces", {})
        if elem_id and elem_id in ef:
            f_data = ef[elem_id]
            text = f"Element {elem_id} forces:\n"
            for key in ("N", "V", "M", "Vy", "Vz", "Mx", "My", "Mz"):
                vals = f_data.get(key, [])
                if vals and any(abs(v) > 0.01 for v in vals):
                    formatted = [f"{v:.1f}" for v in vals]
                    text += f"  {key}: [{', '.join(formatted)}]\n"
            return [], text
        else:
            text = "Element forces summary:\n"
            for eid, f_data in ef.items():
                m_vals = f_data.get("M", f_data.get("Mz", []))
                v_vals = f_data.get("V", f_data.get("Vy", []))
                n_vals = f_data.get("N", [])
                max_m = max((abs(v) for v in m_vals), default=0)
                max_v = max((abs(v) for v in v_vals), default=0)
                max_n = max((abs(v) for v in n_vals), default=0)
                text += f"  Element {eid}: |M|_max={max_m/1e6:.3f} kN·m, |V|_max={max_v/1e3:.3f} kN, |N|_max={max_n/1e3:.3f} kN\n"
            return [], text

    if query == "displacements":
        disps = results.get("displacements", {})
        text = "All nodal displacements (mm):\n"
        for nid, d in disps.items():
            text += f"  Node {nid}: dx={d.get('dx', 0):.6f}, dy={d.get('dy', 0):.6f}, dz={d.get('dz', 0):.6f}"
            rz = d.get("rz", 0)
            if abs(rz) > 1e-8:
                text += f", rz={rz:.6f} rad"
            text += "\n"
        return [], text

    return [], f"Unknown result query: {query}"


def _handle_set_view(args: dict, state: FEAModelState, _root: Path) -> tuple[list[dict], str]:
    action = args.get("action", "fit_view")
    cmd = {"action": action}
    if "scale_factor" in args:
        cmd["scale_factor"] = args["scale_factor"]
    return [cmd], f"View command: {action}"


def _handle_ask_user(args: dict, state: FEAModelState, _root: Path) -> tuple[list[dict], str]:
    """Ask the user a clarifying question. Returns sentinel for the agentic loop."""
    payload = json.dumps({
        "question": args.get("question", ""),
        "options": args.get("options", []),
        "context": args.get("context", ""),
    })
    return [], f"__ASK_USER__|{payload}"


def _handle_todo_write(args: dict, state: FEAModelState, _root: Path) -> tuple[list[dict], str]:
    todos = args.get("todos", [])
    if not isinstance(todos, list):
        return _tool_error("todo_write requires a 'todos' array.")
    normalized: list[dict[str, str]] = []
    for index, todo in enumerate(todos, start=1):
        item, error = _normalize_todo(todo, index)
        if error:
            return _tool_error(error)
        assert item is not None
        normalized.append(item)
    state.plan = normalized
    return [], json.dumps({
        "status": "ok",
        "plan": normalized,
    })


def _handle_record_assumptions(args: dict, state: FEAModelState, _root: Path) -> tuple[list[dict], str]:
    assumptions = args.get("assumptions", [])
    if not isinstance(assumptions, list):
        return _tool_error("fea_record_assumptions requires an 'assumptions' array.")
    for item in assumptions:
        text = str(item).strip()
        if text and text not in state.assumptions:
            state.assumptions.append(text)
    return [], json.dumps({
        "status": "ok",
        "assumptions": state.assumptions,
    })


def _handle_query_model(args: dict, state: FEAModelState, _root: Path) -> tuple[list[dict], str]:
    scope = str(args.get("scope", "summary") or "summary").strip().lower()
    if scope not in {"summary", "geometry", "loads", "supports", "members"}:
        return _tool_error("fea_query_model scope must be one of: summary, geometry, loads, supports, members.")
    return [], _build_semantic_query(scope, state)


def _handle_define_rectilinear_frame(args: dict, state: FEAModelState, root: Path) -> tuple[list[dict], str]:
    semantic_model, error = _normalize_frame_semantic_model(args)
    if error:
        return _tool_error(error)
    assert semantic_model is not None
    return _rebuild_authoring_state_from_semantic_model(state, semantic_model, root)


def _resolve_extension_lengths(
    existing: list[float],
    requested_count: int,
    provided: Any,
    field_name: str,
) -> tuple[list[float], list[str], str | None]:
    if requested_count <= 0:
        return [], [], None
    notes: list[str] = []
    if provided is not None:
        lengths, error = _normalize_positive_lengths(provided, field_name)
        if error:
            return [], notes, error
        assert lengths is not None
        if len(lengths) != requested_count:
            return [], notes, f"'{field_name}' must contain exactly {requested_count} value(s)."
        return lengths, notes, None
    if existing:
        reused = float(existing[-1])
        notes.append(f"Reused {reused:.0f} mm for {requested_count} new {field_name}.")
        return [reused] * requested_count, notes, None
    return [], notes, f"'{field_name}' must be provided when there is no existing pattern to extend."


def _handle_patch_frame_geometry(args: dict, state: FEAModelState, root: Path) -> tuple[list[dict], str]:
    semantic_model = deepcopy(state.semantic_model)
    if not isinstance(semantic_model, dict) or semantic_model.get("kind") != _SEMANTIC_MODEL_KIND:
        return _tool_error("fea_patch_frame_geometry requires an existing rectilinear semantic frame model.")

    operation = str(args.get("operation", "extend") or "extend").strip().lower()
    if operation not in {"extend", "replace"}:
        return _tool_error("fea_patch_frame_geometry operation must be 'extend' or 'replace'.")

    geometry = semantic_model.setdefault("geometry", {})
    notes: list[str] = []

    if operation == "replace":
        if "spans_x" in args:
            spans_x, error = _normalize_positive_lengths(args.get("spans_x"), "spans_x")
            if error:
                return _tool_error(error)
            geometry["spans_x"] = spans_x
        if "storey_heights" in args:
            storey_heights, error = _normalize_positive_lengths(args.get("storey_heights"), "storey_heights")
            if error:
                return _tool_error(error)
            geometry["storey_heights"] = storey_heights
        if "spans_z" in args:
            spans_z, error = _normalize_positive_lengths(args.get("spans_z"), "spans_z")
            if error:
                return _tool_error(error)
            geometry["spans_z"] = spans_z
            semantic_model["dimension"] = "3d"
    else:
        add_bays_x = int(args.get("additional_bays_x", 0) or 0)
        add_storeys = int(args.get("additional_storeys", 0) or 0)
        add_bays_z = int(args.get("additional_bays_z", 0) or 0)
        if min(add_bays_x, add_storeys, add_bays_z) < 0:
            return _tool_error("Geometry extension counts must be non-negative integers.")

        spans_x = list(geometry.get("spans_x", []))
        storey_heights = list(geometry.get("storey_heights", []))
        spans_z = list(geometry.get("spans_z", []))

        new_spans_x, new_notes, error = _resolve_extension_lengths(spans_x, add_bays_x, args.get("new_spans_x"), "new_spans_x")
        if error:
            return _tool_error(error)
        notes.extend(new_notes)

        new_storeys, new_notes, error = _resolve_extension_lengths(storey_heights, add_storeys, args.get("new_storey_heights"), "new_storey_heights")
        if error:
            return _tool_error(error)
        notes.extend(new_notes)

        target_dimension = str(args.get("dimension", semantic_model.get("dimension", "2d")) or semantic_model.get("dimension", "2d")).strip().lower()
        if target_dimension not in _SEMANTIC_DIMENSIONS:
            return _tool_error("dimension must be '2d' or '3d' when patching geometry.")
        if add_bays_z > 0:
            target_dimension = "3d"

        new_spans_z: list[float] = []
        if target_dimension == "3d":
            z_pattern = spans_z if spans_z else spans_x
            new_spans_z, new_notes, error = _resolve_extension_lengths(z_pattern, add_bays_z, args.get("new_spans_z"), "new_spans_z")
            if error:
                return _tool_error(error)
            if add_bays_z > 0 and not spans_z and not args.get("new_spans_z"):
                notes.append("Promoted the frame to 3D by reusing the X-bay spacing pattern in the Z direction.")
            notes.extend(new_notes)
            geometry["spans_z"] = spans_z + new_spans_z
            semantic_model["dimension"] = "3d"

        geometry["spans_x"] = spans_x + new_spans_x
        geometry["storey_heights"] = storey_heights + new_storeys

    member_families = semantic_model.setdefault("member_families", {})
    if semantic_model.get("dimension") == "3d" and not str(member_families.get("beams_z", {}).get("profile_name", "")).strip():
        beam_x_profile = str(member_families.get("beams_x", {}).get("profile_name", "")).strip().upper()
        member_families["beams_z"] = {"profile_name": beam_x_profile}
        if beam_x_profile:
            notes.append(f"Reused beam_x profile {beam_x_profile} for beam_z members.")

    commands, result = _rebuild_authoring_state_from_semantic_model(state, semantic_model, root)
    if result.startswith("TOOL ERROR:"):
        return commands, result
    if notes:
        result = f"{result} {' '.join(notes)}"
    return commands, result


def _handle_patch_supports(args: dict, state: FEAModelState, root: Path) -> tuple[list[dict], str]:
    semantic_model = deepcopy(state.semantic_model)
    if not isinstance(semantic_model, dict) or semantic_model.get("kind") != _SEMANTIC_MODEL_KIND:
        return _tool_error("fea_patch_supports requires an existing rectilinear semantic frame model.")
    base_support = str(args.get("base_support", "")).strip().lower()
    if base_support not in _SEMANTIC_BASE_SUPPORTS:
        return _tool_error("fea_patch_supports requires base_support 'fixed' or 'pinned'.")
    semantic_model.setdefault("supports", {})["base"] = base_support
    return _rebuild_authoring_state_from_semantic_model(state, semantic_model, root)


def _handle_patch_members(args: dict, state: FEAModelState, root: Path) -> tuple[list[dict], str]:
    semantic_model = deepcopy(state.semantic_model)
    if not isinstance(semantic_model, dict) or semantic_model.get("kind") != _SEMANTIC_MODEL_KIND:
        return _tool_error("fea_patch_members requires an existing rectilinear semantic frame model.")

    member_families = semantic_model.setdefault("member_families", {})
    updated = False
    for arg_name, family_name in (
        ("column_profile", "columns"),
        ("beam_x_profile", "beams_x"),
        ("beam_z_profile", "beams_z"),
    ):
        if arg_name in args:
            profile_name = str(args.get(arg_name, "")).strip().upper()
            if not profile_name:
                return _tool_error(f"'{arg_name}' must be a non-empty profile name.")
            member_families[family_name] = {"profile_name": profile_name}
            updated = True

    if "material_grade" in args:
        material_grade = str(args.get("material_grade", "")).strip().upper()
        if material_grade not in {"S235", "S275", "S355", "S420", "S460"}:
            return _tool_error("fea_patch_members requires a supported material_grade.")
        semantic_model.setdefault("material", {})["grade"] = material_grade
        updated = True

    if not updated:
        return _tool_error("fea_patch_members requires at least one member profile or material_grade change.")
    if semantic_model.get("dimension") == "2d" and "beams_z" not in member_families:
        member_families["beams_z"] = {"profile_name": member_families.get("beams_x", {}).get("profile_name", "")}
    return _rebuild_authoring_state_from_semantic_model(state, semantic_model, root)


def _handle_patch_loads(args: dict, state: FEAModelState, root: Path) -> tuple[list[dict], str]:
    semantic_model = deepcopy(state.semantic_model)
    if not isinstance(semantic_model, dict) or semantic_model.get("kind") != _SEMANTIC_MODEL_KIND:
        return _tool_error("fea_patch_loads requires an existing rectilinear semantic frame model.")

    mode = str(args.get("mode", "upsert") or "upsert").strip().lower()
    if mode not in {"replace_all", "upsert"}:
        return _tool_error("fea_patch_loads mode must be 'replace_all' or 'upsert'.")
    load_cases, error = _normalize_load_cases(args.get("load_cases"))
    if error:
        return _tool_error(error)

    if mode == "replace_all":
        semantic_model["load_cases"] = load_cases
    else:
        existing = {
            str(case.get("id")): deepcopy(case)
            for case in semantic_model.get("load_cases", [])
            if isinstance(case, dict) and str(case.get("id", "")).strip()
        }
        for load_case in load_cases:
            existing[str(load_case.get("id"))] = deepcopy(load_case)
        semantic_model["load_cases"] = list(existing.values())

    return _rebuild_authoring_state_from_semantic_model(state, semantic_model, root)


def _handle_clear(args: dict, state: FEAModelState, _root: Path) -> tuple[list[dict], str]:
    """Clear the entire model so the LLM can rebuild from scratch."""
    _clear_authoring_state(state)
    state.plan.clear()
    state.assumptions.clear()
    state.semantic_model = None
    return [{"action": "clear"}], "Model cleared. Ready to rebuild."


def _handle_check_model(args: dict, state: FEAModelState, _root: Path) -> tuple[list[dict], str]:
    """Validate the model and return structured JSON the LLM can reason about."""
    # Restraint DOF counts for each preset type
    _RESTRAINT_DOF = {
        "pin": 3, "pin_2d": 2,
        "fixed": 6, "fixed_2d": 3,
        "roller_x": 2, "roller_y": 2, "roller_2d": 1,
    }

    analysis: dict = {
        "status": "pass",
        "node_count": len(state.nodes),
        "element_count": len(state.elements),
        "restraint_count": len(state.restraints),
        "load_case_count": len(state.load_cases),
        "errors": [],
        "warnings": [],
        "dof_analysis": {},
        "connectivity": {},
        "geometry": {},
        "remediation": [],
    }

    errors = analysis["errors"]
    warnings = analysis["warnings"]
    remediation = analysis["remediation"]

    # ── Basic existence checks ──
    if not state.nodes:
        errors.append("No nodes defined")
    if not state.elements:
        errors.append("No elements defined")
    if not state.restraints:
        errors.append("No restraints defined — structure has no supports")
    if not state.load_cases:
        warnings.append("No load cases defined")
    else:
        invalid_load_refs: list[str] = []
        for lc_id, load_case in state.load_cases.items():
            for load in load_case.get("loads", []):
                node_id = str(load.get("node_id", "") or "").strip()
                element_id = str(load.get("element_id", "") or "").strip()
                if node_id and node_id not in state.nodes:
                    invalid_load_refs.append(f"{lc_id}: nodal load references undefined node {node_id}")
                if element_id and element_id not in state.elements:
                    invalid_load_refs.append(f"{lc_id}: distributed load references undefined element {element_id}")
        if invalid_load_refs:
            errors.extend(invalid_load_refs)
            remediation.append(
                "Update or remove loads that reference nodes/elements no longer present after the latest geometry edit."
            )

    # ── Element integrity ──
    for eid, el in state.elements.items():
        if not el.get("sectionId"):
            warnings.append(f"Element {eid} has no section assigned")
        if not el.get("materialId"):
            warnings.append(f"Element {eid} has no material assigned")
        for nid in el.get("nodeIds", []):
            if nid not in state.nodes:
                errors.append(f"Element {eid} references undefined node {nid}")

    # ── DOF analysis ──
    atype = state.analysis_type or "beam2d"
    dof_per_node = {"beam2d": 3, "truss2d": 2, "truss3d": 3, "frame3d": 6}.get(atype, 6)
    total_dof = len(state.nodes) * dof_per_node

    restrained_dof = 0
    for r in state.restraints.values():
        rtype = r.get("type", "pin")
        restrained_dof += _RESTRAINT_DOF.get(rtype, 0)

    free_dof = total_dof - restrained_dof
    min_required = {"beam2d": 3, "truss2d": 3, "truss3d": 6, "frame3d": 6}.get(atype, 6)

    analysis["dof_analysis"] = {
        "analysis_type": atype,
        "dof_per_node": dof_per_node,
        "total_dof": total_dof,
        "restrained_dof": restrained_dof,
        "free_dof": free_dof,
        "min_required": min_required,
        "is_potentially_unstable": restrained_dof < min_required,
    }

    if restrained_dof < min_required:
        warnings.append(
            f"Only {restrained_dof} DOFs restrained out of minimum {min_required} needed for {atype}."
        )
        remediation.append(
            f"Add more restraints. For {atype}, 'fixed' supports restrain "
            f"{_RESTRAINT_DOF.get('fixed_2d' if '2d' in atype else 'fixed', 6)} DOFs each."
        )

    # ── Connectivity / load path analysis ──
    if state.nodes and state.elements:
        adjacency: dict[str, set[str]] = {nid: set() for nid in state.nodes}
        for el in state.elements.values():
            nids = el.get("nodeIds", [])
            if len(nids) >= 2:
                adjacency.setdefault(nids[0], set()).add(nids[1])
                adjacency.setdefault(nids[1], set()).add(nids[0])

        # BFS from supported nodes
        visited: set[str] = set()
        queue = [nid for nid in state.restraints if nid in adjacency]
        while queue:
            n = queue.pop()
            if n in visited:
                continue
            visited.add(n)
            for nb in adjacency.get(n, set()):
                if nb not in visited:
                    queue.append(nb)

        disconnected = [nid for nid in state.nodes if nid not in visited]

        # Identify loaded nodes without a path to supports
        loaded_nodes: set[str] = set()
        for lc in state.load_cases.values():
            for load in lc.get("loads", []):
                if load.get("node_id"):
                    loaded_nodes.add(str(load["node_id"]))
                if load.get("element_id"):
                    el = state.elements.get(str(load["element_id"]))
                    if el:
                        loaded_nodes.update(el.get("nodeIds", []))
        loaded_but_disconnected = list(loaded_nodes & set(disconnected))

        analysis["connectivity"] = {
            "all_connected": len(disconnected) == 0,
            "disconnected_nodes": disconnected,
            "loaded_nodes_without_path": loaded_but_disconnected,
        }

        if disconnected:
            errors.append(f"Disconnected nodes: {', '.join(disconnected)} — no load path to supports")
            remediation.append(
                f"Connect nodes {', '.join(disconnected)} to the rest of the structure with elements, "
                f"or add restraints directly at those nodes."
            )
    else:
        analysis["connectivity"] = {"all_connected": False, "disconnected_nodes": [], "loaded_nodes_without_path": []}

    # ── Geometric checks ──
    zero_length = []
    for eid, el in state.elements.items():
        nids = el.get("nodeIds", [])
        if len(nids) >= 2:
            n1 = state.nodes.get(nids[0], {})
            n2 = state.nodes.get(nids[1], {})
            dx = n2.get("x", 0) - n1.get("x", 0)
            dy = n2.get("y", 0) - n1.get("y", 0)
            dz = n2.get("z", 0) - n1.get("z", 0)
            L = (dx**2 + dy**2 + dz**2) ** 0.5
            if L < 1e-6:
                zero_length.append(eid)

    # Check if all nodes are collinear (same Y) — might mean a frame is missing columns
    collinear = False
    if atype in ("beam2d", "truss2d") and len(state.nodes) > 2:
        y_vals = {round(n.get("y", 0), 1) for n in state.nodes.values()}
        if len(y_vals) == 1:
            collinear = True

    analysis["geometry"] = {
        "zero_length_elements": zero_length,
        "collinear_warning": collinear,
    }

    if zero_length:
        errors.append(f"Zero-length elements: {', '.join(zero_length)}")
        remediation.append(f"Check coordinates for elements {', '.join(zero_length)} — start and end nodes coincide.")
    if collinear:
        warnings.append("All nodes have the same Y coordinate. If this is a frame, add column nodes at different heights.")

    # ── Set overall status ──
    if errors:
        analysis["status"] = "fail"
    elif warnings:
        analysis["status"] = "warnings"

    return [], json.dumps(analysis, indent=2)


# ── Handler registry ──────────────────────────────────────────────

_TOOL_HANDLERS = {
    "todo_write": _handle_todo_write,
    "fea_query_model": _handle_query_model,
    "fea_define_rectilinear_frame": _handle_define_rectilinear_frame,
    "fea_patch_frame_geometry": _handle_patch_frame_geometry,
    "fea_patch_supports": _handle_patch_supports,
    "fea_patch_members": _handle_patch_members,
    "fea_patch_loads": _handle_patch_loads,
    "fea_add_nodes": _handle_add_nodes,
    "fea_add_elements": _handle_add_elements,
    "fea_assign_sections": _handle_assign_sections,
    "fea_assign_material": _handle_assign_material,
    "fea_set_restraints": _handle_set_restraints,
    "fea_add_loads": _handle_add_loads,
    "fea_set_analysis_type": _handle_set_analysis_type,
    "fea_solve": _handle_solve,
    "fea_get_results": _handle_get_results,
    "fea_set_view": _handle_set_view,
    "fea_check_model": _handle_check_model,
    "ask_user": _handle_ask_user,
    "fea_ask_user": _handle_ask_user,
    "fea_record_assumptions": _handle_record_assumptions,
    "fea_clear": _handle_clear,
}
