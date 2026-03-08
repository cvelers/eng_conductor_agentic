"""FEA tool executor — processes FEA analyst tool calls into frontend commands.

Each tool call generates JSON commands that the frontend FEA engine executes.
No computation happens here; this is a command translator.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Load profile database for validation ──────────────────────────

_profile_db: dict | None = None


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
        return [], f"Unknown FEA tool: {tool_name}"
    return handler(args, model_state, project_root)


def _handle_add_nodes(args: dict, state: FEAModelState, _root: Path) -> tuple[list[dict], str]:
    nodes = args.get("nodes", [])
    # Normalize IDs to strings (LLM may return ints)
    for n in nodes:
        n["id"] = str(n["id"])
    commands = [{"action": "add_nodes", "nodes": nodes}]
    for n in nodes:
        state.nodes[n["id"]] = {"x": n.get("x", 0), "y": n.get("y", 0), "z": n.get("z", 0)}
    return commands, f"Added {len(nodes)} nodes: {', '.join(n['id'] for n in nodes)}"


def _handle_add_elements(args: dict, state: FEAModelState, _root: Path) -> tuple[list[dict], str]:
    elements = args.get("elements", [])
    # Normalize IDs and node_ids to strings, handle variant key names
    for e in elements:
        e["id"] = str(e["id"])
        # LLM may use many different key names for node connectivity
        nids = (
            e.get("node_ids")
            or e.get("nodeIds")
            or e.get("nodes")
            or e.get("node_id")  # singular
        )
        # Handle start_node/end_node, node_start/node_end, from/to, n1/n2, etc.
        if not nids:
            start = (
                e.get("node_start") or e.get("nodeStart") or e.get("start_node") or e.get("startNode")
                or e.get("from_node") or e.get("fromNode") or e.get("from")
                or e.get("node1") or e.get("n1") or e.get("start") or e.get("i")
                or e.get("node_i") or e.get("nodeI")
            )
            end = (
                e.get("node_end") or e.get("nodeEnd") or e.get("end_node") or e.get("endNode")
                or e.get("to_node") or e.get("toNode") or e.get("to")
                or e.get("node2") or e.get("n2") or e.get("end") or e.get("j")
                or e.get("node_j") or e.get("nodeJ")
            )
            if start is not None and end is not None:
                nids = [start, end]
        if not nids:
            # Last resort: log raw element keys for debugging
            logger.warning("fea_add_elements: could not find node connectivity", extra={"element_keys": list(e.keys()), "raw_element": {k: v for k, v in e.items() if k != "id"}})
            nids = []
        e["node_ids"] = [str(nid) for nid in (nids if isinstance(nids, list) else [nids])]
    commands = [{"action": "add_elements", "elements": elements}]
    for e in elements:
        state.elements[e["id"]] = {"type": e.get("type", "beam"), "nodeIds": e.get("node_ids", [])}
    logger.info("fea_add_elements", extra={"elements": [{"id": e["id"], "node_ids": e.get("node_ids")} for e in elements]})
    return commands, f"Added {len(elements)} elements: {', '.join(e['id'] for e in elements)}"


def _handle_assign_sections(args: dict, state: FEAModelState, root: Path) -> tuple[list[dict], str]:
    elem_ids = [str(eid) for eid in args.get("element_ids", [])]
    profile_name = args.get("profile_name", "IPE300")
    props = _lookup_profile(root, profile_name)

    # Include properties so the frontend doesn't depend on fetching the JSON database
    cmd = {"action": "assign_section", "element_ids": elem_ids, "profile_name": profile_name}
    if props:
        cmd["properties"] = props
    commands = [cmd]

    if props:
        sec_info = f"Assigned {profile_name} (A={props.get('A',0)} mm², Iy={props.get('Iy',0)} mm⁴, h={props.get('h',0)} mm)"
    else:
        sec_info = f"Assigned {profile_name} (profile not found in database — will use defaults)"

    for eid in elem_ids:
        if eid in state.elements:
            state.elements[eid]["sectionId"] = f"sec_{profile_name}"
    state.sections[f"sec_{profile_name}"] = props or {"profileName": profile_name}

    return commands, sec_info


def _handle_assign_material(args: dict, state: FEAModelState, _root: Path) -> tuple[list[dict], str]:
    elem_ids = [str(eid) for eid in args.get("element_ids", [])]
    grade = args.get("grade", "S355")
    commands = [{"action": "assign_material", "element_ids": elem_ids, "grade": grade}]

    for eid in elem_ids:
        if eid in state.elements:
            state.elements[eid]["materialId"] = f"mat_{grade}"
    state.materials[f"mat_{grade}"] = {"name": grade}

    return commands, f"Assigned material {grade} to elements {', '.join(str(e) for e in elem_ids)}"


def _handle_set_restraints(args: dict, state: FEAModelState, _root: Path) -> tuple[list[dict], str]:
    _RESTRAINT_ALIASES = {
        "pinned": "pin", "hinge": "pin", "simply_supported": "pin",
        "roller": "roller_x", "roller_horizontal": "roller_x",
        "fixed_support": "fixed", "encastre": "fixed", "clamped": "fixed",
    }
    restraints = args.get("restraints", [])
    for r in restraints:
        r["node_id"] = str(r.get("node_id", r.get("nodeId", "")))
        # Normalize restraint type
        rtype = r.get("type", "pin").lower().replace("-", "_").replace(" ", "_")
        r["type"] = _RESTRAINT_ALIASES.get(rtype, rtype)
    commands = [{"action": "set_restraints", "restraints": restraints}]
    for r in restraints:
        nid = r.get("node_id", "")
        state.restraints[nid] = {"type": r.get("type", "pin")}
    descs = [f"{r.get('node_id')}: {r.get('type', 'pin')}" for r in restraints]
    return commands, f"Set restraints: {'; '.join(descs)}"


def _handle_add_loads(args: dict, state: FEAModelState, _root: Path) -> tuple[list[dict], str]:
    lc_id = str(args.get("load_case_id", "LC1"))
    loads = args.get("loads", [])
    for load in loads:
        if "node_id" in load:
            load["node_id"] = str(load["node_id"])
        if "element_id" in load:
            load["element_id"] = str(load["element_id"])
        # Normalize load type variants
        lt = load.get("type", "").lower()
        if lt in ("udl", "uniform", "uniform_distributed", "line_load"):
            load["type"] = "distributed"
            # Normalize value field to qy (downward) if no qx/qy/qz specified
            if "value" in load and not any(k in load for k in ("qx", "qy", "qz")):
                val = load.pop("value")
                load["qy"] = -abs(val)  # downward convention
        elif lt in ("point", "concentrated", "point_load"):
            load["type"] = "nodal"
    commands = [{"action": "add_loads", "load_case_id": lc_id, "loads": loads}]

    if lc_id not in state.load_cases:
        state.load_cases[lc_id] = {"loads": []}
    state.load_cases[lc_id]["loads"].extend(loads)

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

    return commands, f"Load case {lc_id}: {'; '.join(descs)}"


def _handle_set_analysis_type(args: dict, state: FEAModelState, _root: Path) -> tuple[list[dict], str]:
    atype = args.get("type", "beam2d")
    state.analysis_type = atype
    commands = [{"action": "set_analysis_type", "type": atype}]
    return commands, f"Analysis type set to {atype}"


def _handle_solve(args: dict, state: FEAModelState, _root: Path) -> tuple[list[dict], str]:
    lc_id = args.get("load_case_id", "LC1")
    state.solved = False
    # This is a special tool — the caller (FEAAnalystLoop) handles the solve request specially
    return [{"action": "solve", "load_case_id": lc_id}], f"__SOLVE_REQUEST__|{lc_id}"


def _handle_get_results(args: dict, state: FEAModelState, _root: Path) -> tuple[list[dict], str]:
    query = args.get("query", "summary")
    # Normalize query aliases
    _QUERY_ALIASES = {
        "displacements": "all_displacements",
        "nodal_displacements": "all_displacements",
        "displacement": "max_displacement",
        "max_disp": "max_displacement",
        "forces": "element_forces",
        "member_forces": "element_forces",
        "internal_forces": "element_forces",
        "reaction": "reactions",
        "support_reactions": "reactions",
    }
    query = _QUERY_ALIASES.get(query, query)
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
        disps = results.get("displacements", {})
        text = "Nodal displacements (mm):\n"
        for nid, d in disps.items():
            mag = (d.get("dx", 0)**2 + d.get("dy", 0)**2 + d.get("dz", 0)**2)**0.5
            if mag > 1e-6:
                text += f"  Node {nid}: dx={d.get('dx', 0):.4f}, dy={d.get('dy', 0):.4f}, dz={d.get('dz', 0):.4f} (mag={mag:.4f})\n"
        return [], text

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

    if query == "all_displacements":
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


def _handle_clear(args: dict, state: FEAModelState, _root: Path) -> tuple[list[dict], str]:
    """Clear the entire model so the LLM can rebuild from scratch."""
    state.nodes.clear()
    state.elements.clear()
    state.sections.clear()
    state.materials.clear()
    state.restraints.clear()
    state.load_cases.clear()
    state.solved = False
    state.results = None
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
    "fea_ask_user": _handle_ask_user,
    "fea_clear": _handle_clear,
}
