from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterator

from backend.config import Settings
from backend.llm.base import LLMProvider
from backend.registries.document_registry import ClauseRecord
from backend.registries.tool_registry import ToolRegistryEntry
from backend.retrieval.agentic_search import AgenticRetriever, RetrievedClause
from backend.schemas import ChatResponse, Citation, RetrievalTraceStep, ToolTraceStep
from backend.tools.runner import MCPToolRunner
from backend.utils.citations import build_citation_address
from backend.utils.parsing import apply_defaults, parse_user_inputs

logger = logging.getLogger(__name__)

_UNIT_SUFFIXES = [
    ("_kNm", " (kNm)"),
    ("_kN", " (kN)"),
    ("_MPa", " (MPa)"),
    ("_GPa", " (GPa)"),
    ("_mm", " (mm)"),
    ("_cm2", " (cm²)"),
    ("_cm3", " (cm³)"),
    ("_cm4", " (cm⁴)"),
    ("_m", " (m)"),
]

_KEY_SUBSCRIPTS: dict[str, str] = {
    "M_Rd": "M<sub>Rd</sub>",
    "N_Rd": "N<sub>Rd</sub>",
    "V_Rd": "V<sub>Rd</sub>",
    "M_Ed": "M<sub>Ed</sub>",
    "N_Ed": "N<sub>Ed</sub>",
    "V_Ed": "V<sub>Ed</sub>",
    "N_b_Rd": "N<sub>b,Rd</sub>",
    "Fv_Rd": "F<sub>v,Rd</sub>",
    "Fv": "F<sub>v</sub>",
    "fy": "f<sub>y</sub>",
    "fu": "f<sub>u</sub>",
    "fub": "f<sub>ub</sub>",
    "Wpl": "W<sub>pl</sub>",
    "Wel": "W<sub>el</sub>",
    "L_cr": "L<sub>cr</sub>",
    "gamma_M0": "γ<sub>M0</sub>",
    "gamma_M1": "γ<sub>M1</sub>",
    "gamma_M2": "γ<sub>M2</sub>",
    "alpha_v": "α<sub>v</sub>",
}

_NARRATIVE_SUB_RE: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"gamma_M([012])"), r"γ<sub>M\1</sub>"),
    (re.compile(r"γ_M([012])"), r"γ<sub>M\1</sub>"),
    (re.compile(r"γM([012])"), r"γ<sub>M\1</sub>"),
    (re.compile(r"N_b,Rd"), "N<sub>b,Rd</sub>"),
    (re.compile(r"Fv[_,]Rd"), "F<sub>v,Rd</sub>"),
    (re.compile(r"M_Rd"), "M<sub>Rd</sub>"),
    (re.compile(r"N_Rd"), "N<sub>Rd</sub>"),
    (re.compile(r"V_Rd"), "V<sub>Rd</sub>"),
    (re.compile(r"M_Ed"), "M<sub>Ed</sub>"),
    (re.compile(r"N_Ed"), "N<sub>Ed</sub>"),
    (re.compile(r"V_Ed"), "V<sub>Ed</sub>"),
    (re.compile(r"F_v\b"), "F<sub>v</sub>"),
    (re.compile(r"f_ub"), "f<sub>ub</sub>"),
    (re.compile(r"f_y"), "f<sub>y</sub>"),
    (re.compile(r"f_u\b"), "f<sub>u</sub>"),
    (re.compile(r"W_pl"), "W<sub>pl</sub>"),
    (re.compile(r"W_el"), "W<sub>el</sub>"),
    (re.compile(r"L_cr"), "L<sub>cr</sub>"),
    (re.compile(r"α_?v\b"), "α<sub>v</sub>"),
]


@dataclass
class PlanResult:
    mode: str
    tools: list[str]
    rationale: str


TOOL_KEYWORD_MAP = {
    "simple_beam_calculator": [
        "simply supported", "simple beam", "udl", "uniform load", "point load", "bending moment",
        "beam moment", "beam shear", "beam deflection",
    ],
    "cantilever_beam_calculator": ["cantilever"],
    "steel_grade_properties": [
        "steel grade", "yield strength", "fy value", "fu value", "material properties",
        "steel properties", "table 3.1",
    ],
    "effective_length_ec3": ["effective length", "buckling length", "k factor", "l_cr"],
    "column_buckling_ec3": [
        "column buckling", "buckling resistance", "nb,rd", "compression member",
        "flexural buckling", "buckling check",
    ],
    "bolt_shear_ec3": [
        "bolt", "bolt shear", "bolt resistance", "m20", "m16", "m24",
        "bolt capacity", "bolt class", "8.8", "10.9",
    ],
    "weld_resistance_ec3": [
        "weld", "fillet weld", "weld resistance", "weld capacity", "throat",
    ],
    "deflection_check": [
        "deflection check", "deflection limit", "serviceability", "l/250", "l/300",
    ],
    "unit_converter": ["convert", "conversion", "unit convert", "convert unit"],
    "section_classification_ec3": ["classification", "section class", "class 1", "class 2"],
    "member_resistance_ec3": [
        "resistance", "capacity", "m_rd", "n_rd", "v_rd", "axial resistance",
        "shear resistance", "bending resistance",
    ],
    "interaction_check_ec3": ["interaction", "combined", "axial and bending", "utilization"],
    "ipe_moment_resistance_ec3": ["ipe", "moment resistance", "m_rd"],
}


class CentralIntelligenceOrchestrator:
    def __init__(
        self,
        *,
        settings: Settings,
        orchestrator_llm: LLMProvider,
        retriever: AgenticRetriever,
        tool_runner: MCPToolRunner,
        tool_registry: list[ToolRegistryEntry],
    ) -> None:
        self.settings = settings
        self.orchestrator_llm = orchestrator_llm
        self.retriever = retriever
        self.tool_runner = tool_runner
        self.tool_registry = {entry.tool_name: entry for entry in tool_registry}

    def run(self, query: str, *, history: list | None = None) -> ChatResponse:
        final_response: ChatResponse | None = None
        for event_type, payload in self.run_stream(query, history=history):
            if event_type == "response":
                final_response = payload
        if final_response is None:
            raise RuntimeError("Orchestrator did not produce a final response.")
        return final_response

    def run_stream(self, raw_query: str, *, history: list | None = None) -> Iterator[tuple[str, Any]]:
        query = self._resolve_followup(raw_query, history or [])

        # --- INTAKE ---
        yield ("machine", {"node": "intake", "status": "active", "title": "Query Intake", "detail": "Analyzing your question..."})
        plan = self._build_plan(query)
        yield ("machine", {"node": "intake", "status": "done", "title": "Query Intake", "detail": "Question understood."})
        yield ("machine", {
            "node": "plan", "status": "done", "title": "Pathway Planning",
            "detail": f"Strategy: {plan.mode} | Tools: {plan.tools or ['none']}",
            "meta": {"mode": plan.mode, "tools": plan.tools, "rationale": plan.rationale},
        })

        requires_tools = plan.mode in {"calculator", "hybrid"} and bool(plan.tools)

        # --- INPUT RESOLUTION ---
        yield ("machine", {"node": "inputs", "status": "active", "title": "Input Resolution", "detail": "Extracting values from your query..."})
        user_inputs = parse_user_inputs(query)
        assumed_inputs, assumptions = apply_defaults(query, user_inputs, self.settings, requires_tools)
        yield ("machine", {
            "node": "inputs", "status": "done", "title": "Input Resolution",
            "detail": f"Found {len(user_inputs)} values, filled {len(assumed_inputs)} defaults.",
            "meta": {"user_inputs": user_inputs, "assumed_inputs": assumed_inputs, "assumptions": assumptions},
        })

        # --- RETRIEVAL ---
        yield ("machine", {"node": "retrieval", "status": "active", "title": "Database Search", "detail": "Searching EC3 clauses..."})
        retrieved: list[RetrievedClause] = []
        retrieval_trace: list[dict[str, object]] = []
        for retrieval_event in self.retriever.iter_retrieve(query, top_k=self.settings.top_k_clauses):
            etype = retrieval_event.get("type")
            if etype == "iteration":
                step = retrieval_event.get("step", {})
                top = retrieval_event.get("top", [])
                top_labels = ", ".join(str(i.get("clause_id", "?")) for i in top) or "none"
                yield ("machine", {
                    "node": "retrieval", "status": "active", "title": "Database Search",
                    "detail": f"Pass {step.get('iteration', '?')}: found {len(step.get('top_clause_ids', []))} matches",
                    "meta": {"iteration": step, "top": top},
                })
            elif etype == "recursive":
                yield ("machine", {"node": "retrieval", "status": "active", "title": "Database Search", "detail": str(retrieval_event.get("detail", "Expanding search..."))})
            elif etype == "final":
                retrieved = retrieval_event.get("results", [])
                retrieval_trace = retrieval_event.get("trace", [])

        yield ("machine", {
            "node": "retrieval", "status": "done", "title": "Database Search",
            "detail": f"Retrieved {len(retrieved)} relevant clauses.",
            "meta": {
                "retrieved_count": len(retrieved),
                "top_clauses": [
                    {"doc_id": i.clause.doc_id, "clause_id": i.clause.clause_id, "title": i.clause.clause_title, "pointer": i.clause.pointer}
                    for i in retrieved[:5]
                ],
            },
        })

        # --- TOOLS ---
        tool_trace: list[ToolTraceStep] = []
        tool_outputs: dict[str, dict[str, Any]] = {}

        if not plan.tools:
            yield ("machine", {"node": "tools", "status": "done", "title": "MCP Tools", "detail": "No tools needed — retrieval-only path."})
        else:
            yield ("machine", {"node": "tools", "status": "active", "title": "MCP Tools", "detail": f"Running {len(plan.tools)} tool(s)..."})
            for idx, tool_name in enumerate(plan.tools, 1):
                inputs = self._build_tool_inputs(tool_name, user_inputs, assumed_inputs, tool_outputs)
                yield ("machine", {
                    "node": "tools", "status": "active", "title": "MCP Tools",
                    "detail": f"[{idx}/{len(plan.tools)}] {tool_name}",
                    "meta": {"tool": tool_name, "inputs": inputs},
                })
                try:
                    payload = self.tool_runner.run(tool_name, inputs)
                    tool_outputs[tool_name] = payload.get("result", {})
                    tool_trace.append(ToolTraceStep(tool_name=tool_name, status="ok", inputs=inputs, outputs=payload.get("result", {})))
                    yield ("machine", {"node": "tools", "status": "active", "title": "MCP Tools", "detail": f"{tool_name} — done", "meta": {"tool": tool_name, "status": "ok"}})
                except Exception as exc:
                    tool_trace.append(ToolTraceStep(tool_name=tool_name, status="error", inputs=inputs, error=str(exc)))
                    yield ("machine", {"node": "tools", "status": "error", "title": "MCP Tools", "detail": f"{tool_name} failed: {exc}", "meta": {"tool": tool_name, "status": "error"}})

            ts = "error" if any(s.status == "error" for s in tool_trace) else "done"
            yield ("machine", {"node": "tools", "status": ts, "title": "MCP Tools", "detail": f"Tools finished ({ts})."})

        # --- COMPOSE ---
        yield ("machine", {"node": "compose", "status": "active", "title": "Composing Answer", "detail": "Building grounded response..."})

        sources = self._collect_sources(retrieved, tool_outputs)
        supported = bool(sources)
        if requires_tools and any(s.status == "error" for s in tool_trace):
            supported = False

        narrative = self._draft_grounded_narrative(query=query, plan=plan, retrieved=retrieved, tool_outputs=tool_outputs, supported=supported)
        answer = self._build_markdown_answer(
            query=query, plan=plan, narrative=narrative, supported=supported,
            user_inputs=user_inputs, assumed_inputs=assumed_inputs, assumptions=assumptions,
            retrieved=retrieved, tool_outputs=tool_outputs, sources=sources, tool_trace=tool_trace,
        )

        yield ("machine", {
            "node": "compose", "status": "done" if supported else "error", "title": "Composing Answer",
            "detail": "Response ready." if supported else "Limited source support.",
            "meta": {
                "supported": supported,
                "used_tools": [s.tool_name for s in tool_trace if s.status == "ok"],
                "used_sources": [{"doc_id": s.doc_id, "clause_id": s.clause_id, "pointer": s.pointer} for s in sources[:8]],
            },
        })

        # --- OUTPUT ---
        yield ("machine", {"node": "output", "status": "active", "title": "Streaming", "detail": "Sending response..."})

        what_i_used = self._build_what_i_used(plan, retrieval_trace, tool_trace)
        response = ChatResponse(
            answer=answer, supported=supported,
            user_inputs=user_inputs, assumed_inputs=assumed_inputs, assumptions=assumptions,
            sources=sources, tool_trace=tool_trace,
            retrieval_trace=[RetrievalTraceStep.model_validate(s) for s in retrieval_trace],
            what_i_used=what_i_used,
        )
        yield ("machine", {"node": "output", "status": "done", "title": "Streaming", "detail": "Complete."})
        yield ("response", response)

    # ---- PLANNING ----
    def _build_plan(self, query: str) -> PlanResult:
        valid_tools = list(self.tool_registry.keys())
        lowered = query.lower()

        heuristic_plan = self._heuristic_plan(lowered, valid_tools)
        if heuristic_plan:
            return heuristic_plan

        if self.orchestrator_llm.available:
            try:
                tool_descriptions = "\n".join(
                    f"- {name}: {self.tool_registry[name].description}"
                    for name in valid_tools
                )
                raw = self.orchestrator_llm.generate(
                    system_prompt="You are the Central Intelligence Orchestrator for Eurocodes. Plan a grounded tool/retrieval pathway. Return JSON only.",
                    user_prompt=(
                        "###TASK:PLAN###\n"
                        f"User query: {query}\n\n"
                        f"Available tools:\n{tool_descriptions}\n\n"
                        "Return JSON: {\"mode\":\"retrieval_only|calculator|hybrid\",\"tools\":[...],\"rationale\":\"...\"}\n"
                        "Rules:\n"
                        "- Use 'retrieval_only' for explanation/conceptual questions\n"
                        "- Use 'calculator' for pure computation queries\n"
                        "- Use 'hybrid' when both explanation and computation are needed\n"
                        "- Order tools in execution dependency order\n"
                        "- Only include tools that are directly relevant"
                    ),
                    temperature=0,
                    max_tokens=300,
                )
                parsed = json.loads(raw)
                mode = parsed.get("mode", "retrieval_only")
                tools = [t for t in parsed.get("tools", []) if t in valid_tools]
                rationale = str(parsed.get("rationale", "LLM-generated plan."))
                if mode in {"retrieval_only", "calculator", "hybrid"}:
                    return PlanResult(mode=mode, tools=tools, rationale=rationale)
            except Exception as exc:
                logger.warning("plan_generation_failed", extra={"error": str(exc)})

        return self._fallback_plan(lowered, valid_tools)

    def _heuristic_plan(self, lowered: str, valid: list[str]) -> PlanResult | None:
        explanation_like = any(t in lowered for t in ["explain", "what is", "what are", "how does", "clause", "rules", "describe", "overview"])

        if "cantilever" in lowered and any(t in lowered for t in ["beam", "moment", "shear", "load", "deflection"]):
            tools = ["cantilever_beam_calculator"]
            if "deflection" in lowered and "check" in lowered:
                tools.append("deflection_check")
            return PlanResult(mode="hybrid" if explanation_like else "calculator", tools=[t for t in tools if t in valid], rationale="Heuristic: cantilever beam query")

        if ("simply supported" in lowered or "simple beam" in lowered or "udl" in lowered) and any(t in lowered for t in ["moment", "shear", "load", "deflection", "span", "beam"]):
            tools = ["simple_beam_calculator"]
            if "deflection" in lowered and "check" in lowered:
                tools.append("deflection_check")
            return PlanResult(mode="hybrid" if explanation_like else "calculator", tools=[t for t in tools if t in valid], rationale="Heuristic: simply supported beam query")

        if any(t in lowered for t in ["bolt", "m20", "m16", "m24", "m12"]) and any(t in lowered for t in ["shear", "resistance", "capacity"]):
            return PlanResult(mode="hybrid" if explanation_like else "calculator", tools=["bolt_shear_ec3"] if "bolt_shear_ec3" in valid else [], rationale="Heuristic: bolt shear query")

        if any(t in lowered for t in ["weld", "fillet"]) and any(t in lowered for t in ["resistance", "capacity", "strength"]):
            return PlanResult(mode="hybrid" if explanation_like else "calculator", tools=["weld_resistance_ec3"] if "weld_resistance_ec3" in valid else [], rationale="Heuristic: weld resistance query")

        if any(t in lowered for t in ["column buckling", "buckling resistance", "compression member", "flexural buckling"]):
            tools = ["column_buckling_ec3"]
            return PlanResult(mode="hybrid" if explanation_like else "calculator", tools=[t for t in tools if t in valid], rationale="Heuristic: column buckling query")

        if any(t in lowered for t in ["effective length", "buckling length"]):
            return PlanResult(mode="hybrid" if explanation_like else "calculator", tools=["effective_length_ec3"] if "effective_length_ec3" in valid else [], rationale="Heuristic: effective length query")

        if any(t in lowered for t in ["steel grade", "yield strength", "material prop"]) and not any(t in lowered for t in ["resistance", "capacity"]):
            return PlanResult(mode="hybrid" if explanation_like else "calculator", tools=["steel_grade_properties"] if "steel_grade_properties" in valid else [], rationale="Heuristic: steel properties query")

        if any(t in lowered for t in ["convert", "conversion"]) and any(t in lowered for t in ["unit", "mm", "kn", "mpa", "psi", "inch", "foot"]):
            return PlanResult(mode="calculator", tools=["unit_converter"] if "unit_converter" in valid else [], rationale="Heuristic: unit conversion query")

        if "deflection" in lowered and any(t in lowered for t in ["check", "limit", "l/250", "serviceability"]):
            return PlanResult(mode="hybrid" if explanation_like else "calculator", tools=["deflection_check"] if "deflection_check" in valid else [], rationale="Heuristic: deflection check query")

        ipe_moment = "ipe" in lowered and any(t in lowered for t in ["moment resistance", "bending resistance", "m_rd"]) and "interaction" not in lowered
        if ipe_moment and "ipe_moment_resistance_ec3" in valid:
            return PlanResult(mode="hybrid" if explanation_like else "calculator", tools=["ipe_moment_resistance_ec3"], rationale="Heuristic: IPE moment resistance")

        return None

    def _fallback_plan(self, lowered: str, valid: list[str]) -> PlanResult:
        explanation_like = any(t in lowered for t in ["explain", "what", "clause", "rules", "classification", "describe"])
        calculation_like = any(t in lowered for t in ["given", "resistance", "capacity", "check", "m_ed", "n_ed", "bending", "calculate", "compute"])
        interaction_like = "interaction" in lowered or ("combined" in lowered and ("bending" in lowered or "axial" in lowered))

        if calculation_like:
            tools = ["section_classification_ec3", "member_resistance_ec3"]
            if interaction_like:
                tools.append("interaction_check_ec3")
            return PlanResult(mode="hybrid" if explanation_like else "calculator", tools=[t for t in tools if t in valid], rationale="Fallback heuristic plan.")

        return PlanResult(mode="retrieval_only", tools=[], rationale="Retrieval-only fallback.")

    # ---- FOLLOW-UP RESOLUTION ----
    def _resolve_followup(self, query: str, history: list) -> str:
        """Expand short follow-up queries using conversation history."""
        if not history:
            return query

        lowered = query.lower().strip()
        is_short = len(query.split()) <= 10
        followup_phrases = [
            "same but", "do it", "again", "now for", "now with",
            "repeat", "what about", "how about", "and for",
            "change to", "try with", "instead of", "but for",
            "ok ", "ok,", "for s", "with s", "use s",
        ]
        is_referential = any(phrase in lowered for phrase in followup_phrases)

        if not is_short and not is_referential:
            return query

        anchor_msg = None
        for msg in history:
            role = msg.role if hasattr(msg, "role") else msg.get("role", "")
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if role == "user" and content:
                anchor_msg = content
                break

        if not anchor_msg:
            return query

        current_inputs = parse_user_inputs(query)

        if self.orchestrator_llm.available:
            try:
                raw = self.orchestrator_llm.generate(
                    system_prompt=(
                        "You expand short follow-up messages into self-contained engineering queries. "
                        "Keep the ORIGINAL intent and all technical parameters from the first question. "
                        "Override only what the follow-up explicitly changes. "
                        "Return ONLY the expanded query as a single sentence. No explanation."
                    ),
                    user_prompt=(
                        f"Original question: {anchor_msg}\n"
                        f"Follow-up: {query}\n\n"
                        "Expanded self-contained query:"
                    ),
                    temperature=0,
                    max_tokens=150,
                )
                resolved = raw.strip()
                if resolved and len(resolved) > len(query):
                    resolved_lower = resolved.lower()
                    all_preserved = all(
                        str(val).lower() in resolved_lower
                        for val in current_inputs.values()
                    )
                    if all_preserved:
                        logger.info("followup_resolved", extra={"original": query, "resolved": resolved})
                        return resolved
                    logger.info("followup_llm_drift", extra={"resolved": resolved})
            except Exception as exc:
                logger.warning("followup_resolution_failed", extra={"error": str(exc)})

        logger.info("followup_heuristic", extra={"original": query, "context": anchor_msg[:120]})
        return f"{query}. {anchor_msg}"

    # ---- TOOL INPUT BUILDERS ----
    def _build_tool_inputs(
        self, tool_name: str, user_inputs: dict[str, Any], assumed_inputs: dict[str, Any], tool_outputs: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        merged = {**assumed_inputs, **user_inputs}

        if tool_name == "section_classification_ec3":
            return {"section_name": merged.get("section_name"), "steel_grade": merged.get("steel_grade", self.settings.default_steel_grade)}

        if tool_name == "member_resistance_ec3":
            gov_class = tool_outputs.get("section_classification_ec3", {}).get("outputs", {}).get("governing_class", 2)
            return {"section_name": merged.get("section_name"), "steel_grade": merged.get("steel_grade", self.settings.default_steel_grade), "section_class": gov_class, "gamma_M0": merged.get("gamma_M0", self.settings.default_gamma_m0)}

        if tool_name == "interaction_check_ec3":
            res = tool_outputs.get("member_resistance_ec3", {}).get("outputs", {})
            return {"MEd_kNm": merged.get("MEd_kNm", self.settings.default_med_knm), "NEd_kN": merged.get("NEd_kN", self.settings.default_ned_kn), "M_Rd_kNm": res.get("M_Rd_kNm"), "N_Rd_kN": res.get("N_Rd_kN")}

        if tool_name == "ipe_moment_resistance_ec3":
            return {"section_name": merged.get("section_name", self.settings.default_section_name), "steel_grade": merged.get("steel_grade", self.settings.default_steel_grade), "section_class": int(merged.get("section_class", 2)), "gamma_M0": merged.get("gamma_M0", self.settings.default_gamma_m0)}

        if tool_name == "simple_beam_calculator":
            inp: dict[str, Any] = {"span_m": merged.get("length_m", 6.0)}
            load_kn = merged.get("load_kn") or merged.get("P_kN")
            load_kn_per_m = merged.get("load_kn_per_m") or merged.get("w_kN_per_m")
            if load_kn:
                inp["load_type"] = "point_mid"
                inp["load_kn"] = float(load_kn)
            elif load_kn_per_m:
                inp["load_type"] = "udl"
                inp["load_kn_per_m"] = float(load_kn_per_m)
            else:
                inp["load_type"] = "udl"
                inp["load_kn_per_m"] = 10.0
            return inp

        if tool_name == "cantilever_beam_calculator":
            inp = {"span_m": merged.get("length_m", 3.0)}
            load_kn = merged.get("load_kn") or merged.get("P_kN")
            load_kn_per_m = merged.get("load_kn_per_m") or merged.get("w_kN_per_m")
            if load_kn_per_m:
                inp["load_type"] = "udl"
                inp["load_kn_per_m"] = float(load_kn_per_m)
            else:
                inp["load_type"] = "point_tip"
                inp["load_kn"] = float(load_kn or 10.0)
            return inp

        if tool_name == "steel_grade_properties":
            return {"steel_grade": merged.get("steel_grade", self.settings.default_steel_grade), "thickness_mm": merged.get("thickness_mm")}

        if tool_name == "effective_length_ec3":
            return {"support_conditions": merged.get("support_conditions", "pinned-pinned"), "system_length_m": merged.get("length_m", 5.0)}

        if tool_name == "column_buckling_ec3":
            return {
                "section_name": merged.get("section_name", self.settings.default_section_name),
                "steel_grade": merged.get("steel_grade", self.settings.default_steel_grade),
                "system_length_m": merged.get("length_m", 5.0),
                "k_factor": merged.get("k_factor", 1.0),
                "buckling_curve": merged.get("buckling_curve", "b"),
                "gamma_M1": merged.get("gamma_M1", 1.0),
            }

        if tool_name == "bolt_shear_ec3":
            inp = {}
            bolt_class = merged.get("bolt_class")
            if bolt_class:
                inp["bolt_class"] = str(bolt_class)
            bolt_diam = merged.get("bolt_diameter_mm")
            if bolt_diam:
                inp["bolt_diameter_mm"] = int(bolt_diam)
            n_bolts = merged.get("n_bolts")
            if n_bolts:
                inp["n_bolts"] = int(n_bolts)
            n_planes = merged.get("n_shear_planes")
            if n_planes:
                inp["n_shear_planes"] = int(n_planes)
            return inp

        if tool_name == "weld_resistance_ec3":
            return {
                "throat_thickness_mm": merged.get("throat_thickness_mm", 5.0),
                "weld_length_mm": merged.get("weld_length_mm", 200.0),
                "steel_grade": merged.get("steel_grade", self.settings.default_steel_grade),
            }

        if tool_name == "deflection_check":
            return {
                "span_m": merged.get("length_m", 6.0),
                "actual_deflection_mm": merged.get("actual_deflection_mm", 20.0),
                "limit_ratio": merged.get("limit_ratio", "L/250"),
            }

        if tool_name == "unit_converter":
            return {
                "value": merged.get("convert_value", 1.0),
                "from_unit": merged.get("from_unit", "mm"),
                "to_unit": merged.get("to_unit", "m"),
            }

        return merged

    # ---- SOURCE COLLECTION ----
    def _collect_sources(self, retrieved: list[RetrievedClause], tool_outputs: dict[str, dict[str, Any]]) -> list[Citation]:
        seen: set[str] = set()
        sources: list[Citation] = []
        for payload in tool_outputs.values():
            for ref in payload.get("clause_references", []):
                doc_id = str(ref.get("doc_id", "unknown-doc"))
                clause_id = str(ref.get("clause_id", "unknown-clause"))
                title = str(ref.get("title", "Tool-linked clause"))
                pointer = str(ref.get("pointer", "tool-output"))
                address = build_citation_address(doc_id, clause_id, pointer)
                if address in seen:
                    continue
                seen.add(address)
                sources.append(Citation(doc_id=doc_id, clause_id=clause_id, clause_title=title, pointer=pointer, citation_address=address))
        for item in retrieved:
            c = Citation(doc_id=item.clause.doc_id, clause_id=item.clause.clause_id, clause_title=item.clause.clause_title, pointer=item.clause.pointer, citation_address=item.clause.citation_address)
            if c.citation_address not in seen:
                seen.add(c.citation_address)
                sources.append(c)
        return sources

    # ---- NARRATIVE GENERATION ----
    def _draft_grounded_narrative(self, *, query: str, plan: PlanResult, retrieved: list[RetrievedClause], tool_outputs: dict[str, dict[str, Any]], supported: bool) -> str:
        if not supported:
            return "I don't have enough information in the currently indexed clauses or tools to give you a reliable answer on this. You'd need to add the relevant Eurocode section or a dedicated calculator tool."

        if not self.orchestrator_llm.available:
            return self._build_fallback_narrative(plan, retrieved, tool_outputs)

        clause_evidence = []
        for c in retrieved[:5]:
            snippet = c.clause.text.strip()[:300]
            clause_evidence.append(f"[{c.clause.clause_id} — {c.clause.clause_title}]: {snippet}")

        tool_evidence: dict[str, Any] = {}
        for tname, tout in tool_outputs.items():
            tool_evidence[tname] = {
                "inputs": tout.get("inputs_used", {}),
                "outputs": tout.get("outputs", {}),
                "notes": tout.get("notes", []),
            }

        try:
            raw = self.orchestrator_llm.generate(
                system_prompt=(
                    "You are a senior structural engineer giving a concise answer to a colleague. "
                    "Rules:\n"
                    "1. First sentence: state the key result with its value. Example: 'The design bending resistance is **M_Rd = 223.08 kNm**'.\n"
                    "2. Then 1-2 sentences on the method/formula used. Mention the governing clause once, e.g. (EC3-1-1, Cl. 6.2.5).\n"
                    "3. Bold **key numerical results** with their engineering symbols.\n"
                    "4. Use ONLY the provided evidence. Never invent values.\n"
                    "5. Keep it to 2-4 sentences total. No sections, no bullet lists, no 'Sources' or 'Assumptions'.\n"
                    "6. Write naturally, as if explaining at a desk review. Always finish your sentences."
                ),
                user_prompt=(
                    f"Question: {query}\n\n"
                    f"Retrieved EC3 clauses:\n" + "\n".join(clause_evidence) + "\n\n"
                    f"Tool results:\n{json.dumps(tool_evidence, default=str)}\n\n"
                    "Write a concise answer starting with the result."
                ),
                temperature=0.15,
                max_tokens=700,
            )
            return raw.strip()
        except Exception as exc:
            logger.warning("answer_generation_failed", extra={"error": str(exc)})
            return self._build_fallback_narrative(plan, retrieved, tool_outputs)

    def _build_fallback_narrative(self, plan: PlanResult, retrieved: list[RetrievedClause], tool_outputs: dict[str, dict[str, Any]]) -> str:
        parts: list[str] = []
        if tool_outputs:
            for tname, tout in tool_outputs.items():
                outputs = tout.get("outputs", {})
                pretty = tname.replace("_ec3", "").replace("_", " ").title()
                headline = ", ".join(f"**{self._format_value(k, v)}**" for k, v in outputs.items() if isinstance(v, (int, float)))
                if headline:
                    parts.append(f"{pretty}: {headline}.")
        if retrieved:
            top = retrieved[0]
            parts.append(f"Based on EC3-1-1, Cl. {top.clause.clause_id} ({top.clause.clause_title}).")
        return " ".join(parts) if parts else "Results computed from the available tools and EC3 database."

    # ---- MARKDOWN BUILDER ----
    def _build_markdown_answer(self, *, query: str, plan: PlanResult, narrative: str, supported: bool,
                                user_inputs: dict[str, Any], assumed_inputs: dict[str, Any], assumptions: list[str],
                                retrieved: list[RetrievedClause], tool_outputs: dict[str, dict[str, Any]],
                                sources: list[Citation], tool_trace: list[ToolTraceStep]) -> str:
        lines: list[str] = []

        if narrative:
            lines.append(narrative)
        elif not supported:
            lines.append("I can't provide a grounded answer with the currently available sources and tools.")
        else:
            lines.append("Here are the results based on the EC3 database and calculation tools.")

        if tool_outputs:
            lines.append("")
            for tool_name in plan.tools:
                payload = tool_outputs.get(tool_name)
                if not payload:
                    continue
                outputs = payload.get("outputs", {})
                inputs_used = payload.get("inputs_used", {})
                if not outputs:
                    continue

                pretty = tool_name.replace("_ec3", "").replace("_", " ").title()
                lines.append(f"<details><summary><strong>{pretty}</strong> — detailed results</summary>\n")

                if inputs_used:
                    lines.append("| Input | Value |")
                    lines.append("|:------|------:|")
                    for k, v in inputs_used.items():
                        lines.append(f"| {self._pretty_key(k)} | {self._format_value(k, v)} |")
                    lines.append("")

                lines.append("| Output | Value |")
                lines.append("|:-------|------:|")
                for k, v in outputs.items():
                    if isinstance(v, (int, float, str, bool)):
                        lines.append(f"| {self._pretty_key(k)} | **{self._format_value(k, v)}** |")
                lines.append("")

                notes = payload.get("notes", [])
                if notes:
                    for n in notes:
                        lines.append(f"> {n}")
                    lines.append("")

                lines.append("</details>\n")

        if assumptions:
            lines.append("<details><summary>Assumptions made</summary>\n")
            for a in assumptions:
                lines.append(f"- {a}")
            lines.append("\n</details>\n")

        if any(s.status == "error" for s in tool_trace):
            lines.append("\n**Tool Errors:**\n")
            for s in tool_trace:
                if s.status == "error":
                    lines.append(f"- {s.tool_name}: {s.error}")

        if sources:
            filtered = self._select_relevant_sources(
                narrative=narrative, sources=sources,
                retrieved=retrieved, tool_outputs=tool_outputs,
            )
            if not filtered:
                filtered = sources
            relevant = [s for s in filtered if s.clause_id and s.clause_id != "0" and s.clause_title and s.clause_title != "text"]
            if relevant:
                lines.append("\n---\n")
                lines.append("**References:**")
                seen_refs: set[str] = set()
                ref_idx = 0
                for s in relevant[:5]:
                    ref_key = self._normalize_clause_id(s.clause_id)
                    if ref_key in seen_refs:
                        continue
                    seen_refs.add(ref_key)
                    ref_idx += 1
                    lines.append(f"{ref_idx}. EN 1993-1-1, Cl. {s.clause_id} — {s.clause_title}")

        return self._format_subscripts("\n".join(lines).strip())

    def _format_value(self, key: str, val: Any) -> str:
        if isinstance(val, bool):
            return "PASS ✓" if val else "FAIL ✗"
        if isinstance(val, float):
            unit = self._guess_unit(key)
            if val == int(val) and abs(val) < 1e6:
                return f"{int(val)} {unit}".strip()
            return f"{val:.2f} {unit}".strip()
        if isinstance(val, int):
            unit = self._guess_unit(key)
            return f"{val} {unit}".strip()
        return str(val)

    def _pretty_key(self, key: str) -> str:
        unit = ""
        base = key
        for suffix, label in _UNIT_SUFFIXES:
            if base.endswith(suffix):
                unit = label
                base = base[: -len(suffix)]
                break

        sorted_subs = sorted(_KEY_SUBSCRIPTS.items(), key=lambda x: -len(x[0]))
        for pattern, replacement in sorted_subs:
            if base == pattern:
                return replacement + unit
            if base.startswith(pattern + "_"):
                rest = base[len(pattern) + 1 :].replace("_", " ")
                return f"{replacement} {rest}" + unit

        return base.replace("_", " ") + unit

    @staticmethod
    def _format_subscripts(text: str) -> str:
        for pattern, replacement in _NARRATIVE_SUB_RE:
            text = pattern.sub(replacement, text)
        return text

    def _guess_unit(self, key: str) -> str:
        key_lower = key.lower()
        if "knm" in key_lower: return "kNm"
        if "kn" in key_lower: return "kN"
        if "mpa" in key_lower: return "MPa"
        if "gpa" in key_lower: return "GPa"
        if "_mm" in key_lower: return "mm"
        if key_lower.endswith("_m"): return "m"
        if "cm2" in key_lower: return "cm²"
        if "cm3" in key_lower: return "cm³"
        if "cm4" in key_lower: return "cm⁴"
        return ""

    @staticmethod
    def _normalize_clause_id(clause_id: str) -> str:
        idx = clause_id.find("(")
        return clause_id[:idx] if idx > 0 else clause_id

    def _select_relevant_sources(
        self,
        *,
        narrative: str,
        sources: list[Citation],
        retrieved: list[RetrievedClause],
        tool_outputs: dict[str, dict[str, Any]],
    ) -> list[Citation]:
        inline_ids: set[str] = set()
        for match in re.finditer(r"Cl\.\s*([\d.]+)", narrative):
            inline_ids.add(match.group(1))

        tool_ids: set[str] = set()
        for payload in tool_outputs.values():
            for ref in payload.get("clause_references", []):
                cid = str(ref.get("clause_id", ""))
                if cid:
                    tool_ids.add(self._normalize_clause_id(cid))

        relevant_ids = inline_ids | tool_ids
        return [s for s in sources if self._normalize_clause_id(s.clause_id) in relevant_ids]

    def _build_what_i_used(self, plan: PlanResult, retrieval_trace: list[dict[str, object]], tool_trace: list[ToolTraceStep]) -> list[str]:
        summaries = [f"Plan: {plan.mode} — {plan.rationale}", f"Retrieval: {len(retrieval_trace)} search pass(es)"]
        if tool_trace:
            chain = " → ".join(s.tool_name for s in tool_trace)
            summaries.append(f"Tool chain: {chain}")
        return summaries
