"""Routing for FEA-capable requests.

Uses a small explicit override for unambiguous FEA commands and falls back to an
LLM classifier for the ambiguous middle ground.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.llm.base import LLMProvider
from backend.utils.json_utils import parse_json_loose

logger = logging.getLogger(__name__)

_EXPLICIT_FEA_CUES = (
    " fea ",
    " fem ",
    "finite element",
    "deformed shape",
    "moment diagram",
    "shear diagram",
    "axial diagram",
    "structural model",
)

FEA_ROUTER_SYSTEM = """\
You are a routing classifier for a structural engineering assistant.

Decide whether the user's request should go to:
- `fea`: the browser-based finite element analysis sub-agent that builds a structural model, runs a solver, and drives a 3D viewer
- `chat`: the general engineering orchestrator for Eurocode lookup, design checks, hand calculations, and discussion

Route to `fea` when the user wants any of these:
- build, create, or analyze a structural model of a beam, frame, truss, portal frame, grillage, or 3D structure
- global analysis results such as reactions, displacements, drift, internal forces, moment/shear/axial diagrams
- self-weight, nodal loads, distributed loads, support conditions, geometry for a structural model
- visual/model actions tied to the FEA viewer such as show model, deformed shape, moment diagram, shear diagram, axial diagram
- continuation of an existing FEA thread

Route to `chat` when the user wants any of these:
- Eurocode clause lookup or interpretation
- member resistance, connection design, buckling, LTB, section classification, material or profile data
- conceptual explanation or hand calculations that do not require a global structural model

Examples:
- "create analyze fea model 2x2 bays frame on selfweight" -> `fea`
- "analyse a portal frame and show the moment diagram" -> `fea`
- "check IPE300 bending resistance in S355 per EN 1993-1-1" -> `chat`
- "what clause governs lateral torsional buckling?" -> `chat`

Return JSON only:
{"route":"fea"|"chat","reason":"one sentence"}
"""


def _history_row(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        raw = item.model_dump()
        return raw if isinstance(raw, dict) else {}
    if isinstance(item, dict):
        return dict(item)
    return {
        "role": str(getattr(item, "role", "")),
        "content": str(getattr(item, "content", "")),
        "response_payload": getattr(item, "response_payload", None),
    }


def _looks_like_prior_fea_turn(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if isinstance(payload.get("assumptions"), list):
        return True
    trace = payload.get("tool_trace")
    if isinstance(trace, list):
        for step in trace:
            if not isinstance(step, dict):
                continue
            tool_name = str(step.get("tool_name", "") or "")
            if tool_name.startswith("fea_") or tool_name in {"ask_user", "todo_write"}:
                return True
    answer = str(payload.get("answer", "") or "")
    return "FEA analysis complete" in answer


def _explicit_fea_override(message: str, history: list[Any] | None = None) -> dict[str, str] | None:
    lower = f" {str(message or '').strip().lower()} "
    if any(_looks_like_prior_fea_turn(_history_row(item).get("response_payload")) for item in (history or [])):
        return {
            "route": "fea",
            "reason": "Continuing an existing FEA thread.",
        }
    if any(cue in lower for cue in _EXPLICIT_FEA_CUES):
        return {
            "route": "fea",
            "reason": "The user explicitly requested FEA/global model analysis.",
        }
    return None


def _build_history_excerpt(history: list[Any] | None, limit: int = 6) -> str:
    lines: list[str] = []
    for item in (history or [])[-limit:]:
        row = _history_row(item)
        role = str(row.get("role", "user") or "user").upper()
        content = str(row.get("content", "") or "").strip()
        payload = row.get("response_payload")
        suffix = " [prior_fea_turn]" if _looks_like_prior_fea_turn(payload) else ""
        if not content and not suffix:
            continue
        snippet = content.replace("\n", " ").strip()
        if len(snippet) > 400:
            snippet = snippet[:399].rstrip() + "…"
        lines.append(f"{role}{suffix}: {snippet or '(no visible content)'}")
    return "\n".join(lines) if lines else "(none)"


def classify_fea_route(
    llm: LLMProvider,
    message: str,
    history: list[Any] | None = None,
) -> dict[str, str]:
    """Return an explicit route decision for the current request."""
    history_excerpt = _build_history_excerpt(history)
    user_prompt = (
        "Recent conversation:\n"
        f"{history_excerpt}\n\n"
        "Current user request:\n"
        f"{message.strip()}\n"
    )
    raw = llm.generate(
        system_prompt=FEA_ROUTER_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=120,
        reasoning_effort=None,
    )
    text = str(raw or "").strip()
    try:
        data = parse_json_loose(text)
    except Exception:
        logger.warning("fea_route_classifier_parse_failed", extra={"raw": text[:300]})
        return {"route": "chat", "reason": "Classifier output was malformed."}
    if not isinstance(data, dict):
        return {"route": "chat", "reason": "Classifier returned a non-object response."}
    route = str(data.get("route", "chat") or "chat").strip().lower()
    if route not in {"fea", "chat"}:
        route = "chat"
    reason = str(data.get("reason", "") or "").strip()
    return {"route": route, "reason": reason}


def should_route_to_fea(
    llm: LLMProvider,
    message: str,
    history: list[Any] | None = None,
) -> bool:
    override = _explicit_fea_override(message, history)
    decision = override or classify_fea_route(llm, message, history)
    logger.info("fea_route_decision", extra={"route": decision["route"], "reason": decision["reason"]})
    return decision["route"] == "fea"
