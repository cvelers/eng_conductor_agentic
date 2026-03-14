"""Routing for FEA-capable requests via an LLM classifier."""

from __future__ import annotations

import logging
import re
from typing import Any

from backend.llm.base import LLMProvider
from backend.utils.json_utils import parse_json_loose

logger = logging.getLogger(__name__)

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
- continuation of a recent conversation that is clearly about structural modeling, solving, or viewer results

Route to `chat` when the user wants any of these:
- Eurocode clause lookup or interpretation
- member resistance, connection design, buckling, LTB, section classification, material or profile data
- conceptual explanation or hand calculations that do not require a global structural model
- short follow-up questions that continue a member-level design check or code discussion from the prior chat turn

Important follow-up rule:
- For short follow-ups such as "what about ltb", "what clause applies", or "show the equation", inherit the topic from the recent conversation
- If the recent conversation was a member-level resistance / Eurocode / hand-calculation discussion, keep routing to `chat`
- Treat a follow-up as `fea` only when the recent conversation is clearly about a structural model, solver results, or viewer behavior
- If the current request explicitly switches into model-building or FEA work, that overrides prior member-level chat context and must route to `fea`
- Requests that explicitly ask to build, create, or analyse a structural model are `fea` even if the previous turn was a resistance check

Examples:
- "create analyze fea model 2x2 bays frame on selfweight" -> `fea`
- "analyse a portal frame and show the moment diagram" -> `fea`
- "check IPE300 bending resistance in S355 per EN 1993-1-1" -> `chat`
- "Given IPE300, S355, what is the bending resistance?" then "what about ltb" -> `chat`
- "what clause governs lateral torsional buckling?" -> `chat`

Return ONLY one lowercase word:
fea
or
chat
"""

_ROUTE_TOKEN_RE = re.compile(r"\b(fea|chat)\b", re.IGNORECASE)
_MALFORMED_ROUTE_RE = re.compile(r'["\']?route["\']?\s*:\s*["\']?(fea|chat)', re.IGNORECASE)


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


def _build_history_excerpt(history: list[Any] | None, limit: int = 6) -> str:
    lines: list[str] = []
    for item in (history or [])[-limit:]:
        row = _history_row(item)
        role = str(row.get("role", "user") or "user").upper()
        content = str(row.get("content", "") or "").strip()
        if not content:
            continue
        snippet = content.replace("\n", " ").strip()
        if len(snippet) > 400:
            snippet = snippet[:399].rstrip() + "…"
        lines.append(f"{role}: {snippet or '(no visible content)'}")
    return "\n".join(lines) if lines else "(none)"


def _recover_route_from_text(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return "chat"

    first_line = raw.splitlines()[0].strip().strip("` ").lower()
    if first_line in {"fea", "chat"}:
        return first_line

    malformed_match = _MALFORMED_ROUTE_RE.search(raw)
    if malformed_match:
        return malformed_match.group(1).lower()

    token_match = _ROUTE_TOKEN_RE.match(first_line)
    if token_match:
        return token_match.group(1).lower()

    return "chat"


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
        max_tokens=32,
        reasoning_effort="",
    )
    text = str(raw or "").strip()
    direct_route = _recover_route_from_text(text)
    if direct_route in {"fea", "chat"} and text.splitlines()[:1]:
        first_line = text.splitlines()[0].strip().strip("` ").lower()
        if first_line in {"fea", "chat"}:
            return {"route": direct_route, "reason": ""}

    try:
        data = parse_json_loose(text)
    except Exception:
        if direct_route != "chat" or _MALFORMED_ROUTE_RE.search(text):
            logger.warning(
                "fea_route_classifier_recovered_from_malformed_output",
                extra={"raw": text[:300], "route": direct_route},
            )
            return {
                "route": direct_route,
                "reason": "Recovered from malformed classifier output.",
            }
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
    decision = classify_fea_route(llm, message, history)
    logger.info("fea_route_decision", extra={"route": decision["route"], "reason": decision["reason"]})
    return decision["route"] == "fea"
