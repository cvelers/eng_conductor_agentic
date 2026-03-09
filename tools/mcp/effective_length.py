from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli
from tools.mcp.clause_ref import clause_ref

TOOL_NAME = "effective_length_ec3"

SUPPORT_CONDITIONS = {
    "pinned-pinned": {"k": 1.0, "description": "Both ends pinned (no rotational restraint)"},
    "fixed-pinned": {"k": 0.7, "description": "One end fixed, one end pinned"},
    "fixed-fixed": {"k": 0.5, "description": "Both ends fixed (full rotational restraint)"},
    "fixed-free": {"k": 2.0, "description": "One end fixed, one end free (cantilever)"},
    "fixed-slide": {"k": 1.0, "description": "One end fixed, one end free to slide (sway)"},
    "pinned-slide": {"k": 2.0, "description": "One end pinned, one end free to slide (sway frame)"},
}


class EffectiveLengthInput(BaseModel):
    support_conditions: Literal[
        "pinned-pinned", "fixed-pinned", "fixed-fixed", "fixed-free", "fixed-slide", "pinned-slide"
    ] = Field(description="End restraint conditions for the member")
    system_length_m: PositiveFloat = Field(description="Physical (system) length of the member in metres")


def calculate(inp: EffectiveLengthInput) -> dict:
    cond = SUPPORT_CONDITIONS[inp.support_conditions]
    k = cond["k"]
    L_cr = k * inp.system_length_m

    return {
        "inputs_used": {
            "support_conditions": inp.support_conditions,
            "system_length_m": inp.system_length_m,
        },
        "outputs": {
            "k_factor": k,
            "L_cr_m": round(L_cr, 4),
            "support_description": cond["description"],
        },
        "clause_references": [
            clause_ref("ec3.en1993-1-1.2005", "BB.1", "Effective buckling length for members in compression"),
        ],
        "notes": [
            f"Buckling length factor k = {k} for {cond['description']}.",
            f"L_cr = k × L = {k} × {inp.system_length_m} = {L_cr:.4f} m",
            "Actual k values may differ for semi-rigid connections; see EC3-1-1 Annex BB for frame analysis.",
        ],
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=EffectiveLengthInput, handler=calculate)
