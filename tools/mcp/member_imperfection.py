from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli
from tools.mcp.section_library import steel_grade_to_fy

TOOL_NAME = "member_imperfection_ec3"

# Table 5.1 – Design values of initial bow imperfection e_0/L
BOW_IMPERFECTIONS = {
    "elastic": {
        "a0": 1.0 / 350.0,
        "a": 1.0 / 300.0,
        "b": 1.0 / 250.0,
        "c": 1.0 / 200.0,
        "d": 1.0 / 150.0,
    },
    "plastic": {
        "a0": 1.0 / 300.0,
        "a": 1.0 / 250.0,
        "b": 1.0 / 200.0,
        "c": 1.0 / 150.0,
        "d": 1.0 / 100.0,
    },
}


class MemberImperfectionInput(BaseModel):
    """Input for §5.3.4 – Member imperfections (bow imperfections for second-order analysis)."""

    buckling_curve: Literal["a0", "a", "b", "c", "d"] = Field(
        description="Buckling curve from Table 6.2"
    )
    analysis_method: Literal["elastic", "plastic"] = Field(
        default="elastic",
        description="Analysis method: 'elastic' or 'plastic' (Table 5.1 column)",
    )
    L_mm: PositiveFloat = Field(description="Member length L in mm")

    # For LTB imperfection (§5.3.4(3))
    include_ltb: bool = Field(
        default=False,
        description="If True, also compute LTB imperfection as k·e_0,d (§5.3.4(3))",
    )
    k_ltb: float = Field(
        default=0.5,
        description="Factor k for LTB imperfection (recommended 0.5, NA may specify)",
    )


def calculate(inp: MemberImperfectionInput) -> dict:
    L = float(inp.L_mm)
    curve = inp.buckling_curve
    method = inp.analysis_method

    notes: list[str] = []

    e0_ratio = BOW_IMPERFECTIONS[method][curve]
    e_0 = e0_ratio * L

    notes.append(f"Table 5.1 ({method}): e_0/L = {e0_ratio:.6f} (1/{1.0 / e0_ratio:.0f})")
    notes.append(f"e_0 = {e0_ratio:.6f} × {L:.1f} = {e_0:.2f} mm")

    outputs: dict = {
        "e0_over_L": round(e0_ratio, 6),
        "e0_over_L_inv": round(1.0 / e0_ratio, 0),
        "e_0_mm": round(e_0, 2),
    }

    if inp.include_ltb:
        k = float(inp.k_ltb)
        e_ltb = k * e_0
        outputs["e_ltb_mm"] = round(e_ltb, 2)
        outputs["k_ltb"] = k
        notes.append(f"LTB imperfection: k·e_0 = {k}·{e_0:.2f} = {e_ltb:.2f} mm")

    return {
        "inputs_used": {
            "buckling_curve": curve,
            "analysis_method": method,
            "L_mm": L,
        },
        "outputs": outputs,
        "clause_references": [
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "5.3.4",
                "title": "Member imperfections",
                "pointer": "en_1993_1_1_2005_structured.json#5.3.4",
            },
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "5.3.2",
                "title": "Table 5.1 – Design values of initial bow imperfection e_0/L",
                "pointer": "en_1993_1_1_2005_structured.json#5.3.2",
            },
        ],
        "notes": notes,
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=MemberImperfectionInput, handler=calculate)
