from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli

TOOL_NAME = "weld_resistance_ec3"

BETA_W = {
    "S235": 0.8,
    "S275": 0.85,
    "S355": 0.9,
    "S420": 1.0,
    "S460": 1.0,
}

FU_VALUES = {
    "S235": 360,
    "S275": 430,
    "S355": 490,
    "S420": 520,
    "S460": 540,
}


class WeldResistanceInput(BaseModel):
    throat_thickness_mm: PositiveFloat = Field(description="Effective throat thickness 'a' in mm")
    weld_length_mm: PositiveFloat = Field(description="Effective weld length in mm")
    steel_grade: str = Field(default="S355", description="Steel grade of connected parts")
    gamma_M2: PositiveFloat = Field(default=1.25, description="Partial safety factor γ_M2")


def calculate(inp: WeldResistanceInput) -> dict:
    grade = inp.steel_grade.strip().upper()
    if grade not in BETA_W:
        raise ValueError(f"Unsupported steel grade '{inp.steel_grade}'. Available: {', '.join(sorted(BETA_W.keys()))}")

    beta_w = BETA_W[grade]
    fu = FU_VALUES[grade]

    fvw_d = fu / (math.sqrt(3) * beta_w * inp.gamma_M2)  # N/mm²

    Fw_Rd_per_mm = fvw_d * inp.throat_thickness_mm / 1000  # kN/mm
    Fw_Rd = Fw_Rd_per_mm * inp.weld_length_mm  # kN

    return {
        "inputs_used": {
            "throat_thickness_mm": inp.throat_thickness_mm,
            "weld_length_mm": inp.weld_length_mm,
            "steel_grade": grade,
            "gamma_M2": inp.gamma_M2,
        },
        "intermediate": {
            "fu_mpa": fu,
            "beta_w": beta_w,
            "fvw_d_mpa": round(fvw_d, 2),
        },
        "outputs": {
            "Fw_Rd_kN": round(Fw_Rd, 2),
            "Fw_Rd_per_mm_kN": round(Fw_Rd_per_mm, 4),
            "fvw_d_mpa": round(fvw_d, 2),
        },
        "clause_references": [
            {
                "doc_id": "ec3.en1993-1-8.2005",
                "clause_id": "4.5.3.3",
                "title": "Simplified method for design resistance of fillet weld",
                "pointer": "en_1993_1_8#4.5.3.3",
            },
        ],
        "notes": [
            f"fvw,d = fu / (√3 × βw × γM2) = {fu} / (√3 × {beta_w} × {inp.gamma_M2}) = {fvw_d:.2f} N/mm²",
            f"Fw,Rd = a × Lw × fvw,d = {inp.throat_thickness_mm} × {inp.weld_length_mm} × {fvw_d:.2f} / 1000 = {Fw_Rd:.2f} kN",
            "This uses the simplified method. Directional method (4.5.3.2) may give higher resistance.",
        ],
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=WeldResistanceInput, handler=calculate)
