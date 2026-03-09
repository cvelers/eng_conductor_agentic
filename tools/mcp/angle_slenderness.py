from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli
from tools.mcp.clause_ref import clause_ref

TOOL_NAME = "angle_slenderness_ec3"

IMPERFECTION_FACTORS = {
    "a0": 0.13,
    "a": 0.21,
    "b": 0.34,
    "c": 0.49,
    "d": 0.76,
}


class AngleSlendernessInput(BaseModel):
    """Input for Annex BB.1.2 – Effective slenderness for angle web members in trusses."""

    axis: Literal["v-v", "y-y"] = Field(description="Buckling axis: 'v-v' (weak) or 'y-y' (strong)")
    lambda_bar: PositiveFloat = Field(description="Non-dimensional slenderness λ̄ about the relevant axis")

    # Section and material
    steel_grade: str = Field(default="S355", description="Steel grade")
    fy_mpa: Optional[PositiveFloat] = Field(default=None, description="Yield strength in MPa")
    gamma_M1: PositiveFloat = Field(default=1.0, description="Partial factor γ_M1")
    area_cm2: PositiveFloat = Field(description="Cross-section area of angle in cm²")

    buckling_curve: Literal["a0", "a", "b", "c", "d"] = Field(
        default="b", description="Buckling curve (typically 'b' for angles in trusses)"
    )


def calculate(inp: AngleSlendernessInput) -> dict:
    from tools.mcp.section_library import steel_grade_to_fy

    fy = float(inp.fy_mpa) if inp.fy_mpa else steel_grade_to_fy(inp.steel_grade)
    gamma_M1 = float(inp.gamma_M1)
    lb = float(inp.lambda_bar)

    # BB.1.2 – Effective slenderness
    if inp.axis == "v-v":
        lambda_eff = 0.35 + 0.7 * lb
        formula = "λ̄_eff,v = 0.35 + 0.7·λ̄_v"
    else:
        lambda_eff = 0.50 + 0.7 * lb
        formula = "λ̄_eff,y = 0.50 + 0.7·λ̄_y"

    # Apply standard buckling curve calculation with effective slenderness
    alpha = IMPERFECTION_FACTORS[inp.buckling_curve]
    phi = 0.5 * (1.0 + alpha * (lambda_eff - 0.2) + lambda_eff**2)
    disc = phi**2 - lambda_eff**2
    chi = 1.0 / (phi + math.sqrt(max(disc, 0.0)))
    chi = min(chi, 1.0)

    A_mm2 = float(inp.area_cm2) * 100.0
    Nb_Rd = chi * A_mm2 * fy / (gamma_M1 * 1000.0)  # kN

    return {
        "inputs_used": {
            "axis": inp.axis,
            "lambda_bar": lb,
            "steel_grade": inp.steel_grade,
            "fy_mpa": fy,
            "gamma_M1": gamma_M1,
            "area_cm2": float(inp.area_cm2),
            "buckling_curve": inp.buckling_curve,
        },
        "intermediate": {
            "lambda_eff": round(lambda_eff, 4),
            "alpha": alpha,
            "phi": round(phi, 4),
            "chi": round(chi, 4),
        },
        "outputs": {
            "lambda_eff": round(lambda_eff, 4),
            "chi": round(chi, 4),
            "Nb_Rd_kN": round(Nb_Rd, 2),
            "formula": formula,
        },
        "clause_references": [
            clause_ref("ec3.en1993-1-1.2005", "BB.1.2", "Angles as web members in trusses"),
        ],
        "notes": [
            f"{formula} = {lambda_eff:.4f}",
            f"χ = {chi:.4f} (curve '{inp.buckling_curve}')",
            f"N_b,Rd = χ·A·fy/γM1 = {Nb_Rd:.2f} kN",
        ],
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=AngleSlendernessInput, handler=calculate)
