from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli
from tools.mcp.section_library import steel_grade_to_fy
from tools.mcp.clause_ref import clause_ref

TOOL_NAME = "bending_shear_axial_ec3"


class BendingShearAxialInput(BaseModel):
    M_Ed_kNm: float = Field(description="Design bending moment M_Ed in kNm")
    V_Ed_kN: float = Field(description="Design shear force V_Ed in kN")
    N_Ed_kN: float = Field(description="Design axial force N_Ed in kN")
    V_pl_Rd_kN: PositiveFloat = Field(description="Plastic shear resistance V_pl,Rd in kN")
    M_N_Rd_kNm: PositiveFloat = Field(description="Reduced moment resistance M_N,Rd from §6.2.9 in kNm")

    steel_grade: str = Field(default="S355", description="Steel grade, e.g. S355")
    fy_mpa: Optional[PositiveFloat] = Field(default=None, description="Yield strength in MPa")
    gamma_M0: PositiveFloat = Field(default=1.0, description="Partial factor γ_M0")


def calculate(inp: BendingShearAxialInput) -> dict:
    fy = float(inp.fy_mpa) if inp.fy_mpa else steel_grade_to_fy(inp.steel_grade)

    V_Ed = abs(float(inp.V_Ed_kN))
    M_Ed = abs(float(inp.M_Ed_kNm))
    V_pl_Rd = float(inp.V_pl_Rd_kN)
    M_N_Rd = float(inp.M_N_Rd_kNm)

    shear_ratio = V_Ed / V_pl_Rd
    high_shear = shear_ratio > 0.5

    notes: list[str] = []

    if not high_shear:
        # §6.2.10(2) – When V_Ed ≤ 0.5·V_pl,Rd, no additional reduction
        M_V_N_Rd = M_N_Rd
        rho = 0.0
        notes.append("V_Ed ≤ 0.5·V_pl,Rd → M_N,Rd from §6.2.9 applies directly.")
    else:
        # §6.2.10(3) – Reduce yield to (1-ρ)·fy on shear area
        rho = (2.0 * V_Ed / V_pl_Rd - 1.0) ** 2
        # Apply same ρ reduction to M_N_Rd
        M_V_N_Rd = M_N_Rd * (1.0 - rho)
        notes.append(
            f"V_Ed > 0.5·V_pl,Rd → ρ = (2·V_Ed/V_pl,Rd − 1)² = {rho:.4f}"
        )
        notes.append(
            f"M_V_N,Rd = M_N,Rd·(1−ρ) = {M_N_Rd:.2f}·{1.0 - rho:.4f} = {M_V_N_Rd:.2f} kNm"
        )

    utilization = M_Ed / M_V_N_Rd if M_V_N_Rd > 0 else float("inf")

    return {
        "inputs_used": {
            "M_Ed_kNm": float(inp.M_Ed_kNm),
            "V_Ed_kN": float(inp.V_Ed_kN),
            "N_Ed_kN": float(inp.N_Ed_kN),
            "V_pl_Rd_kN": V_pl_Rd,
            "M_N_Rd_kNm": M_N_Rd,
            "steel_grade": inp.steel_grade,
            "fy_mpa": fy,
        },
        "intermediate": {
            "shear_ratio": round(shear_ratio, 4),
            "high_shear": high_shear,
            "rho": round(rho, 4),
        },
        "outputs": {
            "M_V_N_Rd_kNm": round(M_V_N_Rd, 2),
            "utilization": round(utilization, 4),
            "pass": utilization <= 1.0,
        },
        "clause_references": [
            clause_ref("ec3.en1993-1-1.2005", "6.2.10", "Bending, shear and axial force"),
        ],
        "notes": notes,
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=BendingShearAxialInput, handler=calculate)
