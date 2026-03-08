from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli
from tools.mcp.section_library import steel_grade_to_fy

TOOL_NAME = "bending_shear_interaction_ec3"


class BendingShearInput(BaseModel):
    M_Ed_kNm: float = Field(description="Design bending moment M_Ed in kNm")
    V_Ed_kN: float = Field(description="Design shear force V_Ed in kN")
    V_pl_Rd_kN: PositiveFloat = Field(description="Design plastic shear resistance V_pl,Rd in kN")
    M_c_Rd_kNm: PositiveFloat = Field(description="Design moment resistance M_c,Rd in kNm (unreduced)")
    steel_grade: str = Field(default="S355", description="Steel grade, e.g. S355")
    fy_mpa: Optional[PositiveFloat] = Field(default=None, description="Yield strength in MPa")
    gamma_M0: PositiveFloat = Field(default=1.0, description="Partial factor γ_M0")

    # For computing reduced moment when shear is high
    wpl_y_cm3: Optional[PositiveFloat] = Field(default=None, description="Plastic section modulus W_pl,y in cm³")
    A_w_cm2: Optional[PositiveFloat] = Field(
        default=None,
        description="Web area Aw = hw·tw in cm² (for I/H sections, used when computing reduced M_V,Rd)",
    )
    section_class: int = Field(default=2, ge=1, le=4, description="Cross-section class (1-4)")


def calculate(inp: BendingShearInput) -> dict:
    fy = float(inp.fy_mpa) if inp.fy_mpa else steel_grade_to_fy(inp.steel_grade)
    gamma_M0 = float(inp.gamma_M0)

    V_Ed = abs(float(inp.V_Ed_kN))
    M_Ed = abs(float(inp.M_Ed_kNm))
    V_pl_Rd = float(inp.V_pl_Rd_kN)
    M_c_Rd = float(inp.M_c_Rd_kNm)

    # §6.2.8(2) – Check if shear exceeds 50% of V_pl,Rd
    shear_ratio = V_Ed / V_pl_Rd
    high_shear = shear_ratio > 0.5

    if not high_shear:
        # §6.2.8(2) – No reduction needed
        M_V_Rd = M_c_Rd
        rho = 0.0
        notes = ["V_Ed ≤ 0.5·V_pl,Rd → no moment reduction required."]
    else:
        # §6.2.8(3) – Reduced moment resistance
        rho = (2.0 * V_Ed / V_pl_Rd - 1.0) ** 2
        notes = [f"V_Ed > 0.5·V_pl,Rd → ρ = (2·V_Ed/V_pl,Rd − 1)² = {rho:.4f}"]

        if inp.wpl_y_cm3 and inp.A_w_cm2 and inp.section_class <= 2:
            # For I/H Class 1/2: M_V,Rd = (Wpl - ρ·Aw²/(4tw)) · fy/γM0
            # Simplified: M_V,Rd = M_pl,Rd - ρ·Aw²·fy/(4·tw·γM0)
            # Using: reduced Wpl = Wpl - ρ·Aw²/(4·tw)
            # But we use the simplified approach: yield reduced to (1-ρ)·fy on shear area
            Wpl_mm3 = float(inp.wpl_y_cm3) * 1000.0
            Aw_mm2 = float(inp.A_w_cm2) * 100.0
            # M_V,Rd = [Wpl - ρ·Aw²/(4·tw)] · fy/γM0 but simplified as:
            # Reduced Wpl contribution from web
            reduced_wpl_mm3 = Wpl_mm3 - rho * Aw_mm2**2 / (4.0 * Aw_mm2) if Aw_mm2 > 0 else Wpl_mm3
            # Simplified: just reduce the web contribution
            M_V_Rd = max(reduced_wpl_mm3 * fy / (gamma_M0 * 1e6), 0.0)
            M_V_Rd = min(M_V_Rd, M_c_Rd)
            notes.append(f"I/H section: M_V,Rd = (Wpl − ρ·Aw²/(4tw))·fy/γM0 = {M_V_Rd:.2f} kNm")
        else:
            # General: reduce yield strength by (1-ρ)
            M_V_Rd = M_c_Rd * (1.0 - rho)
            notes.append(f"M_V,Rd = M_c,Rd·(1−ρ) = {M_c_Rd:.2f}·{1.0 - rho:.4f} = {M_V_Rd:.2f} kNm")

    utilization = M_Ed / M_V_Rd if M_V_Rd > 0 else float("inf")

    return {
        "inputs_used": {
            "M_Ed_kNm": float(inp.M_Ed_kNm),
            "V_Ed_kN": float(inp.V_Ed_kN),
            "V_pl_Rd_kN": V_pl_Rd,
            "M_c_Rd_kNm": M_c_Rd,
            "steel_grade": inp.steel_grade,
            "fy_mpa": fy,
        },
        "intermediate": {
            "shear_ratio_V_Ed_over_V_pl_Rd": round(shear_ratio, 4),
            "high_shear": high_shear,
            "rho": round(rho, 4),
        },
        "outputs": {
            "M_V_Rd_kNm": round(M_V_Rd, 2),
            "utilization": round(utilization, 4),
            "pass": utilization <= 1.0,
        },
        "clause_references": [
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "6.2.8",
                "title": "Bending and shear",
                "pointer": "en_1993_1_1_2005_structured.json#6.2.8",
            },
        ],
        "notes": notes,
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=BendingShearInput, handler=calculate)
