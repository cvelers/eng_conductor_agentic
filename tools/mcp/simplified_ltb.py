from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli
from tools.mcp.section_library import steel_grade_to_fy
from tools.mcp.clause_ref import clause_ref

TOOL_NAME = "simplified_ltb_ec3"


class SimplifiedLtbInput(BaseModel):
    """Input for §6.3.2.4 – Simplified assessment of beams with discrete lateral restraints."""

    steel_grade: str = Field(default="S355", description="Steel grade")
    fy_mpa: Optional[PositiveFloat] = Field(default=None, description="Yield strength in MPa")
    gamma_M1: PositiveFloat = Field(default=1.0, description="Partial factor γ_M1")

    # Restraint spacing
    L_c_mm: PositiveFloat = Field(
        description="Distance between lateral restraints to the compression flange in mm"
    )

    # Section properties for equivalent compression flange
    i_fz_mm: PositiveFloat = Field(
        description="Radius of gyration of the equivalent compression flange i_f,z in mm"
    )

    # Moment
    M_y_Ed_kNm: PositiveFloat = Field(description="Maximum design bending moment M_y,Ed in kNm")

    # Section modulus
    W_y_cm3: PositiveFloat = Field(
        description="Appropriate section modulus W_y for compression flange in cm³"
    )

    # Moment distribution factor
    k_c: PositiveFloat = Field(
        default=1.0,
        description="Slenderness correction factor k_c for moment distribution (Table 6.6)",
    )

    # Slenderness limit
    lambda_c0: Optional[float] = Field(
        default=None,
        description="Slenderness limit λ̄_c0 (recommended: λ̄_LT,0 + 0.1, default 0.5)",
    )


def calculate(inp: SimplifiedLtbInput) -> dict:
    fy = float(inp.fy_mpa) if inp.fy_mpa else steel_grade_to_fy(inp.steel_grade)
    gM1 = float(inp.gamma_M1)
    E = 210000.0
    epsilon = math.sqrt(235.0 / fy)
    lambda_1 = 93.9 * epsilon

    L_c = float(inp.L_c_mm)
    i_fz = float(inp.i_fz_mm)
    k_c = float(inp.k_c)
    M_y_Ed = float(inp.M_y_Ed_kNm)
    W_y = float(inp.W_y_cm3) * 1000.0  # mm³

    l_c0 = float(inp.lambda_c0) if inp.lambda_c0 is not None else 0.5

    notes: list[str] = [
        f"ε = √(235/{fy:.0f}) = {epsilon:.4f}",
        f"λ_1 = 93.9·ε = {lambda_1:.2f}",
    ]

    # §6.3.2.4(1)B – Slenderness of equivalent compression flange
    lambda_f = k_c * L_c / (i_fz * lambda_1)

    # M_c,Rd
    M_c_Rd = W_y * fy / (gM1 * 1e6)  # kNm

    # Check
    limit = l_c0 * M_c_Rd / M_y_Ed
    is_stable = lambda_f <= limit

    notes.append(f"λ̄_f = k_c·L_c/(i_f,z·λ_1) = {k_c:.3f}·{L_c:.1f}/({i_fz:.2f}·{lambda_1:.2f}) = {lambda_f:.4f}")
    notes.append(f"M_c,Rd = W_y·fy/γ_M1 = {M_c_Rd:.2f} kNm")
    notes.append(f"λ̄_c0·M_c,Rd/M_y,Ed = {l_c0:.3f}·{M_c_Rd:.2f}/{M_y_Ed:.2f} = {limit:.4f}")
    notes.append(f"λ̄_f = {lambda_f:.4f} {'≤' if is_stable else '>'} {limit:.4f} → {'STABLE (no LTB)' if is_stable else 'LTB CHECK REQUIRED'}")

    return {
        "inputs_used": {
            "steel_grade": inp.steel_grade,
            "fy_mpa": fy,
            "gamma_M1": gM1,
            "L_c_mm": L_c,
            "i_fz_mm": i_fz,
            "k_c": k_c,
            "M_y_Ed_kNm": M_y_Ed,
            "W_y_cm3": float(inp.W_y_cm3),
        },
        "intermediate": {
            "epsilon": round(epsilon, 4),
            "lambda_1": round(lambda_1, 2),
            "M_c_Rd_kNm": round(M_c_Rd, 2),
        },
        "outputs": {
            "lambda_f": round(lambda_f, 4),
            "lambda_c0": round(l_c0, 3),
            "limit": round(limit, 4),
            "is_stable": is_stable,
            "M_c_Rd_kNm": round(M_c_Rd, 2),
        },
        "clause_references": [
            clause_ref("ec3.en1993-1-1.2005", "6.3.2.4", "Simplified assessment methods for beams with restraints in buildings"),
        ],
        "notes": notes,
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=SimplifiedLtbInput, handler=calculate)
