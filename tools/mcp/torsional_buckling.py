from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli
from tools.mcp.section_library import steel_grade_to_fy
from tools.mcp.clause_ref import clause_ref

TOOL_NAME = "torsional_buckling_ec3"

IMPERFECTION_FACTORS = {
    "a0": 0.13,
    "a": 0.21,
    "b": 0.34,
    "c": 0.49,
    "d": 0.76,
}


class TorsionalBucklingInput(BaseModel):
    """Input for §6.3.1.4 – Torsional and torsional-flexural buckling."""

    steel_grade: str = Field(default="S355", description="Steel grade, e.g. S355")
    fy_mpa: Optional[PositiveFloat] = Field(default=None, description="Yield strength in MPa")
    gamma_M1: PositiveFloat = Field(default=1.0, description="Partial factor γ_M1")
    section_class: int = Field(default=2, ge=1, le=4, description="Cross-section class")

    area_cm2: PositiveFloat = Field(description="Gross area A in cm²")
    A_eff_cm2: Optional[PositiveFloat] = Field(default=None, description="Effective area A_eff in cm² (Class 4)")

    # Elastic critical force for torsional/torsional-flexural mode
    N_cr_T_kN: Optional[PositiveFloat] = Field(
        default=None,
        description="Elastic critical force for torsional buckling N_cr,T in kN",
    )
    N_cr_TF_kN: Optional[PositiveFloat] = Field(
        default=None,
        description="Elastic critical force for torsional-flexural buckling N_cr,TF in kN",
    )

    # Or provide section properties to calculate N_cr,T
    L_cr_m: Optional[PositiveFloat] = Field(default=None, description="Effective length for torsional buckling in m")
    I_z_cm4: Optional[PositiveFloat] = Field(default=None, description="Second moment about z-z in cm⁴")
    I_y_cm4: Optional[PositiveFloat] = Field(default=None, description="Second moment about y-y in cm⁴")
    I_T_cm4: Optional[PositiveFloat] = Field(default=None, description="St. Venant torsion constant in cm⁴")
    I_w_cm6: Optional[float] = Field(default=None, description="Warping constant in cm⁶")

    buckling_curve: Literal["a0", "a", "b", "c", "d"] = Field(
        default="c", description="Buckling curve (typically 'c' for torsional buckling)"
    )


def calculate(inp: TorsionalBucklingInput) -> dict:
    fy = float(inp.fy_mpa) if inp.fy_mpa else steel_grade_to_fy(inp.steel_grade)
    E = 210000.0  # MPa
    G = 81000.0  # MPa
    gamma_M1 = float(inp.gamma_M1)

    A_mm2 = float(inp.area_cm2) * 100.0

    # Determine N_cr (lowest of torsional and torsional-flexural)
    N_cr_kN: float
    mode = "torsional"

    if inp.N_cr_TF_kN is not None:
        N_cr_kN = float(inp.N_cr_TF_kN)
        mode = "torsional-flexural"
    elif inp.N_cr_T_kN is not None:
        N_cr_kN = float(inp.N_cr_T_kN)
    elif inp.L_cr_m and inp.I_T_cm4 is not None:
        # Calculate N_cr,T = (1/is²)·(G·IT + π²·E·Iw/L²)
        L_mm = float(inp.L_cr_m) * 1000.0
        I_T_mm4 = float(inp.I_T_cm4) * 1e4
        I_w_mm6 = float(inp.I_w_cm6) * 1e6 if inp.I_w_cm6 else 0.0
        I_y_mm4 = float(inp.I_y_cm4) * 1e4 if inp.I_y_cm4 else 0.0
        I_z_mm4 = float(inp.I_z_cm4) * 1e4 if inp.I_z_cm4 else 0.0

        # is² = (Iy + Iz)/A  (for doubly symmetric sections, y0 = z0 = 0)
        i_s_sq = (I_y_mm4 + I_z_mm4) / A_mm2

        N_cr_T = (G * I_T_mm4 + math.pi**2 * E * I_w_mm6 / L_mm**2) / i_s_sq / 1000.0  # kN
        N_cr_kN = N_cr_T
    else:
        raise ValueError("Provide N_cr_T_kN or N_cr_TF_kN, or L_cr_m + I_T_cm4 + I_y_cm4 + I_z_cm4.")

    # §6.3.1.4 – Slenderness
    if inp.section_class <= 3:
        lambda_T = math.sqrt(A_mm2 * fy / (N_cr_kN * 1000.0))
    else:
        A_eff = float(inp.A_eff_cm2) * 100.0 if inp.A_eff_cm2 else A_mm2
        lambda_T = math.sqrt(A_eff * fy / (N_cr_kN * 1000.0))

    # Same buckling curve formulas as §6.3.1.2
    alpha = IMPERFECTION_FACTORS[inp.buckling_curve]
    phi = 0.5 * (1.0 + alpha * (lambda_T - 0.2) + lambda_T**2)
    disc = phi**2 - lambda_T**2
    chi = 1.0 / (phi + math.sqrt(max(disc, 0.0)))
    chi = min(chi, 1.0)

    # Buckling resistance
    if inp.section_class <= 3:
        Nb_Rd = chi * A_mm2 * fy / (gamma_M1 * 1000.0)
    else:
        A_eff = float(inp.A_eff_cm2) * 100.0 if inp.A_eff_cm2 else A_mm2
        Nb_Rd = chi * A_eff * fy / (gamma_M1 * 1000.0)

    return {
        "inputs_used": {
            "steel_grade": inp.steel_grade,
            "fy_mpa": fy,
            "gamma_M1": gamma_M1,
            "area_cm2": float(inp.area_cm2),
            "section_class": inp.section_class,
            "buckling_curve": inp.buckling_curve,
        },
        "intermediate": {
            "N_cr_kN": round(N_cr_kN, 2),
            "buckling_mode": mode,
            "lambda_T": round(lambda_T, 4),
            "alpha": alpha,
            "phi": round(phi, 4),
            "chi": round(chi, 4),
        },
        "outputs": {
            "Nb_Rd_kN": round(Nb_Rd, 2),
            "chi": round(chi, 4),
            "lambda_T": round(lambda_T, 4),
            "N_cr_kN": round(N_cr_kN, 2),
        },
        "clause_references": [
            clause_ref("ec3.en1993-1-1.2005", "6.3.1.4", "Slenderness for torsional and torsional-flexural buckling"),
        ],
        "notes": [
            f"Buckling mode: {mode}",
            {"latex": rf"N_{{cr}} = {N_cr_kN:.2f}\,\mathrm{{kN}}"},
            {"latex": rf"\bar{{\lambda}}_T = {lambda_T:.4f},\ \chi = {chi:.4f}"},
            {
                "latex": (
                    rf"N_{{b,Rd}} = \chi \cdot A \cdot f_y / \gamma_{{M1}} = {Nb_Rd:.2f}\,\mathrm{{kN}}"
                ),
            },
        ],
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=TorsionalBucklingInput, handler=calculate)
