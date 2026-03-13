from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli
from tools.mcp.clause_ref import clause_ref

TOOL_NAME = "combined_buckling_check_ec3"


class CombinedBucklingInput(BaseModel):
    """Input for §6.3.3 interaction check – uniform members in bending and axial compression."""

    # Design effects
    N_Ed_kN: PositiveFloat = Field(description="Design compression force N_Ed in kN")
    M_y_Ed_kNm: float = Field(default=0.0, description="Design moment about y-y M_y,Ed in kNm")
    M_z_Ed_kNm: float = Field(default=0.0, description="Design moment about z-z M_z,Ed in kNm")
    delta_M_y_kNm: float = Field(default=0.0, description="Additional moment ΔM_y from Class 4 shift in kNm")
    delta_M_z_kNm: float = Field(default=0.0, description="Additional moment ΔM_z from Class 4 shift in kNm")

    # Characteristic resistances (unfactored by γM1)
    N_Rk_kN: PositiveFloat = Field(description="Characteristic axial resistance N_Rk = A·fy (or Aeff·fy) in kN")
    M_y_Rk_kNm: PositiveFloat = Field(description="Characteristic moment resistance M_y,Rk = Wy·fy in kNm")
    M_z_Rk_kNm: PositiveFloat = Field(default=1e10, description="Characteristic moment resistance M_z,Rk = Wz·fy in kNm")
    gamma_M1: PositiveFloat = Field(default=1.0, description="Partial factor γ_M1")

    # Buckling reduction factors (from §6.3.1 and §6.3.2)
    chi_y: PositiveFloat = Field(description="Flexural buckling reduction factor about y-y χ_y")
    chi_z: PositiveFloat = Field(description="Flexural buckling reduction factor about z-z χ_z")
    chi_LT: PositiveFloat = Field(default=1.0, description="LTB reduction factor χ_LT (1.0 if no LTB)")

    # Interaction factors (Method B per Annex B, simplified)
    k_yy: PositiveFloat = Field(description="Interaction factor k_yy")
    k_yz: float = Field(default=0.0, description="Interaction factor k_yz")
    k_zy: float = Field(default=0.0, description="Interaction factor k_zy")
    k_zz: float = Field(default=0.0, description="Interaction factor k_zz")


def calculate(inp: CombinedBucklingInput) -> dict:
    gM1 = float(inp.gamma_M1)
    N_Ed = float(inp.N_Ed_kN)
    M_y = abs(float(inp.M_y_Ed_kNm)) + abs(float(inp.delta_M_y_kNm))
    M_z = abs(float(inp.M_z_Ed_kNm)) + abs(float(inp.delta_M_z_kNm))

    N_Rk = float(inp.N_Rk_kN)
    M_y_Rk = float(inp.M_y_Rk_kNm)
    M_z_Rk = float(inp.M_z_Rk_kNm)

    chi_y = float(inp.chi_y)
    chi_z = float(inp.chi_z)
    chi_LT = float(inp.chi_LT)

    k_yy = float(inp.k_yy)
    k_yz = float(inp.k_yz)
    k_zy = float(inp.k_zy)
    k_zz = float(inp.k_zz)

    # §6.3.3(4) – Equation 6.61
    eq_6_61_term1 = N_Ed / (chi_y * N_Rk / gM1)
    eq_6_61_term2 = k_yy * M_y / (chi_LT * M_y_Rk / gM1)
    eq_6_61_term3 = k_yz * M_z / (M_z_Rk / gM1) if M_z_Rk < 1e9 else 0.0
    eq_6_61 = eq_6_61_term1 + eq_6_61_term2 + eq_6_61_term3

    # §6.3.3(4) – Equation 6.62
    eq_6_62_term1 = N_Ed / (chi_z * N_Rk / gM1)
    eq_6_62_term2 = k_zy * M_y / (chi_LT * M_y_Rk / gM1)
    eq_6_62_term3 = k_zz * M_z / (M_z_Rk / gM1) if M_z_Rk < 1e9 else 0.0
    eq_6_62 = eq_6_62_term1 + eq_6_62_term2 + eq_6_62_term3

    governing = max(eq_6_61, eq_6_62)

    return {
        "inputs_used": {
            "N_Ed_kN": N_Ed,
            "M_y_Ed_kNm": float(inp.M_y_Ed_kNm),
            "M_z_Ed_kNm": float(inp.M_z_Ed_kNm),
            "chi_y": chi_y,
            "chi_z": chi_z,
            "chi_LT": chi_LT,
            "k_yy": k_yy,
            "k_yz": k_yz,
            "k_zy": k_zy,
            "k_zz": k_zz,
            "gamma_M1": gM1,
        },
        "intermediate": {
            "eq_6_61_axial_term": round(eq_6_61_term1, 4),
            "eq_6_61_My_term": round(eq_6_61_term2, 4),
            "eq_6_61_Mz_term": round(eq_6_61_term3, 4),
            "eq_6_62_axial_term": round(eq_6_62_term1, 4),
            "eq_6_62_My_term": round(eq_6_62_term2, 4),
            "eq_6_62_Mz_term": round(eq_6_62_term3, 4),
        },
        "outputs": {
            "eq_6_61": round(eq_6_61, 4),
            "eq_6_62": round(eq_6_62, 4),
            "governing_utilization": round(governing, 4),
            "governing_equation": "6.61" if eq_6_61 >= eq_6_62 else "6.62",
            "pass": governing <= 1.0,
        },
        "clause_references": [
            clause_ref("ec3.en1993-1-1.2005", "6.3.3(4)", "Uniform members in bending and axial compression"),
        ],
        "notes": [
            {
                "latex": rf"\text{{Eq. 6.61}} = {eq_6_61:.4f}\;(\text{{{'OK' if eq_6_61 <= 1.0 else 'FAIL'}}})",
            },
            {
                "latex": rf"\text{{Eq. 6.62}} = {eq_6_62:.4f}\;(\text{{{'OK' if eq_6_62 <= 1.0 else 'FAIL'}}})",
            },
            "Interaction factors k_ij should be from Annex A (Method 1) or Annex B (Method 2).",
        ],
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=CombinedBucklingInput, handler=calculate)
