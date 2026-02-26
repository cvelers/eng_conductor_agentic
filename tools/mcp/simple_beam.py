from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli

TOOL_NAME = "simple_beam_calculator"


class SimpleBeamInput(BaseModel):
    load_type: Literal["point_mid", "point", "udl"] = Field(
        default="udl",
        description="Load type: 'point_mid' (point at midspan), 'point' (point at position a), 'udl' (uniform distributed load)",
    )
    span_m: PositiveFloat = Field(description="Beam span in metres")
    load_kn: Optional[PositiveFloat] = Field(default=None, description="Point load P in kN")
    load_kn_per_m: Optional[PositiveFloat] = Field(default=None, description="UDL w in kN/m")
    position_a_m: Optional[PositiveFloat] = Field(
        default=None,
        description="Distance of point load from left support (m). Defaults to midspan.",
    )
    E_gpa: PositiveFloat = Field(default=210.0, description="Young's modulus in GPa (default: steel 210)")
    I_cm4: Optional[PositiveFloat] = Field(default=None, description="Second moment of area in cm4 (needed for deflection)")


def calculate(inp: SimpleBeamInput) -> dict:
    L = inp.span_m
    E = inp.E_gpa * 1e6  # kN/m²
    I = inp.I_cm4 * 1e-8 if inp.I_cm4 else None  # m⁴

    if inp.load_type in ("point_mid", "point"):
        P = inp.load_kn
        if P is None:
            raise ValueError("load_kn is required for point load types.")
        a = inp.position_a_m if inp.position_a_m else L / 2.0
        if a > L:
            raise ValueError(f"position_a_m ({a}) cannot exceed span ({L}).")
        b = L - a

        R_A = P * b / L
        R_B = P * a / L
        M_max = P * a * b / L
        V_max = max(R_A, R_B)

        defl = None
        if I:
            if abs(a - L / 2) < 0.001:
                defl = (P * L ** 3) / (48 * E * I) * 1000  # mm
            else:
                defl = (P * b * (L ** 2 - b ** 2) ** 1.5) / (9 * math.sqrt(3) * E * I * L) * 1000

        return {
            "inputs_used": {
                "load_type": inp.load_type,
                "span_m": L,
                "P_kN": P,
                "a_m": round(a, 4),
                "b_m": round(b, 4),
                "E_GPa": inp.E_gpa,
                "I_cm4": inp.I_cm4,
            },
            "intermediate": {
                "R_A_kN": round(R_A, 3),
                "R_B_kN": round(R_B, 3),
            },
            "outputs": {
                "M_max_kNm": round(M_max, 3),
                "V_max_kN": round(V_max, 3),
                "R_A_kN": round(R_A, 3),
                "R_B_kN": round(R_B, 3),
                "delta_max_mm": round(defl, 4) if defl is not None else "N/A (provide I_cm4)",
            },
            "clause_references": [
                {
                    "doc_id": "structural_mechanics",
                    "clause_id": "simply_supported_beam",
                    "title": "Simply supported beam — standard formulae",
                    "pointer": "structural_mechanics#beam_formulas",
                },
            ],
            "notes": ["Standard beam theory (Euler-Bernoulli). Shear deformation neglected."],
        }

    else:
        w = inp.load_kn_per_m
        if w is None:
            raise ValueError("load_kn_per_m is required for UDL type.")

        R_A = w * L / 2
        R_B = R_A
        M_max = w * L ** 2 / 8
        V_max = w * L / 2

        defl = None
        if I:
            defl = (5 * w * L ** 4) / (384 * E * I) * 1000  # mm

        return {
            "inputs_used": {
                "load_type": "udl",
                "span_m": L,
                "w_kN_per_m": w,
                "E_GPa": inp.E_gpa,
                "I_cm4": inp.I_cm4,
            },
            "intermediate": {
                "R_A_kN": round(R_A, 3),
                "R_B_kN": round(R_B, 3),
            },
            "outputs": {
                "M_max_kNm": round(M_max, 3),
                "V_max_kN": round(V_max, 3),
                "R_A_kN": round(R_A, 3),
                "R_B_kN": round(R_B, 3),
                "delta_max_mm": round(defl, 4) if defl is not None else "N/A (provide I_cm4)",
            },
            "clause_references": [
                {
                    "doc_id": "structural_mechanics",
                    "clause_id": "simply_supported_beam",
                    "title": "Simply supported beam under UDL — standard formulae",
                    "pointer": "structural_mechanics#beam_formulas",
                },
            ],
            "notes": ["Standard beam theory (Euler-Bernoulli). Shear deformation neglected."],
        }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=SimpleBeamInput, handler=calculate)
