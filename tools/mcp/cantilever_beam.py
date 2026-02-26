from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli

TOOL_NAME = "cantilever_beam_calculator"


class CantileverBeamInput(BaseModel):
    load_type: Literal["point_tip", "udl"] = Field(
        default="point_tip",
        description="'point_tip' (point load at free end) or 'udl' (uniform distributed load)",
    )
    span_m: PositiveFloat = Field(description="Cantilever length in metres")
    load_kn: Optional[PositiveFloat] = Field(default=None, description="Point load P in kN (tip load)")
    load_kn_per_m: Optional[PositiveFloat] = Field(default=None, description="UDL w in kN/m")
    E_gpa: PositiveFloat = Field(default=210.0, description="Young's modulus in GPa")
    I_cm4: Optional[PositiveFloat] = Field(default=None, description="Second moment of area in cm4")


def calculate(inp: CantileverBeamInput) -> dict:
    L = inp.span_m
    E = inp.E_gpa * 1e6  # kN/m²
    I = inp.I_cm4 * 1e-8 if inp.I_cm4 else None  # m⁴

    if inp.load_type == "point_tip":
        P = inp.load_kn
        if P is None:
            raise ValueError("load_kn is required for point_tip load type.")

        R = P
        M_fixed = P * L
        V_max = P

        defl = None
        if I:
            defl = (P * L ** 3) / (3 * E * I) * 1000  # mm

        return {
            "inputs_used": {"load_type": "point_tip", "span_m": L, "P_kN": P, "E_GPa": inp.E_gpa, "I_cm4": inp.I_cm4},
            "intermediate": {"R_fixed_kN": round(R, 3), "M_fixed_kNm": round(M_fixed, 3)},
            "outputs": {
                "M_max_kNm": round(M_fixed, 3),
                "V_max_kN": round(V_max, 3),
                "R_fixed_kN": round(R, 3),
                "M_fixed_kNm": round(M_fixed, 3),
                "delta_tip_mm": round(defl, 4) if defl is not None else "N/A (provide I_cm4)",
            },
            "clause_references": [
                {"doc_id": "structural_mechanics", "clause_id": "cantilever_beam", "title": "Cantilever beam — point load at tip", "pointer": "structural_mechanics#cantilever"},
            ],
            "notes": ["Euler-Bernoulli beam theory. Maximum moment and reaction at fixed support."],
        }

    else:
        w = inp.load_kn_per_m
        if w is None:
            raise ValueError("load_kn_per_m is required for UDL type.")

        R = w * L
        M_fixed = w * L ** 2 / 2
        V_max = w * L

        defl = None
        if I:
            defl = (w * L ** 4) / (8 * E * I) * 1000  # mm

        return {
            "inputs_used": {"load_type": "udl", "span_m": L, "w_kN_per_m": w, "E_GPa": inp.E_gpa, "I_cm4": inp.I_cm4},
            "intermediate": {"R_fixed_kN": round(R, 3), "M_fixed_kNm": round(M_fixed, 3)},
            "outputs": {
                "M_max_kNm": round(M_fixed, 3),
                "V_max_kN": round(V_max, 3),
                "R_fixed_kN": round(R, 3),
                "M_fixed_kNm": round(M_fixed, 3),
                "delta_tip_mm": round(defl, 4) if defl is not None else "N/A (provide I_cm4)",
            },
            "clause_references": [
                {"doc_id": "structural_mechanics", "clause_id": "cantilever_beam", "title": "Cantilever beam — full UDL", "pointer": "structural_mechanics#cantilever"},
            ],
            "notes": ["Euler-Bernoulli beam theory. Maximum moment and reaction at fixed support."],
        }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=CantileverBeamInput, handler=calculate)
