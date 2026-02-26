from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli
from tools.mcp.section_library import SECTION_LIBRARY, steel_grade_to_fy

TOOL_NAME = "column_buckling_ec3"

IMPERFECTION_FACTORS = {
    "a0": 0.13,
    "a": 0.21,
    "b": 0.34,
    "c": 0.49,
    "d": 0.76,
}


class ColumnBucklingInput(BaseModel):
    section_name: Optional[str] = Field(default=None, description="Section name, e.g. IPE300")
    steel_grade: str = Field(default="S355", description="Steel grade, e.g. S355")
    system_length_m: PositiveFloat = Field(description="Member system length in metres")
    k_factor: PositiveFloat = Field(default=1.0, description="Effective length factor k (default 1.0 = pinned-pinned)")
    buckling_curve: Literal["a0", "a", "b", "c", "d"] = Field(
        default="b",
        description="EC3 buckling curve (a0, a, b, c, d). Default 'b' for hot-rolled I-sections.",
    )
    gamma_M1: PositiveFloat = Field(default=1.0, description="Partial safety factor γ_M1")
    area_cm2: Optional[PositiveFloat] = Field(default=None, description="Cross-section area in cm²")
    I_cm4: Optional[PositiveFloat] = Field(default=None, description="Second moment of area (weak/strong axis) in cm⁴")
    section_class: int = Field(default=2, ge=1, le=4, description="Cross-section class (1-4)")


def calculate(inp: ColumnBucklingInput) -> dict:
    if inp.section_name:
        key = inp.section_name.upper().replace(" ", "")
        if key in SECTION_LIBRARY:
            row = SECTION_LIBRARY[key]
            area = inp.area_cm2 or row.get("area_cm2")
            I_y = inp.I_cm4 or row.get("I_y_cm4")
            if not I_y:
                raise ValueError(f"Section '{inp.section_name}' has no I_y_cm4 in library. Provide I_cm4 manually.")
        else:
            raise ValueError(f"Section '{inp.section_name}' not found in library. Provide area_cm2 and I_cm4 manually.")
    else:
        area = inp.area_cm2
        I_y = inp.I_cm4

    if not area or not I_y:
        raise ValueError("Provide section_name (from library) or both area_cm2 and I_cm4.")

    fy = steel_grade_to_fy(inp.steel_grade)

    A_mm2 = area * 100  # cm² → mm²
    I_mm4 = I_y * 1e4  # cm⁴ → mm⁴
    E_mpa = 210000  # N/mm²

    L_cr = inp.k_factor * inp.system_length_m
    L_cr_mm = L_cr * 1000  # m → mm

    N_cr = math.pi ** 2 * E_mpa * I_mm4 / L_cr_mm ** 2 / 1000  # kN

    i_mm = math.sqrt(I_mm4 / A_mm2)  # mm

    lambda_bar = math.sqrt(A_mm2 * fy / (N_cr * 1000))

    alpha = IMPERFECTION_FACTORS[inp.buckling_curve]
    phi = 0.5 * (1 + alpha * (lambda_bar - 0.2) + lambda_bar ** 2)
    discriminant = phi ** 2 - lambda_bar ** 2
    if discriminant < 0:
        discriminant = 0
    chi = 1.0 / (phi + math.sqrt(discriminant))
    chi = min(chi, 1.0)

    Nb_Rd = chi * A_mm2 * fy / (inp.gamma_M1 * 1000)  # kN

    return {
        "inputs_used": {
            "section_name": inp.section_name,
            "steel_grade": inp.steel_grade,
            "fy_mpa": fy,
            "system_length_m": inp.system_length_m,
            "k_factor": inp.k_factor,
            "L_cr_m": round(L_cr, 4),
            "buckling_curve": inp.buckling_curve,
            "gamma_M1": inp.gamma_M1,
            "area_cm2": round(area, 2),
            "I_cm4": round(I_y, 2),
        },
        "intermediate": {
            "N_cr_kN": round(N_cr, 2),
            "i_mm": round(i_mm, 2),
            "lambda_bar": round(lambda_bar, 4),
            "alpha": alpha,
            "phi": round(phi, 4),
            "chi": round(chi, 4),
        },
        "outputs": {
            "Nb_Rd_kN": round(Nb_Rd, 2),
            "chi": round(chi, 4),
            "lambda_bar": round(lambda_bar, 4),
            "N_cr_kN": round(N_cr, 2),
        },
        "clause_references": [
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "6.3.1.1",
                "title": "Buckling resistance of compression members",
                "pointer": "en_1993_1_1_2005_ocr.json#6.3.1",
            },
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "6.3.1.2",
                "title": "Buckling curves — imperfection factors",
                "pointer": "en_1993_1_1_2005_ocr.json#table_6.1",
            },
        ],
        "notes": [
            f"Nb,Rd = χ·A·fy/γM1 = {chi:.4f} × {area:.2f} cm² × {fy} MPa / {inp.gamma_M1} = {Nb_Rd:.2f} kN",
            f"Relative slenderness λ̄ = {lambda_bar:.4f}",
            f"Reduction factor χ = {chi:.4f} (buckling curve '{inp.buckling_curve}')",
        ],
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=ColumnBucklingInput, handler=calculate)
