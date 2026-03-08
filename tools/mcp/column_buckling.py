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


def _select_buckling_curve(
    h_mm: float, b_mm: float, tf_mm: float, axis: str, fabrication: str = "rolled",
) -> str:
    """Auto-select buckling curve per EC3 Table 6.2."""
    h_over_b = h_mm / b_mm if b_mm > 0 else 999.0
    if fabrication == "rolled":
        if h_over_b > 1.2:
            if tf_mm <= 40:
                return "a" if axis == "y" else "b"
            else:
                return "b" if axis == "y" else "c"
        else:  # h/b <= 1.2 (stocky sections like HEA, HEB)
            if tf_mm <= 100:
                return "b" if axis == "y" else "c"
            else:
                return "d"
    elif fabrication == "welded":
        if tf_mm <= 40:
            return "b" if axis == "y" else "c"
        else:
            return "c" if axis == "y" else "d"
    return "b"  # conservative fallback


class ColumnBucklingInput(BaseModel):
    section_name: Optional[str] = Field(default=None, description="Section name, e.g. IPE300")
    steel_grade: str = Field(default="S355", description="Steel grade, e.g. S355")
    fy_mpa: Optional[PositiveFloat] = Field(default=None, description="Override yield strength in MPa")
    thickness_mm: Optional[PositiveFloat] = Field(
        default=None,
        description="Governing element thickness for fy lookup (used when t > 40 mm).",
    )
    system_length_m: PositiveFloat = Field(description="Member system length in metres")
    k_factor: PositiveFloat = Field(default=1.0, description="Effective length factor k (default 1.0 = pinned-pinned)")
    buckling_axis: Literal["y", "z"] = Field(
        default="y",
        description="Buckling axis: 'y' for strong-axis, 'z' for weak-axis.",
    )
    buckling_curve: Optional[Literal["a0", "a", "b", "c", "d"]] = Field(
        default=None,
        description="EC3 buckling curve. If omitted, auto-selected from Table 6.2 based on section geometry and axis.",
    )
    fabrication: Literal["rolled", "welded"] = Field(
        default="rolled",
        description="Fabrication method: 'rolled' or 'welded'.",
    )
    gamma_M1: PositiveFloat = Field(default=1.0, description="Partial safety factor γ_M1")
    area_cm2: Optional[PositiveFloat] = Field(default=None, description="Cross-section area in cm²")
    I_cm4: Optional[PositiveFloat] = Field(default=None, description="Second moment of area for the buckling axis in cm⁴")
    section_class: int = Field(default=2, ge=1, le=4, description="Cross-section class (1-4)")


def calculate(inp: ColumnBucklingInput) -> dict:
    h_mm = None
    b_mm = None
    tf_mm = None

    if inp.section_name:
        key = inp.section_name.upper().replace(" ", "")
        if key in SECTION_LIBRARY:
            row = SECTION_LIBRARY[key]
            area = inp.area_cm2 or row.get("area_cm2")
            h_mm = row.get("h_mm")
            b_mm = row.get("b_mm")
            tf_mm = row.get("tf_mm")

            # Select correct moment of inertia based on buckling axis
            if inp.I_cm4:
                I = inp.I_cm4
            elif inp.buckling_axis == "z":
                I = row.get("I_z_cm4")
                if not I:
                    raise ValueError(
                        f"Section '{inp.section_name}' has no I_z_cm4 in library. Provide I_cm4 manually."
                    )
            else:
                I = row.get("I_y_cm4")
                if not I:
                    raise ValueError(
                        f"Section '{inp.section_name}' has no I_y_cm4 in library. Provide I_cm4 manually."
                    )
        else:
            raise ValueError(f"Section '{inp.section_name}' not found in library. Provide area_cm2 and I_cm4 manually.")
    else:
        area = inp.area_cm2
        I = inp.I_cm4

    if not area or not I:
        raise ValueError("Provide section_name (from library) or both area_cm2 and I_cm4.")

    # Yield strength — respect explicit override, then thickness-aware lookup
    if inp.fy_mpa:
        fy = float(inp.fy_mpa)
    else:
        t = inp.thickness_mm or (tf_mm if tf_mm else None)
        fy = steel_grade_to_fy(inp.steel_grade, thickness_mm=t)

    # Auto-select buckling curve if not provided
    curve = inp.buckling_curve
    curve_source = "user-provided"
    if curve is None:
        if h_mm and b_mm and tf_mm:
            curve = _select_buckling_curve(h_mm, b_mm, tf_mm, inp.buckling_axis, inp.fabrication)
            curve_source = "auto (Table 6.2)"
        else:
            curve = "b"  # safe default
            curve_source = "default (no geometry for Table 6.2)"

    A_mm2 = area * 100  # cm² → mm²
    I_mm4 = I * 1e4  # cm⁴ → mm⁴
    E_mpa = 210000  # N/mm²

    L_cr = inp.k_factor * inp.system_length_m
    L_cr_mm = L_cr * 1000  # m → mm

    N_cr = math.pi ** 2 * E_mpa * I_mm4 / L_cr_mm ** 2 / 1000  # kN

    i_mm = math.sqrt(I_mm4 / A_mm2)  # mm

    lambda_bar = math.sqrt(A_mm2 * fy / (N_cr * 1000))

    alpha = IMPERFECTION_FACTORS[curve]
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
            "buckling_axis": inp.buckling_axis,
            "buckling_curve": curve,
            "buckling_curve_source": curve_source,
            "fabrication": inp.fabrication,
            "gamma_M1": inp.gamma_M1,
            "area_cm2": round(area, 2),
            "I_cm4": round(I, 2),
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
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "Table 6.2",
                "title": "Selection of buckling curve for cross-section",
                "pointer": "en_1993_1_1_2005_ocr.json#table_6.2",
            },
        ],
        "notes": [
            f"Buckling axis: {inp.buckling_axis}-{inp.buckling_axis} ({curve_source}: curve '{curve}', α = {alpha})",
            f"Nb,Rd = χ·A·fy/γM1 = {chi:.4f} × {area:.2f} cm² × {fy} MPa / {inp.gamma_M1} = {Nb_Rd:.2f} kN",
            f"Relative slenderness λ̄ = {lambda_bar:.4f}",
            f"Reduction factor χ = {chi:.4f}",
        ],
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=ColumnBucklingInput, handler=calculate)
