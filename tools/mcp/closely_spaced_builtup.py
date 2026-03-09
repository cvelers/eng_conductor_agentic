from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli
from tools.mcp.section_library import steel_grade_to_fy
from tools.mcp.clause_ref import clause_ref

TOOL_NAME = "closely_spaced_builtup_ec3"

IMPERFECTION_FACTORS = {
    "a0": 0.13,
    "a": 0.21,
    "b": 0.34,
    "c": 0.49,
    "d": 0.76,
}


class CloselySpacedBuiltupInput(BaseModel):
    """Input for §6.4.4 – Closely spaced built-up compression members."""

    member_type: Literal["packing_plates", "star_battened"] = Field(
        description="'packing_plates' (bolted/welded through packing, Fig 6.12) or "
        "'star_battened' (star battened angles, Fig 6.13)"
    )

    steel_grade: str = Field(default="S355", description="Steel grade")
    fy_mpa: Optional[PositiveFloat] = Field(default=None, description="Yield strength in MPa")
    gamma_M1: PositiveFloat = Field(default=1.0, description="Partial factor γ_M1")

    # Overall member
    L_mm: PositiveFloat = Field(description="Member length L in mm")
    N_Ed_kN: PositiveFloat = Field(description="Design compression force N_Ed in kN")

    # Total built-up section properties
    A_total_cm2: PositiveFloat = Field(description="Total area of built-up section in cm²")
    I_total_cm4: PositiveFloat = Field(description="Second moment of area of built-up section in cm⁴")

    # Individual chord/angle
    i_min_mm: PositiveFloat = Field(
        description="Minimum radius of gyration of one chord or one angle in mm"
    )
    spacing_mm: PositiveFloat = Field(
        description="Centre-to-centre spacing of interconnections in mm"
    )

    # For star-battened unequal-leg angles
    is_unequal_leg: bool = Field(
        default=False,
        description="True for unequal-leg star battened angles (uses i_y = i_0/1.15)",
    )
    i_0_mm: Optional[PositiveFloat] = Field(
        default=None,
        description="Minimum radius of gyration of the built-up member i_0 in mm (for unequal legs)",
    )

    buckling_curve: Literal["a0", "a", "b", "c", "d"] = Field(
        default="c", description="Buckling curve"
    )


def calculate(inp: CloselySpacedBuiltupInput) -> dict:
    fy = float(inp.fy_mpa) if inp.fy_mpa else steel_grade_to_fy(inp.steel_grade)
    gM1 = float(inp.gamma_M1)
    E = 210000.0

    L = float(inp.L_mm)
    N_Ed = float(inp.N_Ed_kN)
    A = float(inp.A_total_cm2) * 100.0  # mm²
    I = float(inp.I_total_cm4) * 1e4  # mm⁴
    i_min = float(inp.i_min_mm)
    spacing = float(inp.spacing_mm)

    notes: list[str] = []

    # Check max spacing (Table 6.9)
    if inp.member_type == "packing_plates":
        max_spacing = 15.0 * i_min
        spacing_label = "15·i_min"
    else:
        max_spacing = 70.0 * i_min
        spacing_label = "70·i_min"

    spacing_ok = spacing <= max_spacing
    notes.append(
        f"Max interconnection spacing: {spacing_label} = {max_spacing:.1f} mm → "
        f"spacing = {spacing:.1f} mm {'≤' if spacing_ok else '>'} limit → "
        f"{'OK (treat as integral)' if spacing_ok else 'EXCEEDED (shear stiffness needed)'}"
    )

    if not spacing_ok:
        notes.append("Warning: spacing exceeds limit – member should be checked as §6.4.1–6.4.3")

    # Buckling as single integral member (S_v = ∞)
    i_built = math.sqrt(I / A)

    # For star-battened unequal-leg: i_y = i_0/1.15
    if inp.is_unequal_leg and inp.i_0_mm:
        i_0 = float(inp.i_0_mm)
        i_y = i_0 / 1.15
        notes.append(f"Unequal-leg angles: i_y = i_0/1.15 = {i_0:.2f}/1.15 = {i_y:.2f} mm")
        i_eff = i_y
    else:
        i_eff = i_built

    lambda_bar = (L / i_eff) / (93.9 * math.sqrt(235.0 / fy))

    alpha = IMPERFECTION_FACTORS[inp.buckling_curve]
    phi = 0.5 * (1.0 + alpha * (lambda_bar - 0.2) + lambda_bar**2)
    disc = phi**2 - lambda_bar**2
    chi = 1.0 / (phi + math.sqrt(max(disc, 0.0)))
    chi = min(chi, 1.0)

    Nb_Rd = chi * A * fy / (gM1 * 1000.0)  # kN

    notes.append(f"i_built = √(I/A) = {i_built:.2f} mm")
    notes.append(f"λ̄ = {lambda_bar:.4f}, χ = {chi:.4f}")
    notes.append(f"N_b,Rd = χ·A·fy/γ_M1 = {Nb_Rd:.2f} kN")
    notes.append(f"Utilization = {N_Ed / Nb_Rd:.4f}")

    return {
        "inputs_used": {
            "member_type": inp.member_type,
            "steel_grade": inp.steel_grade,
            "fy_mpa": fy,
            "L_mm": L,
            "N_Ed_kN": N_Ed,
            "spacing_mm": spacing,
            "i_min_mm": i_min,
        },
        "intermediate": {
            "max_spacing_mm": round(max_spacing, 1),
            "spacing_ok": spacing_ok,
            "i_built_mm": round(i_built, 2),
            "lambda_bar": round(lambda_bar, 4),
            "phi": round(phi, 4),
            "chi": round(chi, 4),
        },
        "outputs": {
            "Nb_Rd_kN": round(Nb_Rd, 2),
            "utilization": round(N_Ed / Nb_Rd, 4) if Nb_Rd > 0 else float("inf"),
            "pass": N_Ed <= Nb_Rd,
            "spacing_within_limit": spacing_ok,
        },
        "clause_references": [
            clause_ref("ec3.en1993-1-1.2005", "6.4.4", "Closely spaced built-up members"),
        ],
        "notes": notes,
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=CloselySpacedBuiltupInput, handler=calculate)
