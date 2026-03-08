from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli
from tools.mcp.section_library import steel_grade_to_fy

TOOL_NAME = "stable_length_ec3"


class StableLengthInput(BaseModel):
    """Input for Annex BB.3 – Stable lengths between restraints for members with plastic hinges."""

    method: Literal["elastic", "plastic_lateral", "plastic_torsional"] = Field(
        default="elastic",
        description="'elastic' (BB.3.1.1), 'plastic_lateral' (BB.3.2.1), 'plastic_torsional' (BB.3.1.2/BB.3.2.2)",
    )
    steel_grade: str = Field(default="S355", description="Steel grade, e.g. S355")
    fy_mpa: Optional[PositiveFloat] = Field(default=None, description="Yield strength in MPa")

    # Section properties
    i_z_mm: PositiveFloat = Field(description="Radius of gyration about z-z in mm")
    h_mm: Optional[PositiveFloat] = Field(default=None, description="Section depth h in mm")
    tf_mm: Optional[PositiveFloat] = Field(default=None, description="Flange thickness t_f in mm")

    # For elastic stable length (BB.3.1.1)
    psi: float = Field(
        default=1.0,
        description="Moment ratio ψ = M_min/M_max (-1 ≤ ψ ≤ 1)",
    )

    # For plastic stable lengths (BB.3.2.1)
    N_Ed_kN: float = Field(default=0.0, description="Design axial compression force in kN")
    area_cm2: Optional[PositiveFloat] = Field(default=None, description="Gross area A in cm²")
    wpl_y_cm3: Optional[PositiveFloat] = Field(default=None, description="Plastic modulus W_pl,y in cm³")
    I_T_cm4: Optional[PositiveFloat] = Field(default=None, description="Torsion constant I_T in cm⁴")
    I_z_cm4: Optional[PositiveFloat] = Field(default=None, description="Second moment I_z in cm⁴")
    I_w_cm6: Optional[float] = Field(default=None, description="Warping constant I_w in cm⁶")
    C1: PositiveFloat = Field(default=1.0, description="Moment distribution factor C1")

    # For torsional stable length (BB.3.1.2)
    L_t_m: Optional[PositiveFloat] = Field(
        default=None, description="Length between torsional restraints in m (for BB.3.1.2)"
    )

    member_type: Literal["rolled", "welded_equal", "welded_unequal"] = Field(
        default="rolled", description="Section fabrication type"
    )


def calculate(inp: StableLengthInput) -> dict:
    fy = float(inp.fy_mpa) if inp.fy_mpa else steel_grade_to_fy(inp.steel_grade)
    E = 210000.0
    epsilon = math.sqrt(235.0 / fy)
    i_z = float(inp.i_z_mm)

    notes: list[str] = [f"ε = √(235/fy) = √(235/{fy:.0f}) = {epsilon:.4f}"]
    results: dict = {"epsilon": round(epsilon, 4)}

    if inp.method == "elastic":
        # BB.3.1.1 – Stable length between adjacent lateral restraints (elastic)
        psi = float(inp.psi)

        if 0.625 <= psi <= 1.0:
            L_stable = 35.0 * epsilon * i_z  # mm
            formula = "L_stable = 35·ε·i_z"
        elif -1.0 <= psi < 0.625:
            L_stable = (60.0 - 40.0 * psi) * epsilon * i_z  # mm
            formula = "L_stable = (60 − 40ψ)·ε·i_z"
        else:
            raise ValueError("ψ must be between -1 and 1.")

        L_stable_m = L_stable / 1000.0
        results["L_stable_mm"] = round(L_stable, 1)
        results["L_stable_m"] = round(L_stable_m, 3)
        results["formula"] = formula
        notes.append(f"ψ = {psi:.3f} → {formula} = {L_stable:.1f} mm = {L_stable_m:.3f} m")

    elif inp.method == "plastic_lateral":
        # BB.3.2.1 – Stable length between adjacent lateral restraints (plastic hinges)
        if inp.area_cm2 is None or inp.wpl_y_cm3 is None:
            raise ValueError("plastic_lateral requires area_cm2 and wpl_y_cm3.")

        A_mm2 = float(inp.area_cm2) * 100.0
        Wpl_y_mm3 = float(inp.wpl_y_cm3) * 1000.0
        N_Ed = abs(float(inp.N_Ed_kN)) * 1000.0  # N
        I_T_mm4 = float(inp.I_T_cm4) * 1e4 if inp.I_T_cm4 else 0.0
        I_z_mm4 = float(inp.I_z_cm4) * 1e4 if inp.I_z_cm4 else 0.0
        C1 = float(inp.C1)

        # L_m = 38·i_z / √( (N_Ed/(57.4·A)) + (Wpl,y²·fy²)/(756·C1²·E²·I_T·I_z) )
        term1 = N_Ed / (57.4 * A_mm2) if A_mm2 > 0 else 0
        term2 = (Wpl_y_mm3**2 * fy**2) / (756.0 * C1**2 * E**2 * max(I_T_mm4, 1) * max(I_z_mm4, 1))

        denom = term1 + term2
        if denom > 0:
            L_m = 38.0 * i_z / math.sqrt(denom)
        else:
            L_m = float("inf")

        # Apply 0.85 factor for welded sections
        if inp.member_type != "rolled":
            L_m *= 0.85
            notes.append("Welded section: L_m multiplied by 0.85")

        L_m_m = L_m / 1000.0
        results["L_m_mm"] = round(L_m, 1)
        results["L_m_m"] = round(L_m_m, 3)
        notes.append(f"L_m = 38·i_z/√(...) = {L_m:.1f} mm = {L_m_m:.3f} m")

    elif inp.method == "plastic_torsional":
        # BB.3.1.2 – Stable length between torsional restraints
        if inp.h_mm is None or inp.tf_mm is None:
            raise ValueError("Torsional stable length requires h_mm and tf_mm.")

        h = float(inp.h_mm)
        tf = float(inp.tf_mm)
        h_tf = h / tf

        # L_k for uniform member:
        # L_k = (5.4 + 600·fy/E) / √(5.4·(fy/E)·(h/tf)² − 1) · (h/tf)
        # Simplified from BB.3.1.2
        fy_E = fy / E
        numer = 5.4 + 600.0 * fy_E
        inner = 5.4 * fy_E * h_tf**2 - 1.0
        if inner <= 0:
            L_k = float("inf")
            notes.append("h/tf ratio gives stable length → no LTB concern")
        else:
            L_k_factor = numer / math.sqrt(inner)
            L_k = L_k_factor * i_z  # mm

        L_k_m = L_k / 1000.0
        results["L_k_mm"] = round(L_k, 1) if L_k != float("inf") else float("inf")
        results["L_k_m"] = round(L_k_m, 3) if L_k != float("inf") else float("inf")
        notes.append(f"L_k = {L_k:.1f} mm = {L_k_m:.3f} m")

    return {
        "inputs_used": {
            "method": inp.method,
            "steel_grade": inp.steel_grade,
            "fy_mpa": fy,
            "i_z_mm": i_z,
            "member_type": inp.member_type,
        },
        "outputs": results,
        "clause_references": [
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "BB.3",
                "title": "Stable lengths of segment",
                "pointer": "en_1993_1_1_2005_structured.json#BB.3.1.1",
            },
        ],
        "notes": notes,
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=StableLengthInput, handler=calculate)
