from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt

from tools.mcp.cli import run_cli
from tools.mcp.section_library import steel_grade_to_fy

TOOL_NAME = "builtup_column_ec3"

IMPERFECTION_FACTORS = {
    "a0": 0.13,
    "a": 0.21,
    "b": 0.34,
    "c": 0.49,
    "d": 0.76,
}


class BuiltupColumnInput(BaseModel):
    """Input for §6.4 – Uniform built-up compression members."""

    member_type: Literal["laced", "battened"] = Field(
        description="Type of built-up member: 'laced' or 'battened'"
    )
    steel_grade: str = Field(default="S355", description="Steel grade")
    fy_mpa: Optional[PositiveFloat] = Field(default=None, description="Yield strength in MPa")
    gamma_M0: PositiveFloat = Field(default=1.0, description="Partial factor γ_M0")
    gamma_M1: PositiveFloat = Field(default=1.0, description="Partial factor γ_M1")

    # Overall member
    L_m: PositiveFloat = Field(description="Overall member length in m")
    N_Ed_kN: PositiveFloat = Field(description="Design compression force N_Ed in kN")
    M_Ed_kNm: float = Field(default=0.0, description="Design bending moment M_Ed in kNm (from imperfections or load)")

    # Chord properties
    n_chords: PositiveInt = Field(default=2, description="Number of chords")
    A_ch_cm2: PositiveFloat = Field(description="Area of one chord in cm²")
    I_ch_cm4: PositiveFloat = Field(description="Second moment of one chord (about own axis) in cm⁴")
    h_0_mm: PositiveFloat = Field(description="Distance between chord centroids h_0 in mm")

    # Battened member specific
    a_mm: Optional[PositiveFloat] = Field(
        default=None, description="Distance between battens a in mm (for battened members)"
    )
    I_b_cm4: Optional[PositiveFloat] = Field(
        default=None, description="Second moment of one batten about batten axis in cm⁴"
    )

    # Laced member specific
    S_v_kN: Optional[PositiveFloat] = Field(
        default=None, description="Shear stiffness of lacing S_v in kN (if known)"
    )
    A_d_cm2: Optional[PositiveFloat] = Field(
        default=None, description="Area of one diagonal lacing member in cm²"
    )
    d_mm: Optional[PositiveFloat] = Field(
        default=None, description="Length of diagonal lacing member in mm"
    )

    buckling_curve: Literal["a0", "a", "b", "c", "d"] = Field(
        default="c", description="Buckling curve for overall member"
    )


def calculate(inp: BuiltupColumnInput) -> dict:
    fy = float(inp.fy_mpa) if inp.fy_mpa else steel_grade_to_fy(inp.steel_grade)
    E = 210000.0  # MPa
    gamma_M1 = float(inp.gamma_M1)

    L_mm = float(inp.L_m) * 1000.0
    N_Ed = float(inp.N_Ed_kN)
    n = int(inp.n_chords)
    A_ch_mm2 = float(inp.A_ch_cm2) * 100.0
    I_ch_mm4 = float(inp.I_ch_cm4) * 1e4
    h_0 = float(inp.h_0_mm)

    notes: list[str] = []

    # §6.4.1(6) – Bow imperfection e_0 = L/500
    e_0 = L_mm / 500.0

    if inp.member_type == "battened":
        # §6.4.3.1 – Battened compression members
        if inp.a_mm is None or inp.I_b_cm4 is None:
            raise ValueError("Battened members require a_mm and I_b_cm4.")

        a = float(inp.a_mm)
        I_b_mm4 = float(inp.I_b_cm4) * 1e4

        # S_v = 24·E·I_ch / (a² · [1 + 2·I_ch·h_0/(n·I_b·a)])
        # But S_v ≤ 2·π²·E·I_ch/a²
        denom = 1.0 + 2.0 * I_ch_mm4 * h_0 / (n * I_b_mm4 * a)
        S_v = 24.0 * E * I_ch_mm4 / (a**2 * denom) / 1000.0  # kN
        S_v_max = 2.0 * math.pi**2 * E * I_ch_mm4 / a**2 / 1000.0  # kN
        S_v = min(S_v, S_v_max)

        # I_eff = 0.5·h_0²·A_ch + 2·μ·I_ch  (μ = 0 for battened as simplification per §6.4.3.1)
        mu = 0.0
        I_eff = 0.5 * h_0**2 * A_ch_mm2 + 2.0 * mu * I_ch_mm4
        notes.append(f"S_v = {S_v:.2f} kN (battened shear stiffness)")
        notes.append(f"I_eff = 0.5·h₀²·A_ch = {I_eff / 1e4:.2f} cm⁴")

    else:
        # §6.4.2 – Laced compression members
        if inp.S_v_kN is not None:
            S_v = float(inp.S_v_kN)
        elif inp.A_d_cm2 and inp.d_mm and inp.a_mm:
            # S_v from lacing geometry (N-type): S_v = n·A_d·E·a·d²/(d³) simplified
            # Standard formula: S_v = 2·E·A_d·a·h_0²/(d³) for single-braced
            A_d_mm2 = float(inp.A_d_cm2) * 100.0
            d = float(inp.d_mm)
            a = float(inp.a_mm)
            S_v = n * E * A_d_mm2 * h_0**2 / d**3 * a / 1000.0  # kN
        else:
            raise ValueError("Laced members require S_v_kN or A_d_cm2 + d_mm + a_mm.")

        I_eff = 0.5 * h_0**2 * A_ch_mm2
        notes.append(f"S_v = {S_v:.2f} kN (laced shear stiffness)")
        notes.append(f"I_eff = 0.5·h₀²·A_ch = {I_eff / 1e4:.2f} cm⁴")

    # §6.4.1(6) – Effective critical force including shear flexibility
    N_cr_eff = math.pi**2 * E * I_eff / L_mm**2 / 1000.0  # kN (Euler, no shear correction)

    # With shear correction: 1/N_cr = 1/N_E + 1/S_v
    if S_v > 0:
        N_E = math.pi**2 * E * I_eff / L_mm**2 / 1000.0
        N_cr = 1.0 / (1.0 / N_E + 1.0 / S_v) if (1.0 / N_E + 1.0 / S_v) > 0 else N_E
    else:
        N_cr = N_cr_eff

    # Slenderness and buckling
    A_total_mm2 = n * A_ch_mm2
    lambda_bar = math.sqrt(A_total_mm2 * fy / (N_cr * 1000.0))

    alpha = IMPERFECTION_FACTORS[inp.buckling_curve]
    phi = 0.5 * (1.0 + alpha * (lambda_bar - 0.2) + lambda_bar**2)
    disc = phi**2 - lambda_bar**2
    chi = 1.0 / (phi + math.sqrt(max(disc, 0.0)))
    chi = min(chi, 1.0)

    Nb_Rd = chi * A_total_mm2 * fy / (gamma_M1 * 1000.0)

    # §6.4.1(7) – Design chord forces
    # M_Ed from imperfection: M_Ed = N_Ed · e_0 · N_cr/(N_cr - N_Ed) (amplified)
    if N_cr > N_Ed:
        M_Ed_imp = N_Ed * e_0 / 1000.0 * N_cr / (N_cr - N_Ed)  # kNm
    else:
        M_Ed_imp = float("inf")

    M_Ed_total = abs(float(inp.M_Ed_kNm)) + M_Ed_imp

    # Chord force: N_ch,Ed = N_Ed/n + M_Ed·h_0·A_ch / (2·I_eff)
    N_ch_Ed = N_Ed / n + M_Ed_total * 1e6 * h_0 * A_ch_mm2 / (2.0 * I_eff) / 1000.0

    # Shear force for chord/lacing check
    V_Ed = math.pi * M_Ed_total / (float(inp.L_m))  # kN (approximate sinusoidal distribution)

    notes.append(f"e_0 = L/500 = {e_0:.1f} mm")
    notes.append(f"N_cr = {N_cr:.2f} kN (with shear flexibility)")
    notes.append(f"λ̄ = {lambda_bar:.4f}, χ = {chi:.4f}")
    notes.append(f"N_b,Rd = {Nb_Rd:.2f} kN")
    notes.append(f"M_Ed (from imperfection) = {M_Ed_imp:.2f} kNm")
    notes.append(f"N_ch,Ed = {N_ch_Ed:.2f} kN (maximum chord force)")

    return {
        "inputs_used": {
            "member_type": inp.member_type,
            "steel_grade": inp.steel_grade,
            "fy_mpa": fy,
            "L_m": float(inp.L_m),
            "N_Ed_kN": N_Ed,
            "n_chords": n,
            "A_ch_cm2": float(inp.A_ch_cm2),
            "h_0_mm": h_0,
        },
        "intermediate": {
            "e_0_mm": round(e_0, 1),
            "S_v_kN": round(S_v, 2),
            "I_eff_cm4": round(I_eff / 1e4, 2),
            "N_cr_kN": round(N_cr, 2),
            "lambda_bar": round(lambda_bar, 4),
            "chi": round(chi, 4),
        },
        "outputs": {
            "Nb_Rd_kN": round(Nb_Rd, 2),
            "N_ch_Ed_kN": round(N_ch_Ed, 2),
            "V_Ed_kN": round(V_Ed, 2),
            "M_Ed_imperfection_kNm": round(M_Ed_imp, 2),
            "utilization_overall": round(N_Ed / Nb_Rd, 4) if Nb_Rd > 0 else float("inf"),
            "pass_overall": N_Ed <= Nb_Rd,
        },
        "clause_references": [
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "6.4.1",
                "title": "Built-up compression members – General",
                "pointer": "en_1993_1_1_2005_structured.json#6.4.1",
            },
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "6.4.2.1" if inp.member_type == "laced" else "6.4.3.1",
                "title": f"{'Laced' if inp.member_type == 'laced' else 'Battened'} compression members",
                "pointer": f"en_1993_1_1_2005_structured.json#{'6.4.2.1' if inp.member_type == 'laced' else '6.4.3.1'}",
            },
        ],
        "notes": notes,
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=BuiltupColumnInput, handler=calculate)
