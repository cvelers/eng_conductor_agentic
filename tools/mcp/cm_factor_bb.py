from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli

TOOL_NAME = "cm_factor_bb_ec3"


class CmFactorBBInput(BaseModel):
    """Input for BB.3.3.1 – Modification factor C_m for linear moment gradients."""

    # Section / member properties
    I_z_cm4: PositiveFloat = Field(description="Second moment of area I_z in cm⁴")
    I_T_cm4: PositiveFloat = Field(description="Torsion constant I_T in cm⁴")
    I_w_cm6: PositiveFloat = Field(description="Warping constant I_w in cm⁶")
    i_y_mm: PositiveFloat = Field(description="Radius of gyration about y-y in mm")
    i_z_mm: PositiveFloat = Field(description="Radius of gyration about z-z in mm")

    # Length and geometry
    L_t_mm: PositiveFloat = Field(description="Distance between torsional restraints L_t in mm")
    a_mm: float = Field(
        default=0.0,
        description="Distance between centroid of member and centroid of restraining members in mm",
    )

    # Moment ratio
    beta_t: float = Field(
        description="Ratio of smaller end moment to larger end moment β_t (-1 ≤ β_t ≤ 1). "
        "Moments producing compression in non-restrained flange are positive.",
    )


def calculate(inp: CmFactorBBInput) -> dict:
    E = 210000.0  # MPa
    G = 81000.0  # MPa

    Iz = float(inp.I_z_cm4) * 1e4  # mm⁴
    IT = float(inp.I_T_cm4) * 1e4  # mm⁴
    Iw = float(inp.I_w_cm6) * 1e6  # mm⁶
    iy = float(inp.i_y_mm)
    iz = float(inp.i_z_mm)
    Lt = float(inp.L_t_mm)
    a = float(inp.a_mm)
    bt = max(float(inp.beta_t), -1.0)

    notes: list[str] = []

    # N_crE = π²·E·I_z / L_t²
    N_crE = math.pi**2 * E * Iz / Lt**2  # N

    # i_s² = i_y² + i_z² + a²
    is2 = iy**2 + iz**2 + a**2

    # N_crT = (1/i_s²)·(π²·E·I_z·a² / L_t² + π²·E·I_w / L_t² + G·I_T)
    N_crT = (1.0 / is2) * (
        math.pi**2 * E * Iz * a**2 / Lt**2
        + math.pi**2 * E * Iw / Lt**2
        + G * IT
    )  # N

    eta = N_crE / N_crT if N_crT > 0 else 0.0

    notes.append(f"N_crE = π²·E·I_z/L_t² = {N_crE / 1000.0:.2f} kN")
    notes.append(f"i_s² = i_y² + i_z² + a² = {is2:.2f} mm²")
    notes.append(f"N_crT = {N_crT / 1000.0:.2f} kN")
    notes.append(f"η = N_crE/N_crT = {eta:.4f}")

    # B_0, B_1, B_2
    B_0 = (1.0 + 10.0 * eta) / (1.0 + 20.0 * eta)
    sqrt_eta = math.sqrt(max(eta, 0.0))
    B_1 = 5.0 * sqrt_eta / (math.pi + 10.0 * sqrt_eta)
    B_2 = 0.5 / (1.0 + math.pi * sqrt_eta) - 0.5 / (1.0 + 20.0 * eta)

    notes.append(f"B_0 = {B_0:.4f}")
    notes.append(f"B_1 = {B_1:.4f}")
    notes.append(f"B_2 = {B_2:.4f}")

    # C_m = 1 / (B_0 + B_1·β_t + B_2·β_t²)
    denom = B_0 + B_1 * bt + B_2 * bt**2
    if denom <= 0:
        C_m = 2.5  # safety cap
        notes.append("Warning: denominator ≤ 0, C_m capped at 2.5")
    else:
        C_m = 1.0 / denom

    notes.append(f"C_m = 1/(B_0 + B_1·β_t + B_2·β_t²) = 1/({denom:.4f}) = {C_m:.4f}")

    return {
        "inputs_used": {
            "I_z_cm4": float(inp.I_z_cm4),
            "I_T_cm4": float(inp.I_T_cm4),
            "I_w_cm6": float(inp.I_w_cm6),
            "i_y_mm": iy,
            "i_z_mm": iz,
            "L_t_mm": Lt,
            "a_mm": a,
            "beta_t": bt,
        },
        "intermediate": {
            "N_crE_kN": round(N_crE / 1000.0, 2),
            "N_crT_kN": round(N_crT / 1000.0, 2),
            "eta": round(eta, 4),
            "B_0": round(B_0, 4),
            "B_1": round(B_1, 4),
            "B_2": round(B_2, 4),
        },
        "outputs": {
            "C_m": round(C_m, 4),
        },
        "clause_references": [
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "BB.3.3.1",
                "title": "Modification factor C_m for linear moment gradients",
                "pointer": "en_1993_1_1_2005_structured.json#BB.3.3.1",
            },
        ],
        "notes": notes,
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=CmFactorBBInput, handler=calculate)
