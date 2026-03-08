from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli

TOOL_NAME = "interaction_factors_b_ec3"


class InteractionFactorsBInput(BaseModel):
    """Input for Annex B [informative] – Method 2 interaction factors k_ij (clause 6.3.3)."""

    section_type: Literal["I", "RHS"] = Field(
        description="Section type: 'I' for I/H-sections, 'RHS' for rectangular hollow sections"
    )
    susceptible_to_torsion: bool = Field(
        default=False,
        description="Whether the member is susceptible to torsional deformations (Table B.2 vs B.1)",
    )

    # Slenderness
    lambda_bar_y: float = Field(description="Non-dimensional slenderness λ̄_y about y-y axis")
    lambda_bar_z: float = Field(description="Non-dimensional slenderness λ̄_z about z-z axis")

    # Reduction factors
    chi_y: PositiveFloat = Field(description="Buckling reduction factor χ_y about y-y")
    chi_z: PositiveFloat = Field(description="Buckling reduction factor χ_z about z-z")

    # Forces / resistances
    N_Ed_kN: PositiveFloat = Field(description="Design axial compression force N_Ed in kN")
    N_Rk_kN: PositiveFloat = Field(description="Characteristic resistance N_Rk = A·fy in kN")
    gamma_M1: PositiveFloat = Field(default=1.0, description="Partial factor γ_M1")

    # Equivalent uniform moment factors
    C_my: float = Field(description="Equivalent uniform moment factor C_my (Table B.3)")
    C_mz: float = Field(description="Equivalent uniform moment factor C_mz (Table B.3)")
    C_mLT: float = Field(
        default=1.0,
        description="Equivalent uniform moment factor C_mLT (Table B.3, for torsional susceptibility)",
    )


def calculate(inp: InteractionFactorsBInput) -> dict:
    N = float(inp.N_Ed_kN)
    NRk = float(inp.N_Rk_kN)
    gM1 = float(inp.gamma_M1)
    ly = float(inp.lambda_bar_y)
    lz = float(inp.lambda_bar_z)
    cy = float(inp.chi_y)
    cz = float(inp.chi_z)
    Cmy = float(inp.C_my)
    Cmz = float(inp.C_mz)
    CmLT = float(inp.C_mLT)

    n_y = N / (cy * NRk / gM1)
    n_z = N / (cz * NRk / gM1)

    notes: list[str] = [
        f"n_y = N_Ed/(χ_y·N_Rk/γ_M1) = {n_y:.4f}",
        f"n_z = N_Ed/(χ_z·N_Rk/γ_M1) = {n_z:.4f}",
    ]

    # ── Table B.1 ── (not susceptible to torsional deformations)
    # k_yy
    k_yy_calc = Cmy * (1.0 + (ly - 0.2) * n_y)
    k_yy_max = Cmy * (1.0 + 0.8 * n_y)
    k_yy = min(k_yy_calc, k_yy_max)
    notes.append(f"k_yy = min({k_yy_calc:.4f}, {k_yy_max:.4f}) = {k_yy:.4f}")

    # k_zz
    k_zz_calc = Cmz * (1.0 + (2.0 * lz - 0.6) * n_z)
    k_zz_max = Cmz * (1.0 + 1.4 * n_z)
    k_zz = min(k_zz_calc, k_zz_max)
    notes.append(f"k_zz = min({k_zz_calc:.4f}, {k_zz_max:.4f}) = {k_zz:.4f}")

    # k_yz and k_zy depend on section type
    if inp.section_type == "I":
        k_yz = 0.6 * k_zz
        k_zy_b1 = 0.6 * k_yy
        notes.append(f"k_yz = 0.6·k_zz = {k_yz:.4f} (I-section)")
        notes.append(f"k_zy (Table B.1) = 0.6·k_yy = {k_zy_b1:.4f} (I-section)")
    else:
        k_yz = k_zz
        k_zy_b1 = k_yy
        notes.append(f"k_yz = k_zz = {k_yz:.4f} (RHS)")
        notes.append(f"k_zy (Table B.1) = k_yy = {k_zy_b1:.4f} (RHS)")

    k_zy = k_zy_b1

    # ── Table B.2 ── (susceptible to torsional deformations)
    if inp.susceptible_to_torsion:
        # k_yy and k_yz remain from Table B.1
        # k_zy is modified
        denom = max(CmLT - 0.25, 0.01)

        if inp.section_type == "I":
            k_zy_calc = (1.0 - 0.05 * lz / denom) * n_z
            k_zy_calc = 1.0 - 0.05 * lz / denom * n_z
            k_zy_max = 1.0 - 0.05 / denom * n_z
            k_zy = min(k_zy_calc, k_zy_max)
        else:
            k_zy_calc = 1.0 - 0.1 * lz / denom * n_z
            k_zy_max = 1.0 - 0.1 / denom * n_z
            k_zy = min(k_zy_calc, k_zy_max)

        # For λ̄_z < 0.4, alternative formula
        if lz < 0.4:
            k_zy_alt = 0.6 + lz
            k_zy_alt = min(k_zy_alt, 1.0)
            k_zy = max(k_zy, k_zy_alt) if k_zy_alt <= 1.0 else k_zy
            notes.append(f"λ̄_z < 0.4: k_zy alternative = 0.6 + λ̄_z = {k_zy_alt:.4f}")

        notes.append(f"k_zy (Table B.2, torsion-susceptible) = {k_zy:.4f}")

    # k_zz for Table B.2 is same as B.1

    return {
        "inputs_used": {
            "section_type": inp.section_type,
            "susceptible_to_torsion": inp.susceptible_to_torsion,
            "lambda_bar_y": ly,
            "lambda_bar_z": lz,
            "chi_y": cy,
            "chi_z": cz,
            "N_Ed_kN": N,
            "N_Rk_kN": NRk,
            "gamma_M1": gM1,
            "C_my": Cmy,
            "C_mz": Cmz,
            "C_mLT": CmLT,
        },
        "intermediate": {
            "n_y": round(n_y, 4),
            "n_z": round(n_z, 4),
        },
        "outputs": {
            "k_yy": round(k_yy, 4),
            "k_yz": round(k_yz, 4),
            "k_zy": round(k_zy, 4),
            "k_zz": round(k_zz, 4),
        },
        "clause_references": [
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "B.1" if not inp.susceptible_to_torsion else "B.2",
                "title": f"Method 2 interaction factors – Table B.{'1' if not inp.susceptible_to_torsion else '2'}",
                "pointer": "en_1993_1_1_2005_structured.json#B",
            },
        ],
        "notes": notes,
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=InteractionFactorsBInput, handler=calculate)
