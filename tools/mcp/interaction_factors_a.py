from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli

TOOL_NAME = "interaction_factors_a_ec3"


class InteractionFactorsAInput(BaseModel):
    """Input for Annex A [informative] – Method 1 interaction factors k_ij (clause 6.3.3)."""

    # Cross-section class
    section_class: Literal[1, 2, 3, 4] = Field(
        default=1, description="Cross-section class (1, 2, 3, or 4)"
    )

    # Section moduli
    W_pl_y_cm3: PositiveFloat = Field(description="Plastic section modulus W_pl,y in cm³")
    W_el_y_cm3: PositiveFloat = Field(description="Elastic section modulus W_el,y in cm³")
    W_pl_z_cm3: PositiveFloat = Field(description="Plastic section modulus W_pl,z in cm³")
    W_el_z_cm3: PositiveFloat = Field(description="Elastic section modulus W_el,z in cm³")

    # Torsional properties
    I_T_cm4: PositiveFloat = Field(description="St. Venant torsional constant I_T in cm⁴")
    I_y_cm4: PositiveFloat = Field(description="Second moment of area I_y in cm⁴")

    # Slenderness
    lambda_bar_y: float = Field(description="Non-dimensional slenderness λ̄_y about y-y")
    lambda_bar_z: float = Field(description="Non-dimensional slenderness λ̄_z about z-z")
    lambda_bar_LT: float = Field(default=0.0, description="Non-dimensional slenderness for LTB λ̄_LT")
    lambda_bar_0: float = Field(
        default=0.0,
        description="LTB slenderness for uniform moment (ψ=1.0) λ̄_0",
    )

    # Reduction factors
    chi_y: PositiveFloat = Field(description="Flexural buckling reduction factor χ_y")
    chi_z: PositiveFloat = Field(description="Flexural buckling reduction factor χ_z")
    chi_LT: PositiveFloat = Field(default=1.0, description="LTB reduction factor χ_LT")

    # Critical forces
    N_cr_y_kN: PositiveFloat = Field(description="Elastic critical force N_cr,y in kN")
    N_cr_z_kN: PositiveFloat = Field(description="Elastic critical force N_cr,z in kN")
    N_cr_T_kN: Optional[PositiveFloat] = Field(
        default=None, description="Elastic critical torsional force N_cr,T in kN"
    )

    # Design forces
    N_Ed_kN: PositiveFloat = Field(description="Design axial compression force N_Ed in kN")
    M_y_Ed_kNm: float = Field(default=0.0, description="Design moment M_y,Ed in kNm")
    M_z_Ed_kNm: float = Field(default=0.0, description="Design moment M_z,Ed in kNm")

    # Resistances
    N_Rk_kN: PositiveFloat = Field(description="Characteristic axial resistance N_Rk in kN")
    M_pl_y_Rd_kNm: PositiveFloat = Field(description="Design plastic moment M_pl,y,Rd in kNm")
    M_pl_z_Rd_kNm: PositiveFloat = Field(description="Design plastic moment M_pl,z,Rd in kNm")
    gamma_M0: PositiveFloat = Field(default=1.0, description="Partial factor γ_M0")

    # Uniform moment factors from Table A.2
    C_my_0: float = Field(description="Uniform moment factor C_my,0 from Table A.2")
    C_mz_0: float = Field(description="Uniform moment factor C_mz,0 from Table A.2")

    # C1 factor
    C_1: PositiveFloat = Field(
        default=1.0, description="Factor C_1 depending on loading and end conditions"
    )


def calculate(inp: InteractionFactorsAInput) -> dict:
    N = float(inp.N_Ed_kN)
    Ncry = float(inp.N_cr_y_kN)
    Ncrz = float(inp.N_cr_z_kN)
    NcrT = float(inp.N_cr_T_kN) if inp.N_cr_T_kN else Ncrz
    NRk = float(inp.N_Rk_kN)
    gM0 = float(inp.gamma_M0)
    cy = float(inp.chi_y)
    cz = float(inp.chi_z)
    cLT = float(inp.chi_LT)

    ly = float(inp.lambda_bar_y)
    lz = float(inp.lambda_bar_z)
    l0 = float(inp.lambda_bar_0)
    lLT = float(inp.lambda_bar_LT)
    l_max = max(ly, lz)

    My = abs(float(inp.M_y_Ed_kNm))
    Mz = abs(float(inp.M_z_Ed_kNm))
    Mply = float(inp.M_pl_y_Rd_kNm)
    Mplz = float(inp.M_pl_z_Rd_kNm)

    Wply = float(inp.W_pl_y_cm3)
    Wely = float(inp.W_el_y_cm3)
    Wplz = float(inp.W_pl_z_cm3)
    Welz = float(inp.W_el_z_cm3)

    IT = float(inp.I_T_cm4) * 1e4  # mm⁴
    Iy = float(inp.I_y_cm4) * 1e4  # mm⁴

    Cmy0 = float(inp.C_my_0)
    Cmz0 = float(inp.C_mz_0)
    C1 = float(inp.C_1)

    notes: list[str] = []

    # ── Auxiliary parameters ──
    w_y = min(Wply / Wely, 1.5)
    w_z = min(Wplz / Welz, 1.5)
    n_pl = N / (NRk / gM0)
    a_LT = max(1.0 - IT / Iy, 0.0)  # use raw units ratio (cm⁴/cm⁴ cancels)
    # Correct a_LT using original cm⁴ values
    a_LT = max(1.0 - float(inp.I_T_cm4) / float(inp.I_y_cm4), 0.0)

    mu_y = (1.0 - N / Ncry) / (1.0 - cy * N / Ncry)
    mu_z = (1.0 - N / Ncrz) / (1.0 - cz * N / Ncrz)

    notes.append(f"w_y = min(W_pl,y/W_el,y, 1.5) = {w_y:.4f}")
    notes.append(f"w_z = min(W_pl,z/W_el,z, 1.5) = {w_z:.4f}")
    notes.append(f"n_pl = N_Ed/(N_Rk/γ_M0) = {n_pl:.4f}")
    notes.append(f"a_LT = max(1 − I_T/I_y, 0) = {a_LT:.4f}")
    notes.append(f"μ_y = {mu_y:.4f}, μ_z = {mu_z:.4f}")

    # ── C_my, C_mz, C_mLT ──
    threshold = 0.2 * math.sqrt(C1)

    if l0 <= threshold:
        C_my = Cmy0
        C_mz = Cmz0
        C_mLT = 1.0
        notes.append(f"λ̄_0 = {l0:.4f} ≤ 0.2√C1 = {threshold:.4f} → C_mLT = 1.0")
    else:
        # ε_y = M_y,Ed / N_Ed · A / W_el,y  (for class 1,2,3)
        if N > 0 and My > 0:
            eps_y = (My / N) * (NRk / gM0) / (Wely * 1000.0 / (float(inp.W_el_y_cm3)))
            # Simplified: ε_y = M_y,Ed/N_Ed · A/W_el,y
            # A = NRk/fy, we use n_pl relationship
            eps_y = (My * 1e6) / (N * 1000.0) * 1.0 / (Wely * 1000.0)  # approximate
            eps_y = max(eps_y, 0.0)
        else:
            eps_y = 0.0

        sqrt_ey = math.sqrt(max(eps_y, 0.0))
        C_my = Cmy0 + (1.0 - Cmy0) * sqrt_ey * a_LT / (1.0 + sqrt_ey * a_LT)
        C_mz = Cmz0

        # C_mLT
        nz_ratio = N / Ncrz
        nt_ratio = N / NcrT if NcrT > 0 else 0.0
        denom_inner = (1.0 - nz_ratio) / (1.0 - nt_ratio) if (1.0 - nt_ratio) > 0 else 1.0
        if denom_inner > 0:
            C_mLT = C_my**2 * a_LT / math.sqrt(denom_inner)
        else:
            C_mLT = 1.0
        C_mLT = max(C_mLT, 1.0)

        notes.append(f"λ̄_0 = {l0:.4f} > 0.2√C1 → modified C_my, C_mLT")
        notes.append(f"ε_y = {eps_y:.4f}, C_my = {C_my:.4f}, C_mLT = {C_mLT:.4f}")

    notes.append(f"C_my = {C_my:.4f}, C_mz = {C_mz:.4f}, C_mLT = {C_mLT:.4f}")

    # ── Auxiliary terms b_LT, c_LT, d_LT, e_LT ──
    My_ratio = My / (cLT * Mply) if cLT * Mply > 0 else 0.0
    Mz_ratio = Mz / Mplz if Mplz > 0 else 0.0

    b_LT = 0.5 * a_LT * l0**2 * My_ratio * Mz_ratio
    c_LT = 10.0 * a_LT * l0**2 * My / (C_my * cLT * Mply) if C_my * cLT * Mply > 0 else 0.0
    d_LT = (
        2.0 * a_LT * l0**2 * My / (C_my * cLT * Mply) * Mz / (C_mz * Mplz)
        if C_my * cLT * Mply * C_mz * Mplz > 0
        else 0.0
    )
    e_LT = 1.7 * a_LT * l0**2 * My / (C_my * cLT * Mply) if C_my * cLT * Mply > 0 else 0.0

    # ── C_yy, C_yz, C_zy, C_zz ──
    C_yy_calc = 1.0 + (w_y - 1.0) * (
        (2.0 - 1.6 / w_y * C_my**2 * l_max - 1.6 / w_y * C_my**2 * l_max**2) * n_pl
        - b_LT
    )
    C_yy = max(C_yy_calc, Wely / Wply)

    C_yz_calc = 1.0 + (w_z - 1.0) * (
        (2.0 - 14.0 * C_mz**2 * l_max**2 / w_z**5) * n_pl - c_LT
    )
    C_yz = max(C_yz_calc, 0.6 * math.sqrt(w_z / w_y) * Welz / Wplz)

    C_zy_calc = 1.0 + (w_y - 1.0) * (
        (2.0 - 14.0 * C_my**2 * l_max**2 / w_y**5) * n_pl - d_LT
    )
    C_zy = max(C_zy_calc, 0.6 * math.sqrt(w_y / w_z) * Wely / Wply)

    C_zz_calc = 1.0 + (w_z - 1.0) * (
        (2.0 - 1.6 / w_z * C_mz**2 * l_max - 1.6 / w_z * C_mz**2 * l_max**2) * n_pl
        - e_LT
    )
    C_zz = max(C_zz_calc, Welz / Wplz)

    notes.append(f"C_yy = {C_yy:.4f}, C_yz = {C_yz:.4f}")
    notes.append(f"C_zy = {C_zy:.4f}, C_zz = {C_zz:.4f}")

    # ── Interaction factors k_ij ──
    k_yy = C_my * C_mLT * mu_y / (1.0 - N / Ncry) / C_yy
    k_yz = C_mz * mu_y / (1.0 - N / Ncrz) / C_yz * 0.6 * math.sqrt(w_z / w_y)
    k_zy = C_my * C_mLT * mu_z / (1.0 - N / Ncry) / C_zy * 0.6 * math.sqrt(w_y / w_z)
    k_zz = C_mz * mu_z / (1.0 - N / Ncrz) / C_zz

    notes.append(f"k_yy = {k_yy:.4f}")
    notes.append(f"k_yz = {k_yz:.4f}")
    notes.append(f"k_zy = {k_zy:.4f}")
    notes.append(f"k_zz = {k_zz:.4f}")

    return {
        "inputs_used": {
            "section_class": inp.section_class,
            "lambda_bar_y": ly,
            "lambda_bar_z": lz,
            "lambda_bar_LT": lLT,
            "lambda_bar_0": l0,
            "chi_y": cy,
            "chi_z": cz,
            "chi_LT": cLT,
            "N_Ed_kN": N,
            "N_Rk_kN": NRk,
        },
        "intermediate": {
            "w_y": round(w_y, 4),
            "w_z": round(w_z, 4),
            "n_pl": round(n_pl, 4),
            "a_LT": round(a_LT, 4),
            "mu_y": round(mu_y, 4),
            "mu_z": round(mu_z, 4),
            "C_my": round(C_my, 4),
            "C_mz": round(C_mz, 4),
            "C_mLT": round(C_mLT, 4),
            "C_yy": round(C_yy, 4),
            "C_yz": round(C_yz, 4),
            "C_zy": round(C_zy, 4),
            "C_zz": round(C_zz, 4),
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
                "clause_id": "A.1",
                "title": "Method 1 – Interaction factors k_ij (Table A.1)",
                "pointer": "en_1993_1_1_2005_structured.json#A.1",
            },
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "A.2",
                "title": "Equivalent uniform moment factors C_mi,0 (Table A.2)",
                "pointer": "en_1993_1_1_2005_structured.json#A.2",
            },
        ],
        "notes": notes,
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=InteractionFactorsAInput, handler=calculate)
