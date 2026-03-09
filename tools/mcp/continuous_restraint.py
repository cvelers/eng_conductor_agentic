from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli
from tools.mcp.clause_ref import clause_ref

TOOL_NAME = "continuous_restraint_ec3"


class ContinuousRestraintInput(BaseModel):
    """Input for BB.2 – Continuous restraints from sheeting."""

    check_type: Literal["lateral", "torsional"] = Field(
        description="'lateral' for BB.2.1 (lateral restraint from sheeting), "
        "'torsional' for BB.2.2 (torsional restraint from sheeting)"
    )

    # ── BB.2.1 – Lateral restraint ──
    S_stiffness: Optional[PositiveFloat] = Field(
        default=None,
        description="Shear stiffness S per unit beam length provided by sheeting (N/mm)",
    )
    I_w_cm6: Optional[float] = Field(default=None, description="Warping constant I_w in cm⁶")
    I_T_cm4: Optional[PositiveFloat] = Field(default=None, description="Torsion constant I_T in cm⁴")
    I_z_cm4: Optional[PositiveFloat] = Field(default=None, description="Second moment I_z in cm⁴")
    L_mm: Optional[PositiveFloat] = Field(default=None, description="Beam length L in mm")
    h_mm: Optional[PositiveFloat] = Field(default=None, description="Beam depth h in mm")
    connected_every_rib: bool = Field(
        default=True,
        description="True if sheeting connected at every rib, False if at every second rib (use 0.20·S)",
    )

    # ── BB.2.2 – Torsional restraint ──
    C_theta_k: Optional[PositiveFloat] = Field(
        default=None,
        description="Rotational stiffness C_θ,k per unit length provided to beam (Nmm/rad/mm)",
    )
    M_pl_k_kNm: Optional[PositiveFloat] = Field(
        default=None,
        description="Characteristic plastic moment of the beam M_pl,k in kNm",
    )
    analysis_type: Literal["elastic", "plastic"] = Field(
        default="elastic",
        description="'elastic' (K_v=0.35) or 'plastic' (K_v=1.0)",
    )
    K_theta: PositiveFloat = Field(
        default=4.0,
        description="Factor K_θ for moment distribution and restraint type (Table BB.1, default 4.0)",
    )


def calculate(inp: ContinuousRestraintInput) -> dict:
    E = 210000.0  # MPa
    G = 81000.0  # MPa
    notes: list[str] = []

    if inp.check_type == "lateral":
        # BB.2.1 – Check if S ≥ S_required
        if inp.S_stiffness is None or inp.L_mm is None or inp.h_mm is None:
            raise ValueError("lateral check requires S_stiffness, L_mm, h_mm.")
        if inp.I_w_cm6 is None or inp.I_T_cm4 is None or inp.I_z_cm4 is None:
            raise ValueError("lateral check requires I_w_cm6, I_T_cm4, I_z_cm4.")

        S = float(inp.S_stiffness)
        L = float(inp.L_mm)
        h = float(inp.h_mm)
        Iw = float(inp.I_w_cm6) * 1e6  # mm⁶
        IT = float(inp.I_T_cm4) * 1e4  # mm⁴
        Iz = float(inp.I_z_cm4) * 1e4  # mm⁴

        if not inp.connected_every_rib:
            S_eff = 0.20 * S
            notes.append(f"Connected at every second rib: S_eff = 0.20·S = {S_eff:.2f}")
        else:
            S_eff = S

        # S_required = (E·I_w·π²/L² + G·I_T + E·I_z·π²/L²·0.25·h²) · 70/h²
        pi2_L2 = math.pi**2 / L**2
        bracket = E * Iw * pi2_L2 + G * IT + E * Iz * pi2_L2 * 0.25 * h**2
        S_required = bracket * 70.0 / h**2

        is_restrained = S_eff >= S_required

        notes.append(f"S_required = {S_required:.2f} N/mm")
        notes.append(f"S_eff = {S_eff:.2f} N/mm")
        notes.append(
            f"S_eff {'≥' if is_restrained else '<'} S_required → "
            f"{'Laterally restrained' if is_restrained else 'NOT laterally restrained'}"
        )

        return {
            "inputs_used": {
                "check_type": "lateral",
                "S_stiffness": S,
                "L_mm": L,
                "h_mm": h,
                "connected_every_rib": inp.connected_every_rib,
            },
            "outputs": {
                "S_effective": round(S_eff, 2),
                "S_required": round(S_required, 2),
                "is_restrained": is_restrained,
            },
            "clause_references": [
                clause_ref("ec3.en1993-1-1.2005", "BB.2.1", "Continuous lateral restraints"),
            ],
            "notes": notes,
        }

    else:
        # BB.2.2 – Check if C_θ,k > C_θ,k,required
        if inp.C_theta_k is None or inp.M_pl_k_kNm is None or inp.I_z_cm4 is None:
            raise ValueError("torsional check requires C_theta_k, M_pl_k_kNm, I_z_cm4.")

        Ctheta = float(inp.C_theta_k)
        Mpl = float(inp.M_pl_k_kNm) * 1e6  # Nmm
        Iz = float(inp.I_z_cm4) * 1e4  # mm⁴

        K_v = 0.35 if inp.analysis_type == "elastic" else 1.0
        K_theta = float(inp.K_theta)

        # C_θ,k,required = M_pl,k² / (E·I_z) · K_θ · K_v
        C_required = Mpl**2 / (E * Iz) * K_theta * K_v

        is_sufficient = Ctheta > C_required

        notes.append(f"K_v = {K_v} ({inp.analysis_type} analysis)")
        notes.append(f"K_θ = {K_theta} (Table BB.1)")
        notes.append(f"C_θ,k,required = M_pl,k²/(E·I_z)·K_θ·K_v = {C_required:.2f} Nmm/rad/mm")
        notes.append(f"C_θ,k = {Ctheta:.2f} Nmm/rad/mm")
        notes.append(
            f"C_θ,k {'>' if is_sufficient else '≤'} C_required → "
            f"{'Torsionally restrained' if is_sufficient else 'NOT torsionally restrained'}"
        )

        return {
            "inputs_used": {
                "check_type": "torsional",
                "C_theta_k": Ctheta,
                "M_pl_k_kNm": float(inp.M_pl_k_kNm),
                "analysis_type": inp.analysis_type,
                "K_theta": K_theta,
                "K_v": K_v,
            },
            "outputs": {
                "C_theta_required": round(C_required, 2),
                "C_theta_provided": round(Ctheta, 2),
                "is_sufficient": is_sufficient,
            },
            "clause_references": [
                clause_ref("ec3.en1993-1-1.2005", "BB.2.2", "Continuous torsional restraints"),
            ],
            "notes": notes,
        }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=ContinuousRestraintInput, handler=calculate)
