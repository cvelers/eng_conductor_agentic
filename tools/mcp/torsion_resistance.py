from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli
from tools.mcp.section_library import steel_grade_to_fy

TOOL_NAME = "torsion_resistance_ec3"


class TorsionResistanceInput(BaseModel):
    section_type: Literal["open", "hollow"] = Field(description="'open' (I/H/channel) or 'hollow' (RHS/CHS)")
    steel_grade: str = Field(default="S355", description="Steel grade, e.g. S355")
    fy_mpa: Optional[PositiveFloat] = Field(default=None, description="Yield strength in MPa")
    gamma_M0: PositiveFloat = Field(default=1.0, description="Partial factor γ_M0")

    # For hollow sections: uniform torsion dominates
    T_Ed_kNm: float = Field(default=0.0, description="Design torsional moment T_Ed in kNm")
    V_Ed_kN: float = Field(default=0.0, description="Design shear force V_Ed in kN (for combined check)")

    # Hollow section params
    A_k_cm2: Optional[PositiveFloat] = Field(default=None, description="Area enclosed by centre-line of thin walls in cm²")
    t_min_mm: Optional[PositiveFloat] = Field(default=None, description="Minimum wall thickness in mm (for τ_t,Ed)")

    # Plastic shear resistance (from §6.2.6)
    V_pl_Rd_kN: Optional[PositiveFloat] = Field(default=None, description="Plastic shear resistance V_pl,Rd in kN")

    # Open section – St. Venant torsion capacity
    W_T_cm3: Optional[PositiveFloat] = Field(
        default=None,
        description="St. Venant torsional section modulus in cm³ (= I_T / t_max for open sections)",
    )


def calculate(inp: TorsionResistanceInput) -> dict:
    fy = float(inp.fy_mpa) if inp.fy_mpa else steel_grade_to_fy(inp.steel_grade)
    gamma_M0 = float(inp.gamma_M0)
    tau_Rd = fy / (math.sqrt(3) * gamma_M0)  # MPa – design shear strength

    T_Ed = float(inp.T_Ed_kNm)
    V_Ed = float(inp.V_Ed_kN)

    results: dict = {
        "tau_Rd_mpa": round(tau_Rd, 2),
    }
    notes: list[str] = []

    if inp.section_type == "hollow":
        # §6.2.7(5) – Shear stress from uniform torsion in hollow section
        if inp.A_k_cm2 is None or inp.t_min_mm is None:
            raise ValueError("Hollow sections require A_k_cm2 and t_min_mm.")
        A_k_mm2 = float(inp.A_k_cm2) * 100.0
        t_min = float(inp.t_min_mm)

        # τ_t,Ed = T_Ed / (2·Ak·t)
        tau_t_Ed = abs(T_Ed) * 1e6 / (2.0 * A_k_mm2 * t_min)  # MPa
        results["tau_t_Ed_mpa"] = round(tau_t_Ed, 2)

        # T_Rd from shear stress limit
        T_Rd_kNm = tau_Rd * 2.0 * A_k_mm2 * t_min / 1e6
        results["T_Rd_kNm"] = round(T_Rd_kNm, 2)

        torsion_util = abs(T_Ed) / T_Rd_kNm if T_Rd_kNm > 0 else float("inf")
        results["torsion_utilization"] = round(torsion_util, 4)
        results["torsion_pass"] = torsion_util <= 1.0

        # §6.2.7(6) – Combined shear + torsion for hollow sections
        if inp.V_pl_Rd_kN and V_Ed > 0:
            V_pl_Rd = float(inp.V_pl_Rd_kN)
            # Reduced shear resistance: V_pl,T,Rd = V_pl,Rd · √(1 - (τ_t,Ed / (fy/(√3·γM0)))²)
            tau_ratio = tau_t_Ed / tau_Rd
            if tau_ratio >= 1.0:
                V_pl_T_Rd = 0.0
            else:
                V_pl_T_Rd = V_pl_Rd * math.sqrt(1.0 - tau_ratio**2)
            results["V_pl_T_Rd_kN"] = round(V_pl_T_Rd, 2)
            combined_util = V_Ed / V_pl_T_Rd if V_pl_T_Rd > 0 else float("inf")
            results["combined_shear_torsion_utilization"] = round(combined_util, 4)
            results["combined_pass"] = combined_util <= 1.0
            notes.append(
                f"V_pl,T,Rd = V_pl,Rd·√(1−(τ_t,Ed/τ_Rd)²) = {V_pl_Rd:.1f}·√(1−{tau_ratio:.3f}²) = {V_pl_T_Rd:.2f} kN"
            )

    else:
        # Open section: §6.2.7(1) – T_Ed / T_Rd ≤ 1.0
        if inp.W_T_cm3 is None:
            raise ValueError("Open sections require W_T_cm3 (torsional section modulus).")
        W_T_mm3 = float(inp.W_T_cm3) * 1000.0

        # T_Rd = W_T · τ_Rd (elastic St. Venant)
        T_Rd_kNm = W_T_mm3 * tau_Rd / 1e6
        results["T_Rd_kNm"] = round(T_Rd_kNm, 2)

        torsion_util = abs(T_Ed) / T_Rd_kNm if T_Rd_kNm > 0 else float("inf")
        results["torsion_utilization"] = round(torsion_util, 4)
        results["torsion_pass"] = torsion_util <= 1.0

        # §6.2.7(7) – Combined shear + torsion for open sections (I/H)
        if inp.V_pl_Rd_kN and V_Ed > 0:
            V_pl_Rd = float(inp.V_pl_Rd_kN)
            # Reduced: V_pl,T,Rd = V_pl,Rd · (1 - τ_t,Ed/τ_Rd)  (for I/H sections)
            tau_t_Ed = abs(T_Ed) * 1e6 / W_T_mm3 if W_T_mm3 > 0 else 0
            results["tau_t_Ed_mpa"] = round(tau_t_Ed, 2)
            tau_ratio = tau_t_Ed / tau_Rd
            V_pl_T_Rd = V_pl_Rd * max(1.0 - tau_ratio, 0.0)
            results["V_pl_T_Rd_kN"] = round(V_pl_T_Rd, 2)
            combined_util = V_Ed / V_pl_T_Rd if V_pl_T_Rd > 0 else float("inf")
            results["combined_shear_torsion_utilization"] = round(combined_util, 4)
            results["combined_pass"] = combined_util <= 1.0

    notes.insert(0, f"τ_Rd = fy/(√3·γM0) = {fy:.0f}/(√3·{gamma_M0}) = {tau_Rd:.2f} MPa")

    return {
        "inputs_used": {
            "section_type": inp.section_type,
            "steel_grade": inp.steel_grade,
            "fy_mpa": fy,
            "gamma_M0": gamma_M0,
            "T_Ed_kNm": T_Ed,
            "V_Ed_kN": V_Ed,
        },
        "outputs": results,
        "clause_references": [
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "6.2.7",
                "title": "Torsion",
                "pointer": "en_1993_1_1_2005_structured.json#6.2.7",
            },
        ],
        "notes": notes,
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=TorsionResistanceInput, handler=calculate)
