from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveFloat, model_validator

from tools.mcp.cli import run_cli
from tools.mcp.section_library import SECTION_LIBRARY, steel_grade_to_fy
from tools.mcp.clause_ref import clause_ref

TOOL_NAME = "bending_axial_interaction_ec3"


class BendingAxialInput(BaseModel):
    N_Ed_kN: float = Field(description="Design axial force N_Ed in kN (positive = compression)")
    M_y_Ed_kNm: float = Field(default=0.0, description="Design bending moment about y-y axis M_y,Ed in kNm")
    M_z_Ed_kNm: float = Field(default=0.0, description="Design bending moment about z-z axis M_z,Ed in kNm")

    section_name: Optional[str] = Field(default=None, description="Section name, e.g. IPE300")
    section_type: Literal["I_H", "rectangular", "circular", "other"] = Field(
        default="I_H", description="Cross-section type"
    )
    steel_grade: str = Field(default="S355", description="Steel grade, e.g. S355")
    fy_mpa: Optional[PositiveFloat] = Field(default=None, description="Yield strength in MPa")
    gamma_M0: PositiveFloat = Field(default=1.0, description="Partial factor γ_M0")
    section_class: int = Field(default=1, ge=1, le=2, description="Cross-section class (1 or 2 for plastic)")

    area_cm2: Optional[PositiveFloat] = Field(default=None, description="Gross area A in cm²")
    wpl_y_cm3: Optional[PositiveFloat] = Field(default=None, description="Plastic modulus W_pl,y in cm³")
    wpl_z_cm3: Optional[PositiveFloat] = Field(default=None, description="Plastic modulus W_pl,z in cm³")
    h_mm: Optional[PositiveFloat] = Field(default=None, description="Section height h in mm")
    b_mm: Optional[PositiveFloat] = Field(default=None, description="Flange width b in mm")
    tf_mm: Optional[PositiveFloat] = Field(default=None, description="Flange thickness t_f in mm")
    tw_mm: Optional[PositiveFloat] = Field(default=None, description="Web thickness t_w in mm")

    @model_validator(mode="after")
    def fill_from_library(self) -> "BendingAxialInput":
        if self.section_name:
            key = self.section_name.upper().replace(" ", "")
            if key in SECTION_LIBRARY:
                row = SECTION_LIBRARY[key]
                if self.area_cm2 is None:
                    self.area_cm2 = float(row["area_cm2"])
                if self.wpl_y_cm3 is None:
                    self.wpl_y_cm3 = float(row["wpl_y_cm3"])
                if self.h_mm is None:
                    self.h_mm = float(row.get("h_mm", 0))
                if self.b_mm is None:
                    self.b_mm = float(row.get("b_mm", 0))
                if self.tf_mm is None:
                    self.tf_mm = float(row.get("tf_mm", 0))
                if self.tw_mm is None:
                    self.tw_mm = float(row.get("tw_mm", 0))
        if self.fy_mpa is None:
            self.fy_mpa = steel_grade_to_fy(self.steel_grade)
        return self


def calculate(inp: BendingAxialInput) -> dict:
    fy = float(inp.fy_mpa)
    gamma_M0 = float(inp.gamma_M0)
    N_Ed = abs(float(inp.N_Ed_kN))
    M_y_Ed = abs(float(inp.M_y_Ed_kNm))
    M_z_Ed = abs(float(inp.M_z_Ed_kNm))

    if inp.area_cm2 is None or inp.wpl_y_cm3 is None:
        raise ValueError("Provide section_name or area_cm2 and wpl_y_cm3.")

    A_mm2 = float(inp.area_cm2) * 100.0
    Wpl_y_mm3 = float(inp.wpl_y_cm3) * 1000.0

    N_pl_Rd = A_mm2 * fy / gamma_M0 / 1000.0  # kN
    M_pl_y_Rd = Wpl_y_mm3 * fy / gamma_M0 / 1e6  # kNm

    n = N_Ed / N_pl_Rd  # axial ratio

    notes: list[str] = []
    results: dict = {
        "N_pl_Rd_kN": round(N_pl_Rd, 2),
        "M_pl_y_Rd_kNm": round(M_pl_y_Rd, 2),
        "n": round(n, 4),
    }

    if inp.section_type == "rectangular":
        # §6.2.9.1(3) – Rectangular solid section
        M_N_y_Rd = M_pl_y_Rd * (1.0 - n**2)
        notes.append(f"Rectangular: M_N,Rd = M_pl,Rd·[1 − n²] = {M_pl_y_Rd:.2f}·[1 − {n:.4f}²] = {M_N_y_Rd:.2f} kNm")

    elif inp.section_type == "I_H":
        # §6.2.9.1(5) – I/H sections
        if inp.h_mm is None or inp.b_mm is None or inp.tf_mm is None or inp.tw_mm is None:
            raise ValueError("I/H sections require h_mm, b_mm, tf_mm, tw_mm.")

        h = float(inp.h_mm)
        b = float(inp.b_mm)
        tf = float(inp.tf_mm)
        tw = float(inp.tw_mm)

        # a = (A - 2·b·tf) / A ≤ 0.5
        a = min((A_mm2 - 2.0 * b * tf) / A_mm2, 0.5)
        results["a"] = round(a, 4)

        # M_N,y,Rd = M_pl,y,Rd · (1 - n) / (1 - 0.5·a) but ≤ M_pl,y,Rd
        if n <= a:
            M_N_y_Rd = M_pl_y_Rd
            notes.append(f"n ≤ a ({n:.4f} ≤ {a:.4f}) → M_N,y,Rd = M_pl,y,Rd = {M_pl_y_Rd:.2f} kNm")
        else:
            M_N_y_Rd = min(M_pl_y_Rd * (1.0 - n) / (1.0 - 0.5 * a), M_pl_y_Rd)
            notes.append(
                f"M_N,y,Rd = M_pl,y,Rd·(1−n)/(1−0.5a) = {M_pl_y_Rd:.2f}·(1−{n:.4f})/(1−0.5·{a:.4f}) = {M_N_y_Rd:.2f} kNm"
            )

        # z-z axis if M_z_Ed provided
        if M_z_Ed > 0 and inp.wpl_z_cm3:
            Wpl_z_mm3 = float(inp.wpl_z_cm3) * 1000.0
            M_pl_z_Rd = Wpl_z_mm3 * fy / gamma_M0 / 1e6
            results["M_pl_z_Rd_kNm"] = round(M_pl_z_Rd, 2)

            # M_N,z,Rd (simplified for I/H):
            # For n ≤ a: M_N,z,Rd = M_pl,z,Rd
            # For n > a: M_N,z,Rd = M_pl,z,Rd · [1 - ((n-a)/(1-a))²]
            if n <= a:
                M_N_z_Rd = M_pl_z_Rd
            else:
                M_N_z_Rd = M_pl_z_Rd * (1.0 - ((n - a) / (1.0 - a)) ** 2)
            results["M_N_z_Rd_kNm"] = round(M_N_z_Rd, 2)

            # §6.2.9.1(6) – Biaxial bending interaction
            # [M_y,Ed / M_N,y,Rd]^α + [M_z,Ed / M_N,z,Rd]^β ≤ 1.0
            # For I/H: α = 2, β = 5n but β ≥ 1
            alpha_exp = 2.0
            beta_exp = max(5.0 * n, 1.0)
            results["alpha"] = alpha_exp
            results["beta"] = round(beta_exp, 4)

            biaxial_util = 0.0
            if M_N_y_Rd > 0:
                biaxial_util += (M_y_Ed / M_N_y_Rd) ** alpha_exp
            if M_N_z_Rd > 0:
                biaxial_util += (M_z_Ed / M_N_z_Rd) ** beta_exp

            results["biaxial_utilization"] = round(biaxial_util, 4)
            results["biaxial_pass"] = biaxial_util <= 1.0

    elif inp.section_type == "circular":
        # §6.2.9.1(4) – Circular hollow sections
        M_N_y_Rd = M_pl_y_Rd * (1.0 - n**1.7)
        notes.append(f"Circular: M_N,Rd = M_pl,Rd·[1 − n^1.7] = {M_N_y_Rd:.2f} kNm")

    else:
        # Conservative: linear interaction
        M_N_y_Rd = M_pl_y_Rd * (1.0 - n)
        notes.append(f"General: M_N,Rd = M_pl,Rd·(1 − n) = {M_N_y_Rd:.2f} kNm (conservative)")

    results["M_N_y_Rd_kNm"] = round(M_N_y_Rd, 2)

    # Uniaxial utilization for y-y
    util_y = M_y_Ed / M_N_y_Rd if M_N_y_Rd > 0 else float("inf")
    results["utilization_y"] = round(util_y, 4)
    results["pass_y"] = util_y <= 1.0

    return {
        "inputs_used": {
            "N_Ed_kN": float(inp.N_Ed_kN),
            "M_y_Ed_kNm": float(inp.M_y_Ed_kNm),
            "M_z_Ed_kNm": float(inp.M_z_Ed_kNm),
            "section_name": inp.section_name,
            "section_type": inp.section_type,
            "steel_grade": inp.steel_grade,
            "fy_mpa": fy,
            "gamma_M0": gamma_M0,
        },
        "intermediate": {
            "N_pl_Rd_kN": round(N_pl_Rd, 2),
            "n": round(n, 4),
        },
        "outputs": results,
        "clause_references": [
            clause_ref("ec3.en1993-1-1.2005", "6.2.9.1", "Bending and axial force – Class 1 and 2 cross-sections"),
        ],
        "notes": notes,
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=BendingAxialInput, handler=calculate)
