from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, Field, PositiveFloat, model_validator

from tools.mcp.cli import run_cli
from tools.mcp.section_library import SECTION_LIBRARY, steel_grade_to_fy

TOOL_NAME = "bending_shear_interaction_ec3"


class BendingShearInput(BaseModel):
    M_Ed_kNm: float = Field(description="Design bending moment M_Ed in kNm")
    V_Ed_kN: float = Field(description="Design shear force V_Ed in kN")

    # Provide section_name to auto-compute all section properties and resistances.
    section_name: Optional[str] = Field(
        default=None,
        description="Section name, e.g. IPE500. Auto-computes V_pl,Rd, M_c,Rd, Aw, tw.",
    )
    V_pl_Rd_kN: Optional[PositiveFloat] = Field(
        default=None,
        description="Design plastic shear resistance V_pl,Rd in kN. Computed from section if section_name given.",
    )
    M_c_Rd_kNm: Optional[PositiveFloat] = Field(
        default=None,
        description="Design moment resistance M_c,Rd in kNm (unreduced). Computed from section if section_name given.",
    )

    steel_grade: str = Field(default="S355", description="Steel grade, e.g. S355")
    fy_mpa: Optional[PositiveFloat] = Field(default=None, description="Yield strength in MPa")
    gamma_M0: PositiveFloat = Field(default=1.0, description="Partial factor γ_M0")

    # For exact I/H reduced-moment formula per EC3 6.2.8(3)
    wpl_y_cm3: Optional[PositiveFloat] = Field(
        default=None, description="Plastic section modulus W_pl,y in cm³"
    )
    A_w_cm2: Optional[PositiveFloat] = Field(
        default=None,
        description="Web area Aw = hw·tw in cm² where hw = h − 2tf (for I/H sections)",
    )
    tw_mm: Optional[PositiveFloat] = Field(
        default=None,
        description="Web thickness tw in mm (required for exact I/H 6.2.8(3) formula)",
    )
    section_class: int = Field(default=2, ge=1, le=4, description="Cross-section class (1-4)")

    @model_validator(mode="after")
    def fill_from_section(self) -> "BendingShearInput":
        if self.section_name:
            key = self.section_name.upper().replace(" ", "")
            if key in SECTION_LIBRARY:
                row = SECTION_LIBRARY[key]
                fy = self.fy_mpa or steel_grade_to_fy(self.steel_grade)
                gamma = float(self.gamma_M0)

                h = float(row.get("h_mm", 0))
                b = float(row.get("b_mm", 0))
                tf = float(row.get("tf_mm", 0))
                tw = float(row.get("tw_mm", 0))
                r = float(row.get("r_mm", 0))
                area_mm2 = float(row.get("area_cm2", 0)) * 100.0
                wpl = float(row.get("wpl_y_cm3", 0))

                # Web thickness (for EC3 6.2.8(3) formula denominator)
                if self.tw_mm is None and tw > 0:
                    self.tw_mm = tw

                # Web area Aw = hw × tw per EC3 6.2.8(3) (hw = clear depth between flanges)
                hw = h - 2.0 * tf
                if self.A_w_cm2 is None and hw > 0 and tw > 0:
                    self.A_w_cm2 = round(hw * tw / 100.0, 4)

                if self.wpl_y_cm3 is None and wpl > 0:
                    self.wpl_y_cm3 = wpl

                # Shear area per EC3 6.2.6(3) for rolled I/H: Av = A − 2b·tf + (tw + 2r)·tf
                if self.V_pl_Rd_kN is None and area_mm2 > 0 and b > 0 and tf > 0:
                    Av_mm2 = area_mm2 - 2.0 * b * tf + (tw + 2.0 * r) * tf
                    Av_mm2 = max(Av_mm2, 1.0)  # guard against bad data
                    self.V_pl_Rd_kN = Av_mm2 * fy / (math.sqrt(3.0) * gamma) / 1000.0

                # Moment resistance M_c,Rd = Wpl · fy / γM0 (Class 1/2)
                if self.M_c_Rd_kNm is None and wpl > 0 and self.section_class <= 2:
                    self.M_c_Rd_kNm = wpl * 1000.0 * fy / (gamma * 1.0e6)

        if self.V_pl_Rd_kN is None:
            raise ValueError(
                "V_pl_Rd_kN is required. Provide it directly or supply section_name for auto-computation."
            )
        if self.M_c_Rd_kNm is None:
            raise ValueError(
                "M_c_Rd_kNm is required. Provide it directly or supply section_name for auto-computation."
            )
        return self


def calculate(inp: BendingShearInput) -> dict:
    fy = float(inp.fy_mpa) if inp.fy_mpa else steel_grade_to_fy(inp.steel_grade)
    gamma_M0 = float(inp.gamma_M0)

    V_Ed = abs(float(inp.V_Ed_kN))
    M_Ed = abs(float(inp.M_Ed_kNm))
    V_pl_Rd = float(inp.V_pl_Rd_kN)
    M_c_Rd = float(inp.M_c_Rd_kNm)

    # §6.2.8(2) – Check if shear exceeds 50% of V_pl,Rd
    shear_ratio = V_Ed / V_pl_Rd
    high_shear = shear_ratio > 0.5

    if not high_shear:
        # §6.2.8(2) – No reduction needed
        M_V_Rd = M_c_Rd
        rho = 0.0
        notes = ["V_Ed ≤ 0.5·V_pl,Rd → no moment reduction required."]
    else:
        # §6.2.8(3) – Reduced moment resistance
        rho = (2.0 * V_Ed / V_pl_Rd - 1.0) ** 2
        notes = [f"V_Ed > 0.5·V_pl,Rd → ρ = (2·V_Ed/V_pl,Rd − 1)² = {rho:.6f}"]

        if inp.wpl_y_cm3 and inp.A_w_cm2 and inp.tw_mm and inp.section_class <= 2:
            # §6.2.8(3) for doubly-symmetric I/H Class 1/2:
            # M_V,Rd = [Wpl,y − ρ·Aw²/(4·tw)] · fy/γM0
            Wpl_mm3 = float(inp.wpl_y_cm3) * 1000.0
            Aw_mm2 = float(inp.A_w_cm2) * 100.0
            tw = float(inp.tw_mm)
            # Aw²/(4·tw) has units mm³, subtracted from Wpl in mm³
            reduced_wpl_mm3 = Wpl_mm3 - rho * Aw_mm2 ** 2 / (4.0 * tw)
            M_V_Rd = max(reduced_wpl_mm3 * fy / (gamma_M0 * 1.0e6), 0.0)
            M_V_Rd = min(M_V_Rd, M_c_Rd)
            notes.append(
                f"I/H Class 1/2: M_V,Rd = (Wpl − ρ·Aw²/(4tw))·fy/γM0"
                f" = ({Wpl_mm3:.0f} − {rho:.6f}·{Aw_mm2:.1f}²/(4·{tw}))·{fy}/{gamma_M0}/1e6"
                f" = {M_V_Rd:.2f} kNm"
            )
        else:
            # General approach: M_V,Rd = M_c,Rd · (1 − ρ)
            M_V_Rd = M_c_Rd * (1.0 - rho)
            notes.append(
                f"General: M_V,Rd = M_c,Rd·(1−ρ) = {M_c_Rd:.2f}·{1.0 - rho:.4f} = {M_V_Rd:.2f} kNm"
            )

    utilization = M_Ed / M_V_Rd if M_V_Rd > 0 else float("inf")

    return {
        "inputs_used": {
            "section_name": inp.section_name,
            "M_Ed_kNm": float(inp.M_Ed_kNm),
            "V_Ed_kN": float(inp.V_Ed_kN),
            "V_pl_Rd_kN": V_pl_Rd,
            "M_c_Rd_kNm": M_c_Rd,
            "steel_grade": inp.steel_grade,
            "fy_mpa": fy,
            "wpl_y_cm3": float(inp.wpl_y_cm3) if inp.wpl_y_cm3 else None,
            "A_w_cm2": float(inp.A_w_cm2) if inp.A_w_cm2 else None,
            "tw_mm": float(inp.tw_mm) if inp.tw_mm else None,
        },
        "intermediate": {
            "shear_ratio_V_Ed_over_V_pl_Rd": round(shear_ratio, 4),
            "high_shear": high_shear,
            "rho": round(rho, 6),
        },
        "outputs": {
            "M_V_Rd_kNm": round(M_V_Rd, 2),
            "utilization": round(utilization, 4),
            "pass": utilization <= 1.0,
        },
        "clause_references": [
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "6.2.8",
                "title": "Bending and shear",
                "pointer": "en_1993_1_1_2005_structured.json#6.2.8",
            },
        ],
        "notes": notes,
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=BendingShearInput, handler=calculate)
