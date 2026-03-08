from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveFloat, model_validator

from tools.mcp.cli import run_cli
from tools.mcp.section_library import SECTION_LIBRARY, steel_grade_to_fy

TOOL_NAME = "ltb_resistance_ec3"

# EC3 Table 6.3 – Imperfection factors for LTB curves
LTB_IMPERFECTION = {
    "a": 0.21,
    "b": 0.34,
    "c": 0.49,
    "d": 0.76,
}

# EC3 Table 6.4 – Recommended LTB curves for rolled sections (general case)
# h/b ≤ 2 → curve a; h/b > 2 → curve b
# For welded sections: h/b ≤ 2 → curve c; h/b > 2 → curve d


class LTBResistanceInput(BaseModel):
    section_name: Optional[str] = Field(default=None, description="Section name, e.g. IPE300")
    steel_grade: str = Field(default="S355", description="Steel grade, e.g. S355")
    fy_mpa: Optional[PositiveFloat] = Field(default=None, description="Yield strength in MPa")
    gamma_M1: PositiveFloat = Field(default=1.0, description="Partial factor γ_M1")
    section_class: int = Field(default=1, ge=1, le=4, description="Cross-section class (1-4)")

    # Buckling parameters
    M_cr_kNm: Optional[PositiveFloat] = Field(
        default=None,
        description="Elastic critical moment for LTB M_cr in kNm. If not provided, calculated from section properties.",
    )
    ltb_curve: Literal["a", "b", "c", "d"] = Field(
        default="b", description="LTB curve per EC3 Table 6.4"
    )
    method: Literal["general", "rolled_welded"] = Field(
        default="general",
        description="'general' (§6.3.2.2) or 'rolled_welded' (§6.3.2.3) method",
    )

    # Section properties
    wpl_y_cm3: Optional[PositiveFloat] = Field(default=None, description="W_pl,y in cm³ (Class 1/2)")
    wel_y_cm3: Optional[PositiveFloat] = Field(default=None, description="W_el,y in cm³ (Class 3/4)")
    h_mm: Optional[PositiveFloat] = Field(default=None, description="Section height h in mm")
    b_mm: Optional[PositiveFloat] = Field(default=None, description="Flange width b in mm")

    # For M_cr calculation (if M_cr not directly provided)
    L_cr_m: Optional[PositiveFloat] = Field(default=None, description="Effective length for LTB in m")
    I_z_cm4: Optional[PositiveFloat] = Field(default=None, description="Second moment about z-z in cm⁴")
    I_w_cm6: Optional[float] = Field(default=None, description="Warping constant I_w in cm⁶")
    I_T_cm4: Optional[float] = Field(default=None, description="St. Venant torsion constant I_T in cm⁴")
    C1: PositiveFloat = Field(default=1.0, description="Moment distribution factor C1 (1.0 for uniform moment)")

    # For rolled/welded method §6.3.2.3
    lambda_LT_0: float = Field(default=0.4, description="Plateau length λ_LT,0 (default 0.4 for rolled/welded)")
    beta_LT: float = Field(default=0.75, description="β factor (default 0.75 for rolled/welded)")

    @model_validator(mode="after")
    def fill_from_library(self) -> "LTBResistanceInput":
        if self.section_name:
            key = self.section_name.upper().replace(" ", "")
            if key in SECTION_LIBRARY:
                row = SECTION_LIBRARY[key]
                if self.wpl_y_cm3 is None:
                    self.wpl_y_cm3 = float(row.get("wpl_y_cm3", 0))
                if self.wel_y_cm3 is None:
                    self.wel_y_cm3 = float(row.get("wel_y_cm3", 0))
                if self.h_mm is None:
                    self.h_mm = float(row.get("h_mm", 0))
                if self.b_mm is None:
                    self.b_mm = float(row.get("b_mm", 0))
                if self.I_z_cm4 is None:
                    self.I_z_cm4 = float(row.get("I_z_cm4", 0)) or None
        if self.fy_mpa is None:
            self.fy_mpa = steel_grade_to_fy(self.steel_grade)
        return self


def calculate(inp: LTBResistanceInput) -> dict:
    fy = float(inp.fy_mpa)
    gamma_M1 = float(inp.gamma_M1)

    # Determine appropriate section modulus
    if inp.section_class <= 2:
        if inp.wpl_y_cm3 is None:
            raise ValueError("Class 1/2 requires wpl_y_cm3.")
        W_y_mm3 = float(inp.wpl_y_cm3) * 1000.0
        w_label = "W_pl,y"
    else:
        if inp.wel_y_cm3 is None:
            raise ValueError("Class 3/4 requires wel_y_cm3.")
        W_y_mm3 = float(inp.wel_y_cm3) * 1000.0
        w_label = "W_el,y"

    E = 210000.0  # MPa
    G = 81000.0  # MPa

    # Calculate M_cr if not provided
    M_cr_Nmm: float
    if inp.M_cr_kNm is not None:
        M_cr_Nmm = float(inp.M_cr_kNm) * 1e6
    elif inp.L_cr_m and inp.I_z_cm4 and inp.I_T_cm4 is not None:
        L = float(inp.L_cr_m) * 1000.0  # mm
        I_z = float(inp.I_z_cm4) * 1e4  # mm⁴
        I_T = float(inp.I_T_cm4) * 1e4  # mm⁴
        I_w = float(inp.I_w_cm6) * 1e6 if inp.I_w_cm6 else 0.0  # mm⁶
        C1 = float(inp.C1)

        # M_cr = C1 · (π²·E·Iz/L²) · √(Iw/Iz + L²·G·It/(π²·E·Iz))
        pi2_EIz_L2 = math.pi**2 * E * I_z / L**2
        M_cr_Nmm = C1 * pi2_EIz_L2 * math.sqrt(I_w / I_z + L**2 * G * I_T / (math.pi**2 * E * I_z))
    else:
        raise ValueError("Provide M_cr_kNm directly, or L_cr_m + I_z_cm4 + I_T_cm4 (+I_w_cm6) for calculation.")

    M_cr_kNm = M_cr_Nmm / 1e6

    # §6.3.2.2 – Relative slenderness
    lambda_LT = math.sqrt(W_y_mm3 * fy / M_cr_Nmm)

    notes: list[str] = []
    alpha_LT = LTB_IMPERFECTION[inp.ltb_curve]

    if inp.method == "general":
        # §6.3.2.2 – General case
        Phi_LT = 0.5 * (1.0 + alpha_LT * (lambda_LT - 0.2) + lambda_LT**2)
        disc = Phi_LT**2 - lambda_LT**2
        chi_LT = 1.0 / (Phi_LT + math.sqrt(max(disc, 0.0)))
        chi_LT = min(chi_LT, 1.0)
        notes.append(f"General case (§6.3.2.2): Φ_LT = 0.5·[1 + α(λ̄_LT − 0.2) + λ̄_LT²]")

    else:
        # §6.3.2.3 – Rolled or equivalent welded sections
        lambda_LT_0 = float(inp.lambda_LT_0)
        beta_LT = float(inp.beta_LT)

        Phi_LT = 0.5 * (1.0 + alpha_LT * (lambda_LT - lambda_LT_0) + beta_LT * lambda_LT**2)
        disc = Phi_LT**2 - beta_LT * lambda_LT**2
        chi_LT = 1.0 / (Phi_LT + math.sqrt(max(disc, 0.0)))
        chi_LT = min(chi_LT, 1.0)
        chi_LT = min(chi_LT, 1.0 / lambda_LT**2) if lambda_LT > 0 else chi_LT
        notes.append(
            f"Rolled/welded (§6.3.2.3): λ̄_LT,0 = {lambda_LT_0}, β = {beta_LT}"
        )

    # §6.3.2.1 – Buckling resistance moment
    M_b_Rd_Nmm = chi_LT * W_y_mm3 * fy / gamma_M1
    M_b_Rd_kNm = M_b_Rd_Nmm / 1e6

    notes.insert(0, f"λ̄_LT = √({w_label}·fy/M_cr) = {lambda_LT:.4f}")
    notes.append(f"χ_LT = {chi_LT:.4f} (curve '{inp.ltb_curve}', α_LT = {alpha_LT})")
    notes.append(f"M_b,Rd = χ_LT·{w_label}·fy/γM1 = {M_b_Rd_kNm:.2f} kNm")

    return {
        "inputs_used": {
            "section_name": inp.section_name,
            "steel_grade": inp.steel_grade,
            "fy_mpa": fy,
            "gamma_M1": gamma_M1,
            "section_class": inp.section_class,
            "ltb_curve": inp.ltb_curve,
            "method": inp.method,
            "C1": float(inp.C1),
        },
        "intermediate": {
            "W_y_cm3": round(W_y_mm3 / 1000.0, 2),
            "M_cr_kNm": round(M_cr_kNm, 2),
            "lambda_LT": round(lambda_LT, 4),
            "alpha_LT": alpha_LT,
            "Phi_LT": round(Phi_LT, 4),
            "chi_LT": round(chi_LT, 4),
        },
        "outputs": {
            "M_b_Rd_kNm": round(M_b_Rd_kNm, 2),
            "chi_LT": round(chi_LT, 4),
            "lambda_LT": round(lambda_LT, 4),
            "M_cr_kNm": round(M_cr_kNm, 2),
        },
        "clause_references": [
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "6.3.2.1",
                "title": "Lateral torsional buckling resistance",
                "pointer": "en_1993_1_1_2005_structured.json#6.3.2.1",
            },
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "6.3.2.2" if inp.method == "general" else "6.3.2.3",
                "title": "LTB curves" if inp.method == "general" else "LTB for rolled/welded sections",
                "pointer": "en_1993_1_1_2005_structured.json#6.3.2.2",
            },
        ],
        "notes": notes,
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=LTBResistanceInput, handler=calculate)
