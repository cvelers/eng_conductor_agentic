from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, PositiveFloat, model_validator

from tools.mcp.cli import run_cli
from tools.mcp.section_library import SECTION_LIBRARY, steel_grade_to_fy
from tools.mcp.clause_ref import clause_ref

TOOL_NAME = "tension_resistance_ec3"

def _get_fu(grade: str) -> float:
    """Get fu from the authoritative EC3 Table 3.1 data."""
    from tools.mcp.steel_grade_properties import EC3_TABLE_3_1

    key = grade.strip().upper()
    if key in EC3_TABLE_3_1:
        return float(EC3_TABLE_3_1[key]["fu"])
    raise ValueError(f"No fu value for grade '{grade}'. Available: {sorted(EC3_TABLE_3_1.keys())}")


class TensionResistanceInput(BaseModel):
    section_name: Optional[str] = Field(default=None, description="Section name, e.g. IPE300")
    steel_grade: str = Field(default="S355", description="Steel grade, e.g. S355")
    area_cm2: Optional[PositiveFloat] = Field(default=None, description="Gross cross-section area in cm²")
    A_net_cm2: Optional[PositiveFloat] = Field(default=None, description="Net cross-section area in cm² (after bolt holes)")
    fy_mpa: Optional[PositiveFloat] = Field(default=None, description="Yield strength in MPa")
    fu_mpa: Optional[PositiveFloat] = Field(default=None, description="Ultimate tensile strength in MPa")
    gamma_M0: PositiveFloat = Field(default=1.0, description="Partial factor γ_M0")
    gamma_M2: PositiveFloat = Field(default=1.25, description="Partial factor γ_M2")
    connection_category: Optional[str] = Field(
        default=None,
        description="Connection category: 'A', 'B', or 'C' per EN 1993-1-8 3.4.1",
    )

    @model_validator(mode="after")
    def fill_from_library(self) -> "TensionResistanceInput":
        if self.section_name:
            key = self.section_name.upper().replace(" ", "")
            if key in SECTION_LIBRARY:
                row = SECTION_LIBRARY[key]
                if self.area_cm2 is None:
                    self.area_cm2 = float(row["area_cm2"])
        if self.fy_mpa is None:
            self.fy_mpa = steel_grade_to_fy(self.steel_grade)
        if self.fu_mpa is None:
            try:
                self.fu_mpa = _get_fu(self.steel_grade)
            except ValueError:
                self.fu_mpa = self.fy_mpa * 1.35
        if self.area_cm2 is None:
            raise ValueError("Provide section_name or area_cm2.")
        return self


def calculate(inp: TensionResistanceInput) -> dict:
    fy = float(inp.fy_mpa)
    fu = float(inp.fu_mpa)
    A_mm2 = float(inp.area_cm2) * 100.0
    gamma_M0 = float(inp.gamma_M0)
    gamma_M2 = float(inp.gamma_M2)

    # §6.2.3(2)a – Plastic resistance of gross section
    N_pl_Rd = A_mm2 * fy / gamma_M0 / 1000.0  # kN

    results: dict = {}
    governing = "N_pl_Rd"
    N_t_Rd = N_pl_Rd

    if inp.A_net_cm2 is not None:
        A_net_mm2 = float(inp.A_net_cm2) * 100.0

        # §6.2.3(2)b – Ultimate resistance of net section
        N_u_Rd = 0.9 * A_net_mm2 * fu / gamma_M2 / 1000.0  # kN
        results["N_u_Rd_kN"] = round(N_u_Rd, 2)

        if inp.connection_category and inp.connection_category.upper() == "C":
            # §6.2.3(4) – Category C: net section yielding
            N_net_Rd = A_net_mm2 * fy / gamma_M0 / 1000.0
            results["N_net_Rd_kN"] = round(N_net_Rd, 2)
            N_t_Rd = min(N_pl_Rd, N_u_Rd, N_net_Rd)
            governing = min(
                [("N_pl_Rd", N_pl_Rd), ("N_u_Rd", N_u_Rd), ("N_net_Rd", N_net_Rd)],
                key=lambda x: x[1],
            )[0]
        else:
            N_t_Rd = min(N_pl_Rd, N_u_Rd)
            governing = "N_pl_Rd" if N_pl_Rd <= N_u_Rd else "N_u_Rd"

    # N_pl_Rd_kN reports the governing design tension resistance N_t,Rd
    # (= min(N_pl,Rd, N_u,Rd) per EC3 6.2.3). When no holes are present it
    # equals the gross-section plastic resistance; otherwise it may be lower.
    results["N_pl_Rd_kN"] = round(N_t_Rd, 2)
    results["N_t_Rd_kN"] = round(N_t_Rd, 2)
    results["governing"] = governing
    governing_latex = {
        "N_pl_Rd": r"N_{pl,Rd}",
        "N_u_Rd": r"N_{u,Rd}",
        "N_net_Rd": r"N_{net,Rd}",
    }.get(governing, rf"\mathrm{{{governing}}}")

    return {
        "inputs_used": {
            "section_name": inp.section_name,
            "steel_grade": inp.steel_grade,
            "fy_mpa": fy,
            "fu_mpa": fu,
            "area_cm2": float(inp.area_cm2),
            "A_net_cm2": float(inp.A_net_cm2) if inp.A_net_cm2 else None,
            "gamma_M0": gamma_M0,
            "gamma_M2": gamma_M2,
            "connection_category": inp.connection_category,
        },
        "outputs": results,
        "clause_references": [
            clause_ref("ec3.en1993-1-1.2005", "6.2.3(2)", "Tension resistance"),
        ],
        "notes": [
            {
                "latex": (
                    rf"N_{{pl,Rd}} = A \cdot f_y / \gamma_{{M0}}"
                    rf" = {A_mm2:.0f} \cdot {fy:.0f} / {gamma_M0} = {N_pl_Rd:.2f}\,\mathrm{{kN}}"
                ),
            },
            {
                "latex": rf"\text{{Governing resistance: }} {governing_latex} = {N_t_Rd:.2f}\,\mathrm{{kN}}",
            },
        ],
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=TensionResistanceInput, handler=calculate)
