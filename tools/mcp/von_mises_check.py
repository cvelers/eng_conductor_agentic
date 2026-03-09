from __future__ import annotations

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli
from tools.mcp.section_library import steel_grade_to_fy
from tools.mcp.clause_ref import clause_ref

TOOL_NAME = "von_mises_check_ec3"


class VonMisesInput(BaseModel):
    sigma_x_Ed_mpa: float = Field(default=0.0, description="Longitudinal stress σ_x,Ed in MPa (positive = tension)")
    sigma_z_Ed_mpa: float = Field(default=0.0, description="Transverse stress σ_z,Ed in MPa (positive = tension)")
    tau_Ed_mpa: float = Field(default=0.0, description="Shear stress τ_Ed in MPa")
    steel_grade: str = Field(default="S355", description="Steel grade, e.g. S355")
    fy_mpa: PositiveFloat | None = Field(default=None, description="Yield strength in MPa (overrides steel_grade)")
    gamma_M0: PositiveFloat = Field(default=1.0, description="Partial factor γ_M0")


def calculate(inp: VonMisesInput) -> dict:
    fy = float(inp.fy_mpa) if inp.fy_mpa else steel_grade_to_fy(inp.steel_grade)
    gamma_M0 = float(inp.gamma_M0)
    f_yd = fy / gamma_M0

    sx = float(inp.sigma_x_Ed_mpa)
    sz = float(inp.sigma_z_Ed_mpa)
    tau = float(inp.tau_Ed_mpa)

    # EC3 §6.2.1(5) – Von Mises yield criterion
    # (σx/fyd)² + (σz/fyd)² - (σx/fyd)·(σz/fyd) + 3·(τ/fyd)² ≤ 1.0
    ratio_x = sx / f_yd if f_yd > 0 else 0
    ratio_z = sz / f_yd if f_yd > 0 else 0
    ratio_tau = tau / f_yd if f_yd > 0 else 0

    utilization = ratio_x**2 + ratio_z**2 - ratio_x * ratio_z + 3 * ratio_tau**2

    return {
        "inputs_used": {
            "sigma_x_Ed_mpa": sx,
            "sigma_z_Ed_mpa": sz,
            "tau_Ed_mpa": tau,
            "steel_grade": inp.steel_grade,
            "fy_mpa": fy,
            "gamma_M0": gamma_M0,
            "f_yd_mpa": round(f_yd, 2),
        },
        "intermediate": {
            "sigma_x_ratio": round(ratio_x, 4),
            "sigma_z_ratio": round(ratio_z, 4),
            "tau_ratio": round(ratio_tau, 4),
        },
        "outputs": {
            "utilization": round(utilization, 4),
            "pass": utilization <= 1.0,
            "criterion": "(σx/fyd)² + (σz/fyd)² − (σx·σz)/fyd² + 3·(τ/fyd)² ≤ 1.0",
        },
        "clause_references": [
            clause_ref("ec3.en1993-1-1.2005", "6.2.1(5)", "General yield criterion – Von Mises"),
        ],
        "notes": [
            f"Von Mises utilization = {utilization:.4f} ({'OK' if utilization <= 1.0 else 'FAIL'})",
        ],
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=VonMisesInput, handler=calculate)
