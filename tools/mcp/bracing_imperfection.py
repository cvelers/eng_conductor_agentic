from __future__ import annotations

import math

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt

from tools.mcp.cli import run_cli
from tools.mcp.clause_ref import clause_ref

TOOL_NAME = "bracing_imperfection_ec3"


class BracingImperfectionInput(BaseModel):
    """Input for §5.3.3 – Imperfection for analysis of bracing systems."""

    L_m: PositiveFloat = Field(description="Span of bracing system in m")
    m: PositiveInt = Field(description="Number of members to be restrained")
    N_Ed_total_kN: PositiveFloat = Field(
        description="Sum of axial forces in restrained members ΣN_Ed in kN"
    )
    delta_q_mm: float = Field(
        default=0.0,
        description="In-plane deflection of bracing system due to q_d plus external loads in mm (for iterative check)",
    )


def calculate(inp: BracingImperfectionInput) -> dict:
    L = float(inp.L_m)
    m = int(inp.m)
    N_Ed_sum = float(inp.N_Ed_total_kN)

    # §5.3.3(1) – α_m = √(0.5·(1 + 1/m))
    alpha_m = math.sqrt(0.5 * (1.0 + 1.0 / m))

    # §5.3.3(1) – e_0 = α_m · L / 500
    L_mm = L * 1000.0
    e_0_mm = alpha_m * L_mm / 500.0
    e_0_m = e_0_mm / 1000.0

    # §5.3.3(2) – Equivalent stabilising force per unit length
    # q_d = ΣN_Ed · 8 · (e_0 + δ_q) / L²
    delta_q = float(inp.delta_q_mm)
    q_d_kN_per_m = N_Ed_sum * 8.0 * (e_0_mm + delta_q) / (L_mm**2) * 1000.0  # kN/m

    # Total stabilising force Q = q_d · L
    Q_total_kN = q_d_kN_per_m * L

    notes: list[str] = [
        f"α_m = √(0.5·(1+1/{m})) = {alpha_m:.4f}",
        f"e_0 = α_m·L/500 = {alpha_m:.4f}×{L_mm:.0f}/500 = {e_0_mm:.2f} mm",
        f"q_d = ΣN_Ed·8·(e_0+δ_q)/L² = {N_Ed_sum:.1f}×8×({e_0_mm:.2f}+{delta_q:.2f})/{L_mm:.0f}² = {q_d_kN_per_m:.4f} kN/m",
        f"Total stabilising force Q = q_d·L = {Q_total_kN:.2f} kN",
    ]

    return {
        "inputs_used": {
            "L_m": L,
            "m": m,
            "N_Ed_total_kN": N_Ed_sum,
            "delta_q_mm": delta_q,
        },
        "intermediate": {
            "alpha_m": round(alpha_m, 4),
            "e_0_mm": round(e_0_mm, 2),
        },
        "outputs": {
            "e_0_mm": round(e_0_mm, 2),
            "q_d_kN_per_m": round(q_d_kN_per_m, 4),
            "Q_total_kN": round(Q_total_kN, 2),
            "alpha_m": round(alpha_m, 4),
        },
        "clause_references": [
            clause_ref("ec3.en1993-1-1.2005", "5.3.3", "Imperfection for analysis of bracing systems"),
        ],
        "notes": notes,
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=BracingImperfectionInput, handler=calculate)
