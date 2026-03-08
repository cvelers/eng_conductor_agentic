from __future__ import annotations

import math

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt

from tools.mcp.cli import run_cli

TOOL_NAME = "sway_imperfection_ec3"


class SwayImperfectionInput(BaseModel):
    """Input for §5.3.2 – Global sway imperfection for frames."""

    h_m: PositiveFloat = Field(description="Height of structure in m (for α_h)")
    m: PositiveInt = Field(description="Number of columns in a row contributing to the horizontal force in the plane considered")
    phi_0: float = Field(default=0.005, description="Basic value φ_0 (default 1/200 = 0.005)")

    # Optional: compute equivalent horizontal forces
    compute_equiv_forces: bool = Field(default=False, description="Also compute equivalent horizontal forces")
    N_Ed_total_kN: float = Field(
        default=0.0,
        description="Total vertical load per storey for equivalent horizontal forces in kN",
    )
    num_storeys: PositiveInt = Field(default=1, description="Number of storeys (if different from 1)")


def calculate(inp: SwayImperfectionInput) -> dict:
    h = float(inp.h_m)
    m = int(inp.m)
    phi_0 = float(inp.phi_0)

    # §5.3.2(3) – α_h = 2/√h but 2/3 ≤ α_h ≤ 1.0
    alpha_h = 2.0 / math.sqrt(h)
    alpha_h = max(2.0 / 3.0, min(alpha_h, 1.0))

    # §5.3.2(3) – α_m = √(0.5·(1 + 1/m))
    alpha_m = math.sqrt(0.5 * (1.0 + 1.0 / m))

    # §5.3.2(3) – φ = φ_0 · α_h · α_m
    phi = phi_0 * alpha_h * alpha_m

    # Also express as 1/N
    phi_ratio = round(1.0 / phi) if phi > 0 else float("inf")

    notes: list[str] = [
        f"α_h = 2/√h = 2/√{h} = {2.0 / math.sqrt(h):.4f} → clamped to {alpha_h:.4f}",
        f"α_m = √(0.5·(1+1/{m})) = {alpha_m:.4f}",
        f"φ = φ_0·α_h·α_m = {phi_0}×{alpha_h:.4f}×{alpha_m:.4f} = {phi:.6f} (≈ 1/{phi_ratio})",
    ]

    results: dict = {
        "phi": round(phi, 6),
        "phi_as_ratio": f"1/{phi_ratio}",
        "alpha_h": round(alpha_h, 4),
        "alpha_m": round(alpha_m, 4),
    }

    # Optional: equivalent horizontal forces per storey
    if inp.compute_equiv_forces and inp.N_Ed_total_kN > 0:
        N_Ed = float(inp.N_Ed_total_kN)
        # Equivalent horizontal force per storey = φ · N_Ed
        H_equiv = phi * N_Ed  # kN
        results["H_equiv_per_storey_kN"] = round(H_equiv, 2)
        notes.append(f"Equivalent horizontal force per storey = φ·N_Ed = {phi:.6f}×{N_Ed:.1f} = {H_equiv:.2f} kN")

    return {
        "inputs_used": {
            "h_m": h,
            "m": m,
            "phi_0": phi_0,
        },
        "outputs": results,
        "clause_references": [
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "5.3.2(3)",
                "title": "Imperfections for global analysis of frames",
                "pointer": "en_1993_1_1_2005_structured.json#5.3.2",
            },
        ],
        "notes": notes,
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=SwayImperfectionInput, handler=calculate)
