from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli

TOOL_NAME = "critical_load_factor_ec3"


class CriticalLoadFactorInput(BaseModel):
    """Input for §5.2.1 – Check if second-order effects need to be considered."""

    method: Literal["direct", "portal_frame"] = Field(
        default="portal_frame",
        description="'direct' if αcr known, 'portal_frame' for approximate column method",
    )
    analysis_type: Literal["elastic", "plastic"] = Field(
        default="elastic", description="Type of global analysis"
    )

    # Direct input
    alpha_cr: Optional[float] = Field(
        default=None, description="Critical load factor α_cr = F_cr/F_Ed (if known)"
    )

    # Portal frame method – §5.2.1(4)B
    H_Ed_kN: Optional[float] = Field(
        default=None,
        description="Total horizontal reaction at bottom of storey in kN (from horizontal loads + imperfections)",
    )
    V_Ed_kN: Optional[PositiveFloat] = Field(
        default=None, description="Total vertical load at bottom of storey in kN"
    )
    h_m: Optional[PositiveFloat] = Field(
        default=None, description="Storey height in m"
    )
    delta_H_Ed_mm: Optional[PositiveFloat] = Field(
        default=None,
        description="Horizontal displacement at top of storey relative to bottom (first-order) in mm",
    )

    # Amplification (§5.2.2)
    compute_amplification: bool = Field(
        default=True, description="Also compute amplification factor 1/(1 − 1/αcr)"
    )


def calculate(inp: CriticalLoadFactorInput) -> dict:
    notes: list[str] = []

    if inp.method == "direct":
        if inp.alpha_cr is None:
            raise ValueError("Direct method requires alpha_cr.")
        alpha_cr = float(inp.alpha_cr)
    else:
        # §5.2.1(4)B – Portal frame approximation
        if inp.H_Ed_kN is None or inp.V_Ed_kN is None or inp.h_m is None or inp.delta_H_Ed_mm is None:
            raise ValueError("Portal frame method requires H_Ed_kN, V_Ed_kN, h_m, delta_H_Ed_mm.")

        H_Ed = abs(float(inp.H_Ed_kN))
        V_Ed = float(inp.V_Ed_kN)
        h = float(inp.h_m)
        delta_H = float(inp.delta_H_Ed_mm)

        if delta_H <= 0 or H_Ed <= 0:
            raise ValueError("H_Ed and delta_H_Ed must be positive.")

        # αcr = (H_Ed / V_Ed) · (h / δ_H,Ed)
        h_mm = h * 1000.0
        alpha_cr = (H_Ed / V_Ed) * (h_mm / delta_H)
        notes.append(
            f"α_cr = (H_Ed/V_Ed)·(h/δ_H,Ed) = ({H_Ed:.1f}/{V_Ed:.1f})·({h_mm:.0f}/{delta_H:.2f}) = {alpha_cr:.2f}"
        )

    # §5.2.1(3) – Threshold checks
    if inp.analysis_type == "elastic":
        threshold = 10.0
        second_order_needed = alpha_cr < threshold
        notes.append(f"Elastic analysis: α_cr ≥ {threshold} required → {'OK' if not second_order_needed else 'second-order analysis needed'}")
    else:
        threshold = 15.0
        second_order_needed = alpha_cr < threshold
        notes.append(f"Plastic analysis: α_cr ≥ {threshold} required → {'OK' if not second_order_needed else 'second-order analysis needed'}")

    results: dict = {
        "alpha_cr": round(alpha_cr, 2),
        "threshold": threshold,
        "second_order_required": second_order_needed,
        "first_order_sufficient": not second_order_needed,
    }

    # §5.2.2(6)B – Amplification factor
    if inp.compute_amplification:
        if alpha_cr <= 1.0:
            notes.append("α_cr ≤ 1.0 → structure is unstable under design loads!")
            results["amplification_factor"] = float("inf")
            results["amplification_valid"] = False
        elif alpha_cr < 3.0:
            amplification = 1.0 / (1.0 - 1.0 / alpha_cr)
            results["amplification_factor"] = round(amplification, 4)
            results["amplification_valid"] = False
            notes.append(f"α_cr < 3.0 → amplification 1/(1−1/α_cr) = {amplification:.4f} but method not valid (α_cr must ≥ 3.0)")
        else:
            amplification = 1.0 / (1.0 - 1.0 / alpha_cr)
            results["amplification_factor"] = round(amplification, 4)
            results["amplification_valid"] = True
            notes.append(f"Amplification factor = 1/(1−1/α_cr) = 1/(1−1/{alpha_cr:.2f}) = {amplification:.4f}")

    return {
        "inputs_used": {
            "method": inp.method,
            "analysis_type": inp.analysis_type,
            "H_Ed_kN": float(inp.H_Ed_kN) if inp.H_Ed_kN else None,
            "V_Ed_kN": float(inp.V_Ed_kN) if inp.V_Ed_kN else None,
            "h_m": float(inp.h_m) if inp.h_m else None,
            "delta_H_Ed_mm": float(inp.delta_H_Ed_mm) if inp.delta_H_Ed_mm else None,
        },
        "outputs": results,
        "clause_references": [
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "5.2.1",
                "title": "Effects of deformed geometry of the structure",
                "pointer": "en_1993_1_1_2005_structured.json#5.2.1",
            },
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "5.2.2(6)B",
                "title": "Sway amplification method",
                "pointer": "en_1993_1_1_2005_structured.json#5.2.2",
            },
        ],
        "notes": notes,
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=CriticalLoadFactorInput, handler=calculate)
