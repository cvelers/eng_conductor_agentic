from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli

TOOL_NAME = "class4_effective_ec3"


class Class4EffectiveInput(BaseModel):
    """Input for §6.2.2.5 – Effective Class 4 cross-section properties (additional moment)."""

    N_Ed_kN: PositiveFloat = Field(description="Design axial compression force N_Ed in kN")
    e_N_mm: float = Field(
        description="Shift of centroid of effective area relative to gross cross-section e_N in mm"
    )

    # Optional effective properties
    A_eff_cm2: Optional[PositiveFloat] = Field(
        default=None, description="Effective area A_eff in cm² (from EN 1993-1-5)"
    )
    A_gross_cm2: Optional[PositiveFloat] = Field(
        default=None, description="Gross area A_gross in cm² for comparison"
    )


def calculate(inp: Class4EffectiveInput) -> dict:
    N_Ed = float(inp.N_Ed_kN)
    e_N = float(inp.e_N_mm)

    notes: list[str] = []

    # §6.2.2.5(4) – ΔM_Ed = N_Ed · e_N
    delta_M = N_Ed * e_N / 1000.0  # kNm (N_Ed in kN, e_N in mm → kN·mm → /1000 = kNm)

    notes.append(f"ΔM_Ed = N_Ed · e_N = {N_Ed:.2f} × {e_N:.2f} / 1000 = {delta_M:.4f} kNm")
    notes.append(
        "Note: sign of ΔM depends on the effect in the combination of internal forces (§6.2.9.3(2))"
    )

    outputs: dict = {
        "delta_M_Ed_kNm": round(delta_M, 4),
        "delta_M_Ed_kNmm": round(N_Ed * e_N, 2),
    }

    if inp.A_eff_cm2 and inp.A_gross_cm2:
        ratio = float(inp.A_eff_cm2) / float(inp.A_gross_cm2)
        outputs["A_eff_over_A_gross"] = round(ratio, 4)
        notes.append(f"A_eff/A_gross = {ratio:.4f}")

    return {
        "inputs_used": {
            "N_Ed_kN": N_Ed,
            "e_N_mm": e_N,
            "A_eff_cm2": float(inp.A_eff_cm2) if inp.A_eff_cm2 else None,
            "A_gross_cm2": float(inp.A_gross_cm2) if inp.A_gross_cm2 else None,
        },
        "outputs": outputs,
        "clause_references": [
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "6.2.2.5",
                "title": "Effective Class 4 cross-section properties",
                "pointer": "en_1993_1_1_2005_structured.json#6.2.2.5",
            },
        ],
        "notes": notes,
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=Class4EffectiveInput, handler=calculate)
