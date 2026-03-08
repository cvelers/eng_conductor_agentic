from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt

from tools.mcp.cli import run_cli

TOOL_NAME = "plastic_hinge_restraint_ec3"


class PlasticHingeRestraintInput(BaseModel):
    """Input for §6.3.5.2 – Restraint forces at rotated plastic hinge locations."""

    N_f_Ed_kN: PositiveFloat = Field(
        description="Axial force in the compressed flange at the plastic hinge location N_f,Ed in kN"
    )

    # For local restraint force (§6.3.5.2(3)B)
    check_local: bool = Field(
        default=True,
        description="Check the 2.5% local force requirement",
    )

    # For bracing system forces (§6.3.5.2(5)B)
    check_bracing: bool = Field(
        default=False,
        description="Check the bracing system force Q_m",
    )
    m_members: Optional[PositiveInt] = Field(
        default=None,
        description="Number of members to be restrained (for α_m calculation)",
    )


def calculate(inp: PlasticHingeRestraintInput) -> dict:
    N_f = float(inp.N_f_Ed_kN)
    notes: list[str] = []
    outputs: dict = {}

    if inp.check_local:
        # §6.3.5.2(3)B – Local restraint force = 2.5% of N_f,Ed
        F_local = 0.025 * N_f
        outputs["F_local_kN"] = round(F_local, 2)
        notes.append(f"Local restraint force = 2.5% · N_f,Ed = 0.025 · {N_f:.2f} = {F_local:.2f} kN")

    if inp.check_bracing:
        # §6.3.5.2(5)B – Bracing system force Q_m = 1.5 · α_m · N_f,Ed / 100
        if inp.m_members is None:
            raise ValueError("m_members required for bracing system force check.")

        m = int(inp.m_members)
        alpha_m = math.sqrt(0.5 * (1.0 + 1.0 / m))
        Q_m = 1.5 * alpha_m * N_f / 100.0

        outputs["alpha_m"] = round(alpha_m, 4)
        outputs["Q_m_kN"] = round(Q_m, 2)
        notes.append(f"α_m = √(0.5·(1+1/{m})) = {alpha_m:.4f}")
        notes.append(f"Q_m = 1.5·α_m·N_f,Ed/100 = 1.5·{alpha_m:.4f}·{N_f:.2f}/100 = {Q_m:.2f} kN")

    return {
        "inputs_used": {
            "N_f_Ed_kN": N_f,
            "check_local": inp.check_local,
            "check_bracing": inp.check_bracing,
            "m_members": int(inp.m_members) if inp.m_members else None,
        },
        "outputs": outputs,
        "clause_references": [
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "6.3.5.2",
                "title": "Restraints at rotated plastic hinges",
                "pointer": "en_1993_1_1_2005_structured.json#6.3.5.2",
            },
        ],
        "notes": notes,
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=PlasticHingeRestraintInput, handler=calculate)
