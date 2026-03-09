from __future__ import annotations

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli
from tools.mcp.clause_ref import clause_ref

TOOL_NAME = "interaction_check_ec3"


class InteractionInput(BaseModel):
    MEd_kNm: PositiveFloat
    NEd_kN: PositiveFloat
    M_Rd_kNm: PositiveFloat
    N_Rd_kN: PositiveFloat
    alpha_m: PositiveFloat = Field(default=1.0)
    alpha_n: PositiveFloat = Field(default=1.0)


def check_interaction(input_data: InteractionInput) -> dict:
    moment_term = float(input_data.alpha_m) * float(input_data.MEd_kNm) / float(input_data.M_Rd_kNm)
    axial_term = float(input_data.alpha_n) * float(input_data.NEd_kN) / float(input_data.N_Rd_kN)
    utilization = axial_term + moment_term

    return {
        "inputs_used": input_data.model_dump(),
        "intermediate": {
            "axial_ratio": round(axial_term, 4),
            "moment_ratio": round(moment_term, 4),
        },
        "outputs": {
            "utilization": round(utilization, 4),
            "pass": utilization <= 1.0,
            "criterion": "alpha_n * NEd/NRd + alpha_m * MEd/MRd <= 1.0",
        },
        "clause_references": [
            clause_ref("ec3.en1993-1-1.2005", "6.2.9(1)", "Interaction formulae")
        ],
        "notes": [
            "This is a simplified linear interaction placeholder aligned to EC3 clause structure.",
        ],
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=InteractionInput, handler=check_interaction)
