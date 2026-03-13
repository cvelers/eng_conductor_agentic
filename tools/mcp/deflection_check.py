from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli
from tools.mcp.clause_ref import clause_ref

TOOL_NAME = "deflection_check"

DEFLECTION_LIMITS = {
    "L/250": 250,
    "L/300": 300,
    "L/350": 350,
    "L/400": 400,
    "L/500": 500,
    "L/200": 200,
}


class DeflectionCheckInput(BaseModel):
    span_m: PositiveFloat = Field(description="Beam span in metres")
    actual_deflection_mm: PositiveFloat = Field(description="Calculated or measured deflection in mm")
    limit_ratio: Literal["L/200", "L/250", "L/300", "L/350", "L/400", "L/500"] = Field(
        default="L/250",
        description="Deflection limit ratio. L/250 typical for total (EC0 A1.4.3), L/300 for variable actions.",
    )


def check(inp: DeflectionCheckInput) -> dict:
    denominator = DEFLECTION_LIMITS[inp.limit_ratio]
    allowable_mm = (inp.span_m * 1000) / denominator
    utilization = inp.actual_deflection_mm / allowable_mm
    passes = utilization <= 1.0

    return {
        "inputs_used": {
            "span_m": inp.span_m,
            "actual_deflection_mm": inp.actual_deflection_mm,
            "limit_ratio": inp.limit_ratio,
        },
        "outputs": {
            "allowable_deflection_mm": round(allowable_mm, 2),
            "utilization": round(utilization, 4),
            "pass": passes,
            "status": "OK" if passes else "FAIL",
        },
        "clause_references": [
            clause_ref("ec0.en1990.2002", "A1.4.3", "Vertical deflections (Annex A1, Table A1.4)", pointer="en_1990#annex_a1_table_a1.4"),
        ],
        "notes": [
            {
                "latex": (
                    rf"\delta_{{allow}} = L/{denominator} = {inp.span_m * 1000:.0f}/{denominator}"
                    rf" = {allowable_mm:.2f}\,\mathrm{{mm}}"
                ),
            },
            {
                "latex": (
                    rf"\delta_{{actual}} = {inp.actual_deflection_mm}\,\mathrm{{mm}}"
                    rf"\;\rightarrow\; \text{{utilization}} = {utilization * 100:.2f}\%"
                ),
            },
            f"Result: {'PASS ✓' if passes else 'FAIL ✗'}",
        ],
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=DeflectionCheckInput, handler=check)
