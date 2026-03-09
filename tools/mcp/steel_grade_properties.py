from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli

TOOL_NAME = "steel_grade_properties"

EC3_TABLE_3_1 = {
    "S235": {"fy_t_le_40": 235, "fy_40_lt_t_le_80": 215, "fu": 360},
    "S275": {"fy_t_le_40": 275, "fy_40_lt_t_le_80": 255, "fu": 430},
    "S355": {"fy_t_le_40": 355, "fy_40_lt_t_le_80": 335, "fu": 510},
    "S420": {"fy_t_le_40": 420, "fy_40_lt_t_le_80": 390, "fu": 520},
    "S460": {"fy_t_le_40": 460, "fy_40_lt_t_le_80": 430, "fu": 540},
}


class SteelGradeInput(BaseModel):
    steel_grade: str = Field(description="Steel grade designation, e.g. S355")
    thickness_mm: Optional[PositiveFloat] = Field(
        default=None,
        description="Nominal element thickness in mm. If >40mm, reduced fy applies per EC3 Table 3.1.",
    )


def lookup(inp: SteelGradeInput) -> dict:
    grade = inp.steel_grade.strip().upper()
    if grade not in EC3_TABLE_3_1:
        raise ValueError(
            f"Unsupported steel grade '{inp.steel_grade}'. "
            f"Available: {', '.join(sorted(EC3_TABLE_3_1.keys()))}"
        )

    row = EC3_TABLE_3_1[grade]
    t = inp.thickness_mm or 16.0
    thickness_note = "user-provided" if inp.thickness_mm else "assumed 16 mm (t ≤ 40 mm range)"

    if t <= 40.0:
        fy = row["fy_t_le_40"]
        thickness_range = "t ≤ 40 mm"
    elif t <= 80.0:
        fy = row["fy_40_lt_t_le_80"]
        thickness_range = "40 mm < t ≤ 80 mm"
    else:
        fy = row["fy_40_lt_t_le_80"]
        thickness_range = f"t = {t} mm (> 80 mm — using 40–80 mm value; verify with product standard)"

    import math
    epsilon = math.sqrt(235.0 / fy)

    return {
        "inputs_used": {
            "steel_grade": grade,
            "thickness_mm": t,
            "thickness_note": thickness_note,
        },
        "outputs": {
            "fy_mpa": fy,
            "fu_mpa": row["fu"],
            "epsilon": round(epsilon, 4),
            "thickness_range": thickness_range,
            "E_gpa": 210,
            "G_gpa": 81,
            "poisson_ratio": 0.3,
        },
        "clause_references": [
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "Table 3.1",
                "title": "Nominal values of yield strength fy and ultimate tensile strength fu",
                "pointer": "en_1993_1_1_2005_ocr.json#table_3.1",
            },
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "3.2.6",
                "title": "Design values of material coefficients",
                "pointer": "en_1993_1_1_2005_ocr.json#3.2.6",
            },
        ],
        "notes": [
            f"Yield strength fy = {fy} MPa for {grade}, {thickness_range}.",
            f"ε = √(235/fy) = {epsilon:.4f}",
        ],
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=SteelGradeInput, handler=lookup)
