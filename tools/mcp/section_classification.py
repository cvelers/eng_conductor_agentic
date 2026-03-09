from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveFloat, model_validator

from tools.mcp.cli import run_cli
from tools.mcp.section_library import SECTION_LIBRARY, steel_grade_to_fy
from tools.mcp.clause_ref import clause_ref

TOOL_NAME = "section_classification_ec3"


class SectionClassificationInput(BaseModel):
    section_name: str | None = None
    section_type: Literal["I", "H"] = "I"
    stress_type: Literal["bending", "compression"] = Field(
        default="bending",
        description="Dominant stress condition: 'bending' for beams, 'compression' for columns.",
    )
    h_mm: PositiveFloat | None = Field(default=None, description="Overall depth")
    b_mm: PositiveFloat | None = Field(default=None, description="Flange width")
    tw_mm: PositiveFloat | None = Field(default=None, description="Web thickness")
    tf_mm: PositiveFloat | None = Field(default=None, description="Flange thickness")
    r_mm: Optional[PositiveFloat] = Field(default=None, description="Root fillet radius (for rolled sections)")
    steel_grade: str = "S355"
    fy_mpa: PositiveFloat | None = None
    thickness_mm: Optional[PositiveFloat] = Field(
        default=None,
        description="Governing element thickness for fy lookup (default: max of tf, tw).",
    )

    @model_validator(mode="after")
    def fill_section_dimensions(self) -> "SectionClassificationInput":
        if self.section_name:
            key = self.section_name.upper().replace(" ", "")
            if key in SECTION_LIBRARY:
                row = SECTION_LIBRARY[key]
                if self.h_mm is None:
                    self.h_mm = float(row["h_mm"])
                if self.b_mm is None:
                    self.b_mm = float(row["b_mm"])
                if self.tw_mm is None:
                    self.tw_mm = float(row["tw_mm"])
                if self.tf_mm is None:
                    self.tf_mm = float(row["tf_mm"])
                if self.r_mm is None and "r_mm" in row:
                    self.r_mm = float(row["r_mm"])

        missing = [
            name
            for name in ["h_mm", "b_mm", "tw_mm", "tf_mm"]
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                "Missing section geometry fields: "
                + ", ".join(missing)
                + ". Provide section_name from built-in library or explicit dimensions in mm."
            )

        if self.fy_mpa is None:
            t = self.thickness_mm or max(float(self.tf_mm), float(self.tw_mm))
            self.fy_mpa = steel_grade_to_fy(self.steel_grade, thickness_mm=t)
        return self


def _class_from_limit(value: float, limits: tuple[float, float, float]) -> int:
    if value <= limits[0]:
        return 1
    if value <= limits[1]:
        return 2
    if value <= limits[2]:
        return 3
    return 4


def classify(input_data: SectionClassificationInput) -> dict:
    fy = float(input_data.fy_mpa)
    epsilon = math.sqrt(235.0 / fy)

    r = float(input_data.r_mm) if input_data.r_mm is not None else 0.0

    # EC3 Table 5.2: web clear depth c = h - 2*tf - 2*r
    web_c = float(input_data.h_mm) - 2.0 * float(input_data.tf_mm) - 2.0 * r
    web_ratio = web_c / float(input_data.tw_mm)

    # EC3 Table 5.2: flange outstand c = (b - tw - 2*r) / 2
    flange_c = (float(input_data.b_mm) - float(input_data.tw_mm) - 2.0 * r) / 2.0
    flange_ratio = flange_c / float(input_data.tf_mm)

    # EC3 Table 5.2 limits — stress-type dependent for web (internal part)
    if input_data.stress_type == "bending":
        web_limits = (72.0 * epsilon, 83.0 * epsilon, 124.0 * epsilon)
    else:  # compression
        web_limits = (33.0 * epsilon, 38.0 * epsilon, 42.0 * epsilon)

    # Flanges (outstand parts) — limits are the same for bending and compression
    flange_limits = (9.0 * epsilon, 10.0 * epsilon, 14.0 * epsilon)

    web_class = _class_from_limit(web_ratio, web_limits)
    flange_class = _class_from_limit(flange_ratio, flange_limits)
    governing_class = max(web_class, flange_class)

    return {
        "inputs_used": {
            "section_name": input_data.section_name,
            "stress_type": input_data.stress_type,
            "h_mm": input_data.h_mm,
            "b_mm": input_data.b_mm,
            "tw_mm": input_data.tw_mm,
            "tf_mm": input_data.tf_mm,
            "r_mm": r,
            "fy_mpa": round(fy, 3),
        },
        "intermediate": {
            "epsilon": round(epsilon, 4),
            "web_c_mm": round(web_c, 2),
            "web_slenderness_c_over_t": round(web_ratio, 3),
            "web_class_1_limit": round(web_limits[0], 2),
            "web_class_2_limit": round(web_limits[1], 2),
            "web_class_3_limit": round(web_limits[2], 2),
            "flange_c_mm": round(flange_c, 2),
            "flange_slenderness_c_over_t": round(flange_ratio, 3),
        },
        "outputs": {
            "web_class": web_class,
            "flange_class": flange_class,
            "governing_class": governing_class,
        },
        "clause_references": [
            clause_ref("ec3.en1993-1-1.2005", "5.5.2(1)", "Classification of cross-sections"),
            clause_ref("ec3.en1993-1-1.2005", "Table 5.2", "Width-to-thickness limits"),
        ],
        "notes": [
            f"Stress type: {input_data.stress_type}",
            f"Web c/t = {web_ratio:.2f}, Class {web_class} (limit {web_limits[0]:.2f}/{web_limits[1]:.2f}/{web_limits[2]:.2f})",
            f"Flange c/t = {flange_ratio:.2f}, Class {flange_class}",
        ],
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=SectionClassificationInput, handler=classify)
