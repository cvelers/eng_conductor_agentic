from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli
from tools.mcp.section_library import steel_grade_to_fy

TOOL_NAME = "ct_classification_ec3"


class CtClassificationInput(BaseModel):
    """Input for Table 5.2 – Maximum width-to-thickness ratios for compression parts."""

    part_type: Literal["internal", "outstand_rolled", "outstand_welded", "angle"] = Field(
        description="Type of compression part: 'internal', 'outstand_rolled', 'outstand_welded', or 'angle'"
    )
    stress_type: Literal["bending", "compression", "bending_compression"] = Field(
        description="Stress condition: 'bending', 'compression', or 'bending_compression'"
    )

    c_mm: PositiveFloat = Field(description="Width of the compression part c in mm")
    t_mm: PositiveFloat = Field(description="Thickness of the compression part t in mm")

    steel_grade: str = Field(default="S355", description="Steel grade")
    fy_mpa: Optional[PositiveFloat] = Field(default=None, description="Yield strength in MPa")

    # For bending+compression stress distribution
    alpha: Optional[float] = Field(
        default=None,
        description="Stress ratio α for bending+compression (proportion of part in compression)",
    )
    psi: Optional[float] = Field(
        default=None,
        description="Stress ratio ψ = σ₂/σ₁ for Class 3 bending+compression check (-1 ≤ ψ ≤ 1)",
    )

    # For angle sections
    h_mm: Optional[PositiveFloat] = Field(default=None, description="Angle leg height h in mm")
    b_mm: Optional[PositiveFloat] = Field(default=None, description="Angle leg width b in mm")


def calculate(inp: CtClassificationInput) -> dict:
    fy = float(inp.fy_mpa) if inp.fy_mpa else steel_grade_to_fy(inp.steel_grade)
    epsilon = math.sqrt(235.0 / fy)
    ct = float(inp.c_mm) / float(inp.t_mm)

    notes: list[str] = [
        f"ε = √(235/{fy:.0f}) = {epsilon:.4f}",
        f"c/t = {ct:.2f}",
    ]

    limits: dict[int, float] = {}

    if inp.part_type == "internal":
        if inp.stress_type == "bending":
            limits = {1: 72.0 * epsilon, 2: 83.0 * epsilon, 3: 124.0 * epsilon}
        elif inp.stress_type == "compression":
            limits = {1: 33.0 * epsilon, 2: 38.0 * epsilon, 3: 42.0 * epsilon}
        elif inp.stress_type == "bending_compression":
            alpha = float(inp.alpha) if inp.alpha is not None else 0.5
            psi = float(inp.psi) if inp.psi is not None else 1.0

            # Class 1
            if alpha > 0.5:
                lim1 = 396.0 * epsilon / (13.0 * alpha - 1.0)
            else:
                lim1 = 36.0 * epsilon / alpha if alpha > 0 else float("inf")

            # Class 2
            if alpha > 0.5:
                lim2 = 456.0 * epsilon / (13.0 * alpha - 1.0)
            else:
                lim2 = 41.5 * epsilon / alpha if alpha > 0 else float("inf")

            # Class 3
            if psi > -1.0:
                lim3 = 42.0 * epsilon / (0.67 + 0.33 * psi)
            else:
                lim3 = 62.0 * epsilon * (1.0 - psi) * math.sqrt(-psi)

            limits = {1: lim1, 2: lim2, 3: lim3}
            notes.append(f"α = {alpha:.3f}, ψ = {psi:.3f}")

    elif inp.part_type in ("outstand_rolled", "outstand_welded"):
        if inp.stress_type == "compression":
            if inp.part_type == "outstand_rolled":
                limits = {1: 9.0 * epsilon, 2: 10.0 * epsilon, 3: 14.0 * epsilon}
            else:
                limits = {1: 9.0 * epsilon, 2: 10.0 * epsilon, 3: 14.0 * epsilon}
        elif inp.stress_type == "bending_compression":
            alpha = float(inp.alpha) if inp.alpha is not None else 1.0
            alpha = max(alpha, 0.01)
            if inp.part_type == "outstand_rolled":
                limits = {
                    1: 9.0 * epsilon / alpha,
                    2: 10.0 * epsilon / alpha,
                    3: 21.0 * epsilon * math.sqrt(1.0 / alpha) if alpha <= 1 else 14.0 * epsilon,
                }
            else:
                limits = {
                    1: 9.0 * epsilon / alpha,
                    2: 10.0 * epsilon / alpha,
                    3: 14.0 * epsilon,
                }
            notes.append(f"α = {alpha:.3f}")
        else:
            # Pure bending (tip in compression, root in tension)
            limits = {1: 9.0 * epsilon, 2: 10.0 * epsilon, 3: 14.0 * epsilon}

    elif inp.part_type == "angle":
        # Angles: Class 3 limits only
        h = float(inp.h_mm) if inp.h_mm else float(inp.c_mm)
        b = float(inp.b_mm) if inp.b_mm else float(inp.c_mm)
        t = float(inp.t_mm)
        ht = h / t
        bh_2t = (b + h) / (2.0 * t)

        is_class3 = ht <= 15.0 * epsilon and bh_2t <= 11.5 * epsilon
        limits = {3: 15.0 * epsilon}  # h/t limit
        notes.append(f"h/t = {ht:.2f} ≤ 15ε = {15.0 * epsilon:.2f}: {ht <= 15.0 * epsilon}")
        notes.append(
            f"(b+h)/(2t) = {bh_2t:.2f} ≤ 11.5ε = {11.5 * epsilon:.2f}: {bh_2t <= 11.5 * epsilon}"
        )

        section_class = 3 if is_class3 else 4
        return {
            "inputs_used": {
                "part_type": inp.part_type,
                "h_mm": h,
                "b_mm": b,
                "t_mm": t,
                "fy_mpa": fy,
            },
            "outputs": {
                "section_class": section_class,
                "h_over_t": round(ht, 2),
                "bh_over_2t": round(bh_2t, 2),
                "limit_h_t": round(15.0 * epsilon, 2),
                "limit_bh_2t": round(11.5 * epsilon, 2),
                "epsilon": round(epsilon, 4),
            },
            "clause_references": [
                {
                    "doc_id": "ec3.en1993-1-1.2005",
                    "clause_id": "5.5.2",
                    "title": "Table 5.2 – Classification limits (Angles)",
                    "pointer": "en_1993_1_1_2005_structured.json#5.5.2",
                },
            ],
            "notes": notes,
        }

    # Determine class
    section_class = 4
    for cls in [1, 2, 3]:
        if cls in limits and ct <= limits[cls]:
            section_class = cls
            break

    for cls in sorted(limits.keys()):
        notes.append(f"Class {cls} limit: c/t ≤ {limits[cls]:.2f} → {'OK' if ct <= limits[cls] else 'exceeded'}")

    notes.append(f"Section class = {section_class}")

    return {
        "inputs_used": {
            "part_type": inp.part_type,
            "stress_type": inp.stress_type,
            "c_mm": float(inp.c_mm),
            "t_mm": float(inp.t_mm),
            "fy_mpa": fy,
        },
        "outputs": {
            "c_over_t": round(ct, 2),
            "epsilon": round(epsilon, 4),
            "section_class": section_class,
            "class_1_limit": round(limits.get(1, float("inf")), 2) if 1 in limits else None,
            "class_2_limit": round(limits.get(2, float("inf")), 2) if 2 in limits else None,
            "class_3_limit": round(limits.get(3, float("inf")), 2) if 3 in limits else None,
        },
        "clause_references": [
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "5.5.2",
                "title": "Table 5.2 – Maximum width-to-thickness ratios",
                "pointer": "en_1993_1_1_2005_structured.json#5.5.2",
            },
        ],
        "notes": notes,
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=CtClassificationInput, handler=calculate)
