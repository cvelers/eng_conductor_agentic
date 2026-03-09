from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli
from tools.mcp.clause_ref import clause_ref

TOOL_NAME = "taper_factor_ec3"


class TaperFactorInput(BaseModel):
    """Input for BB.3.3.3 – Taper factor c for non-uniform members."""

    member_type: Literal["tapered", "haunched"] = Field(
        description="Type of non-uniform member: 'tapered' or 'haunched'"
    )

    # Section properties of shallowest section
    h_mm: PositiveFloat = Field(description="Depth h of the shallowest section in mm")
    tf_mm: PositiveFloat = Field(description="Flange thickness t_f of the shallowest section in mm")
    b_mm: Optional[PositiveFloat] = Field(
        default=None, description="Flange width b in mm (for validation: h ≥ 1.2b)"
    )

    # For tapered members
    h_max_mm: Optional[PositiveFloat] = Field(
        default=None, description="Maximum depth of cross-section within L_y in mm (for tapered)"
    )
    h_min_mm: Optional[PositiveFloat] = Field(
        default=None, description="Minimum depth of cross-section within L_y in mm (for tapered)"
    )

    # For haunched members
    h_h_mm: Optional[PositiveFloat] = Field(
        default=None, description="Additional depth of haunch h_h in mm"
    )
    h_s_mm: Optional[PositiveFloat] = Field(
        default=None, description="Depth of un-haunched section h_s in mm"
    )
    L_h_mm: Optional[PositiveFloat] = Field(
        default=None, description="Length of haunch within L_y in mm"
    )
    L_y_mm: Optional[PositiveFloat] = Field(
        default=None,
        description="Length between points where compression flange is laterally restrained in mm",
    )


def calculate(inp: TaperFactorInput) -> dict:
    h = float(inp.h_mm)
    tf = float(inp.tf_mm)
    h_tf = h / tf
    notes: list[str] = []

    # Validation: h ≥ 1.2b and h/tf ≥ 20
    if inp.b_mm:
        b = float(inp.b_mm)
        if h < 1.2 * b:
            notes.append(f"Warning: h = {h:.1f} < 1.2·b = {1.2 * b:.1f} – condition not met")
    if h_tf < 20.0:
        notes.append(f"Warning: h/t_f = {h_tf:.2f} < 20 – condition not met")

    notes.append(f"(h/t_f) = {h_tf:.2f}")

    denom = h_tf - 9.0
    if denom <= 0:
        raise ValueError(f"h/t_f = {h_tf:.2f} must be > 9 for taper factor calculation.")

    if inp.member_type == "tapered":
        # BB.3.3.3 – Tapered members
        if inp.h_max_mm is None or inp.h_min_mm is None:
            raise ValueError("Tapered members require h_max_mm and h_min_mm.")

        h_max = float(inp.h_max_mm)
        h_min = float(inp.h_min_mm)

        if h_min <= 0 or h_max < h_min:
            raise ValueError("h_max must be > h_min > 0.")

        # c = 1 + (3 / (h/tf - 9)) · (h_max/h_min - 1)^(2/3)
        ratio = h_max / h_min - 1.0
        c = 1.0 + 3.0 / denom * ratio ** (2.0 / 3.0) if ratio > 0 else 1.0

        notes.append(f"h_max/h_min = {h_max / h_min:.3f}")
        notes.append(f"c = 1 + 3/(h/t_f − 9)·(h_max/h_min − 1)^(2/3) = {c:.4f}")

    else:
        # BB.3.3.3 – Haunched members
        if inp.h_h_mm is None or inp.h_s_mm is None or inp.L_h_mm is None or inp.L_y_mm is None:
            raise ValueError("Haunched members require h_h_mm, h_s_mm, L_h_mm, L_y_mm.")

        h_h = float(inp.h_h_mm)
        h_s = float(inp.h_s_mm)
        L_h = float(inp.L_h_mm)
        L_y = float(inp.L_y_mm)

        if h_s <= 0 or L_y <= 0:
            raise ValueError("h_s and L_y must be > 0.")

        # c = 1 + (3 / (h/tf - 9)) · (h_h/h_s)^(2/3) · √(L_h/L_y)
        c = 1.0 + 3.0 / denom * (h_h / h_s) ** (2.0 / 3.0) * math.sqrt(L_h / L_y)

        notes.append(f"h_h/h_s = {h_h / h_s:.3f}, L_h/L_y = {L_h / L_y:.3f}")
        notes.append(
            f"c = 1 + 3/(h/t_f − 9)·(h_h/h_s)^(2/3)·√(L_h/L_y) = {c:.4f}"
        )

    return {
        "inputs_used": {
            "member_type": inp.member_type,
            "h_mm": h,
            "tf_mm": tf,
            "h_over_tf": round(h_tf, 2),
        },
        "outputs": {
            "c": round(c, 4),
        },
        "clause_references": [
            clause_ref("ec3.en1993-1-1.2005", "BB.3.3.3", "Taper factor"),
        ],
        "notes": notes,
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=TaperFactorInput, handler=calculate)
