from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt

from tools.mcp.cli import run_cli

TOOL_NAME = "bolt_shear_ec3"

BOLT_PROPERTIES = {
    "4.6": {"fub": 400, "alpha_v": 0.6},
    "4.8": {"fub": 400, "alpha_v": 0.6},
    "5.6": {"fub": 500, "alpha_v": 0.6},
    "5.8": {"fub": 500, "alpha_v": 0.6},
    "6.8": {"fub": 600, "alpha_v": 0.6},
    "8.8": {"fub": 800, "alpha_v": 0.6},
    "10.9": {"fub": 1000, "alpha_v": 0.5},
}

BOLT_AREAS = {
    12: {"A": 113.1, "As": 84.3},
    14: {"A": 153.9, "As": 115.0},
    16: {"A": 201.1, "As": 157.0},
    18: {"A": 254.5, "As": 192.0},
    20: {"A": 314.2, "As": 245.0},
    22: {"A": 380.1, "As": 303.0},
    24: {"A": 452.4, "As": 353.0},
    27: {"A": 572.6, "As": 459.0},
    30: {"A": 706.9, "As": 561.0},
    36: {"A": 1017.9, "As": 817.0},
}

# Standard clearances per EN 1090-2 Table 11
STANDARD_CLEARANCE_MM: dict[int, int] = {
    12: 1, 14: 1, 16: 2, 18: 2, 20: 2, 22: 2, 24: 2,
    27: 3, 30: 3, 33: 3, 36: 3,
}


def bolt_hole_diameter(bolt_diameter_mm: int) -> float:
    """Return standard clearance hole diameter d₀ per EN 1090-2 Table 11."""
    clearance = STANDARD_CLEARANCE_MM.get(bolt_diameter_mm, 2 if bolt_diameter_mm <= 24 else 3)
    return float(bolt_diameter_mm + clearance)


class BoltShearInput(BaseModel):
    bolt_class: Literal["4.6", "4.8", "5.6", "5.8", "6.8", "8.8", "10.9"] = Field(
        default="8.8", description="Bolt property class per EC3-1-8"
    )
    bolt_diameter_mm: Literal[12, 14, 16, 18, 20, 22, 24, 27, 30, 36] = Field(
        default=20, description="Nominal bolt diameter in mm"
    )
    n_shear_planes: PositiveInt = Field(default=1, description="Number of shear planes")
    shear_through_threads: bool = Field(
        default=True,
        description="True if shear plane passes through the threaded portion (uses As instead of A)",
    )
    gamma_M2: PositiveFloat = Field(default=1.25, description="Partial safety factor γ_M2")
    n_bolts: PositiveInt = Field(default=1, description="Number of bolts in the connection")


def calculate(inp: BoltShearInput) -> dict:
    props = BOLT_PROPERTIES[inp.bolt_class]
    areas = BOLT_AREAS[inp.bolt_diameter_mm]

    fub = props["fub"]
    alpha_v = props["alpha_v"]
    A = areas["As"] if inp.shear_through_threads else areas["A"]
    area_label = "As (tensile stress area)" if inp.shear_through_threads else "A (gross area)"

    Fv_Rd_single = alpha_v * fub * A / (inp.gamma_M2 * 1000)  # kN per shear plane per bolt
    Fv_Rd_bolt = Fv_Rd_single * inp.n_shear_planes
    Fv_Rd_total = Fv_Rd_bolt * inp.n_bolts

    return {
        "inputs_used": {
            "bolt_class": inp.bolt_class,
            "bolt_diameter_mm": inp.bolt_diameter_mm,
            "n_shear_planes": inp.n_shear_planes,
            "shear_through_threads": inp.shear_through_threads,
            "gamma_M2": inp.gamma_M2,
            "n_bolts": inp.n_bolts,
        },
        "intermediate": {
            "fub_mpa": fub,
            "alpha_v": alpha_v,
            "A_mm2": round(A, 1),
            "area_used": area_label,
        },
        "outputs": {
            "Fv_Rd_per_bolt_kN": round(Fv_Rd_bolt, 2),
            "Fv_Rd_total_kN": round(Fv_Rd_total, 2),
            "Fv_Rd_single_plane_kN": round(Fv_Rd_single, 2),
        },
        "clause_references": [
            {
                "doc_id": "ec3.en1993-1-8.2005",
                "clause_id": "Table 3.4",
                "title": "Design resistance for bolts in shear",
                "pointer": "en_1993_1_8#table_3.4",
            },
        ],
        "notes": [
            f"Fv,Rd = αv × fub × A / γM2 = {alpha_v} × {fub} × {A:.1f} / {inp.gamma_M2} = {Fv_Rd_single:.2f} kN per plane",
            "Bearing resistance should be checked separately.",
        ],
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=BoltShearInput, handler=calculate)
