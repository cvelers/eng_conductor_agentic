from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, PositiveFloat

from tools.mcp.cli import run_cli

TOOL_NAME = "hollow_section_buckling_ec3"


class HollowSectionBucklingInput(BaseModel):
    """Input for BB.1.3 – Buckling lengths for hollow section members in trusses."""

    member_role: Literal["chord", "brace_bolted", "brace_welded"] = Field(
        description="Member role: 'chord', 'brace_bolted' (bolted connections), "
        "'brace_welded' (welded around perimeter to chords)"
    )
    plane: Literal["in_plane", "out_of_plane"] = Field(
        default="in_plane",
        description="Buckling plane: 'in_plane' or 'out_of_plane'",
    )
    L_system_mm: PositiveFloat = Field(
        description="System length L in mm (distance between joints for in-plane, "
        "distance between lateral supports for out-of-plane)"
    )


def calculate(inp: HollowSectionBucklingInput) -> dict:
    L = float(inp.L_system_mm)
    notes: list[str] = []

    if inp.member_role == "chord":
        # BB.1.3(1)B – Chord: L_cr = 0.9L
        factor = 0.9
        notes.append("BB.1.3(1)B: Hollow section chord – L_cr = 0.9·L")
    elif inp.member_role == "brace_bolted":
        # BB.1.3(2)B – Bolted brace: L_cr = 1.0L
        factor = 1.0
        notes.append("BB.1.3(2)B: Hollow section brace (bolted) – L_cr = 1.0·L")
    elif inp.member_role == "brace_welded":
        # BB.1.3(3)B – Welded brace: L_cr = 0.75L
        factor = 0.75
        notes.append(
            "BB.1.3(3)B: Hollow section brace (welded around perimeter) – L_cr = 0.75·L"
        )
    else:
        raise ValueError(f"Unknown member_role: {inp.member_role}")

    L_cr = factor * L
    notes.append(f"L_cr = {factor}·{L:.1f} = {L_cr:.1f} mm = {L_cr / 1000.0:.3f} m")

    return {
        "inputs_used": {
            "member_role": inp.member_role,
            "plane": inp.plane,
            "L_system_mm": L,
        },
        "outputs": {
            "buckling_length_factor": factor,
            "L_cr_mm": round(L_cr, 1),
            "L_cr_m": round(L_cr / 1000.0, 3),
        },
        "clause_references": [
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "BB.1.3",
                "title": "Hollow sections as web members and chord members",
                "pointer": "en_1993_1_1_2005_structured.json#BB.1.3",
            },
        ],
        "notes": notes,
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=HollowSectionBucklingInput, handler=calculate)
