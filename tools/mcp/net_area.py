from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt

from tools.mcp.cli import run_cli
from tools.mcp.clause_ref import clause_ref

TOOL_NAME = "net_area_ec3"


class HoleRow(BaseModel):
    """Describes a row of bolt holes."""

    n_holes: PositiveInt = Field(description="Number of holes in this row")
    d_0_mm: PositiveFloat = Field(description="Hole diameter d_0 in mm")


class StaggeredPath(BaseModel):
    """Describes a staggered bolt hole pattern along a failure path."""

    n_holes: PositiveInt = Field(description="Number of holes along this path")
    d_0_mm: PositiveFloat = Field(description="Hole diameter d_0 in mm")
    s_mm: list[float] = Field(description="List of stagger pitches s (spacing parallel to load) in mm")
    p_mm: list[float] = Field(description="List of gauge spacings p (spacing perpendicular to load) in mm")


class NetAreaInput(BaseModel):
    """Input for §6.2.2.2 – Net area calculation for bolt holes."""

    # Gross area
    A_gross_cm2: Optional[PositiveFloat] = Field(default=None, description="Gross area A in cm²")
    # Or plate dimensions
    width_mm: Optional[PositiveFloat] = Field(default=None, description="Plate width (perpendicular to force) in mm")
    thickness_mm: Optional[PositiveFloat] = Field(default=None, description="Plate thickness t in mm")

    # Non-staggered holes (simple deduction)
    holes: Optional[list[HoleRow]] = Field(
        default=None, description="List of hole rows for non-staggered pattern"
    )

    # Staggered holes
    staggered_paths: Optional[list[StaggeredPath]] = Field(
        default=None, description="List of candidate failure paths for staggered bolt pattern"
    )


def calculate(inp: NetAreaInput) -> dict:
    # Determine gross area and thickness
    if inp.width_mm and inp.thickness_mm:
        t = float(inp.thickness_mm)
        w = float(inp.width_mm)
        A_gross = w * t  # mm²
    elif inp.A_gross_cm2:
        A_gross = float(inp.A_gross_cm2) * 100.0  # mm²
        t = float(inp.thickness_mm) if inp.thickness_mm else None
    else:
        raise ValueError("Provide width_mm + thickness_mm, or A_gross_cm2.")

    notes: list[str] = [f"A_gross = {A_gross:.1f} mm²"]
    results: dict = {"A_gross_mm2": round(A_gross, 1), "A_gross_cm2": round(A_gross / 100.0, 2)}

    if inp.holes and not inp.staggered_paths:
        # §6.2.2.2(3) – Non-staggered: deduct n·d0·t
        if t is None:
            raise ValueError("Non-staggered hole deduction requires thickness_mm.")
        total_deduction = 0.0
        for row in inp.holes:
            deduction = row.n_holes * row.d_0_mm * t
            total_deduction += deduction

        A_net = A_gross - total_deduction
        results["deduction_mm2"] = round(total_deduction, 1)
        results["A_net_mm2"] = round(max(A_net, 0.0), 1)
        results["A_net_cm2"] = round(max(A_net, 0.0) / 100.0, 2)
        notes.append(f"Deduction = Σ(n·d₀·t) = {total_deduction:.1f} mm²")
        notes.append(f"A_net = A_gross − deduction = {A_net:.1f} mm²")

    elif inp.staggered_paths:
        # §6.2.2.2(4) – Staggered holes: t·(n·d0 − Σ(s²/(4p)))
        if t is None:
            raise ValueError("Staggered hole calculation requires thickness_mm.")

        min_A_net = float("inf")
        path_results: list[dict] = []

        for i, path in enumerate(inp.staggered_paths):
            hole_deduction = path.n_holes * path.d_0_mm
            s2_4p_sum = sum(s**2 / (4.0 * p) for s, p in zip(path.s_mm, path.p_mm) if p > 0)

            net_width_deduction = hole_deduction - s2_4p_sum
            A_net_path = A_gross - t * net_width_deduction

            path_results.append({
                "path_index": i + 1,
                "hole_deduction_mm": round(hole_deduction, 2),
                "s2_4p_sum_mm": round(s2_4p_sum, 2),
                "net_deduction_mm": round(net_width_deduction, 2),
                "A_net_mm2": round(max(A_net_path, 0.0), 1),
            })

            if A_net_path < min_A_net:
                min_A_net = A_net_path

        results["staggered_paths"] = path_results
        results["A_net_mm2"] = round(max(min_A_net, 0.0), 1)
        results["A_net_cm2"] = round(max(min_A_net, 0.0) / 100.0, 2)
        results["governing_path"] = min(
            range(len(path_results)), key=lambda i: path_results[i]["A_net_mm2"]
        ) + 1
        notes.append(f"Governing A_net = {min_A_net:.1f} mm² (path {results['governing_path']})")

    else:
        # No holes – net area equals gross area
        results["A_net_mm2"] = round(A_gross, 1)
        results["A_net_cm2"] = round(A_gross / 100.0, 2)
        notes.append("No holes specified – A_net = A_gross")

    return {
        "inputs_used": {
            "width_mm": float(inp.width_mm) if inp.width_mm else None,
            "thickness_mm": float(inp.thickness_mm) if inp.thickness_mm else None,
            "A_gross_cm2": float(inp.A_gross_cm2) if inp.A_gross_cm2 else None,
        },
        "outputs": results,
        "clause_references": [
            clause_ref("ec3.en1993-1-1.2005", "6.2.2.2", "Net area"),
        ],
        "notes": notes,
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=NetAreaInput, handler=calculate)
