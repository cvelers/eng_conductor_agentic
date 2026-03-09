from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from tools.mcp.cli import run_cli
from tools.mcp.clause_ref import clause_ref

TOOL_NAME = "cm_factor_b_ec3"


class CmFactorBInput(BaseModel):
    """Input for Table B.3 – Equivalent uniform moment factors C_m for Annex B Method 2."""

    moment_type: Literal["end_moments", "span_load"] = Field(
        description="'end_moments' for linear moment diagram (ψ·M), "
        "'span_load' for transverse loading between restraints"
    )

    # For end_moments
    psi: Optional[float] = Field(
        default=None,
        description="Moment ratio ψ = M_min/M_max (-1 ≤ ψ ≤ 1) for linear moment diagram",
    )

    # For span_load
    load_type: Optional[Literal["uniform", "concentrated"]] = Field(
        default=None,
        description="Type of transverse load: 'uniform' or 'concentrated'",
    )
    alpha_h: Optional[float] = Field(
        default=None,
        description="Ratio α_h = M_h/M_s (hogging/span moment ratio, 0 ≤ α_h ≤ 1)",
    )
    alpha_s: Optional[float] = Field(
        default=None,
        description="Ratio α_s = M_s/M_h (span/hogging moment ratio, 0 ≤ α_s ≤ 1)",
    )
    psi_span: Optional[float] = Field(
        default=None,
        description="Moment ratio ψ for span loading cases (-1 ≤ ψ ≤ 1)",
    )

    sway_mode: bool = Field(
        default=False,
        description="If True, C_m = 0.9 for sway buckling mode",
    )


def calculate(inp: CmFactorBInput) -> dict:
    notes: list[str] = []

    if inp.sway_mode:
        C_m = 0.9
        notes.append("Sway buckling mode: C_m = 0.9")
        formula = "C_m = 0.9 (sway mode)"
    elif inp.moment_type == "end_moments":
        if inp.psi is None:
            raise ValueError("psi is required for end_moments type.")
        psi = float(inp.psi)
        C_m = max(0.6 + 0.4 * psi, 0.4)
        formula = f"C_m = max(0.6 + 0.4·ψ, 0.4) = max(0.6 + 0.4·{psi:.3f}, 0.4)"
        notes.append(f"Linear moment diagram: ψ = {psi:.3f}")
        notes.append(formula)
    elif inp.moment_type == "span_load":
        if inp.load_type is None:
            raise ValueError("load_type required for span_load type.")

        psi = float(inp.psi_span) if inp.psi_span is not None else 1.0

        if inp.alpha_h is not None:
            ah = float(inp.alpha_h)
            if inp.load_type == "uniform":
                if psi >= 0:
                    C_m = 0.95 + 0.05 * ah
                else:
                    C_m = 0.95 + 0.05 * ah * (1.0 + 2.0 * psi)
            else:
                if psi >= 0:
                    C_m = 0.90 + 0.10 * ah
                else:
                    C_m = 0.90 + 0.10 * ah * (1.0 + 2.0 * psi)
            formula = f"α_h = {ah:.3f}, load = {inp.load_type}"
            notes.append(f"Span load with α_h: {formula}")

        elif inp.alpha_s is not None:
            a_s = float(inp.alpha_s)
            if a_s >= 0:
                if inp.load_type == "uniform":
                    C_m = max(0.2 + 0.8 * a_s, 0.4)
                else:
                    C_m = max(0.2 + 0.8 * a_s, 0.4)
            else:
                if psi >= 0:
                    if inp.load_type == "uniform":
                        C_m = max(0.1 - 0.8 * a_s, 0.4)
                    else:
                        C_m = max(-0.8 * a_s, 0.4)
                else:
                    if inp.load_type == "uniform":
                        C_m = max(0.1 * (1.0 - psi) - 0.8 * a_s, 0.4)
                    else:
                        C_m = max(0.2 * (-psi) - 0.8 * a_s, 0.4)
            formula = f"α_s = {a_s:.3f}, load = {inp.load_type}"
            notes.append(f"Span load with α_s: {formula}")
        else:
            raise ValueError("Either alpha_h or alpha_s required for span_load type.")

        C_m = max(C_m, 0.4)
        notes.append(f"C_m ≥ 0.4 → C_m = {C_m:.4f}")
    else:
        raise ValueError("Invalid moment_type.")

    notes.append(f"C_m = {C_m:.4f}")

    return {
        "inputs_used": {
            "moment_type": inp.moment_type,
            "psi": float(inp.psi) if inp.psi is not None else None,
            "load_type": inp.load_type,
            "sway_mode": inp.sway_mode,
        },
        "outputs": {
            "C_m": round(C_m, 4),
        },
        "clause_references": [
            clause_ref("ec3.en1993-1-1.2005", "B.3", "Table B.3 – Equivalent uniform moment factors C_m"),
        ],
        "notes": notes,
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=CmFactorBInput, handler=calculate)
