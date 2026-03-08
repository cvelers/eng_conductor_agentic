"""Tool output validation layer.

Validates tool outputs for physical reasonableness and consistency.
Injected immediately after tool_runner.run() returns in agent_loop.py.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Physically reasonable ranges for common output keys
_RANGE_CHECKS: dict[str, tuple[float, float]] = {
    "governing_class": (1, 4),
    "web_class": (1, 4),
    "flange_class": (1, 4),
    "chi": (0.0, 1.0),
    "lambda_bar": (0.0, 10.0),
    "Nb_Rd_kN": (0.0, 100_000.0),
    "M_Rd_kNm": (0.0, 100_000.0),
    "V_Rd_kN": (0.0, 100_000.0),
    "N_t_Rd_kN": (0.0, 100_000.0),
    "N_pl_Rd_kN": (0.0, 100_000.0),
    "N_u_Rd_kN": (0.0, 100_000.0),
    "N_cr_kN": (0.0, 1_000_000.0),
    "Fv_Rd_per_bolt_kN": (0.0, 2_000.0),
    "Fv_Rd_total_kN": (0.0, 50_000.0),
    "utilization": (0.0, 50.0),
}

# Physically reasonable ranges for intermediate / inputs_used values
_INPUT_CHECKS: dict[str, tuple[float, float]] = {
    "epsilon": (0.5, 1.5),
    "fy_mpa": (200.0, 700.0),
    "fub_mpa": (300.0, 1200.0),
    "alpha_v": (0.4, 0.7),
}


class ToolOutputValidator:
    """Validates tool outputs for physical reasonableness."""

    def validate(
        self,
        tool_name: str,
        result: dict[str, Any],
        inputs: dict[str, Any],
    ) -> list[str]:
        """Return list of warning strings. Empty = all OK."""
        warnings: list[str] = []
        outputs = result.get("outputs", {})
        intermediate = result.get("intermediate", {})
        inputs_used = result.get("inputs_used", {})

        # Range checks on outputs
        for key, (lo, hi) in _RANGE_CHECKS.items():
            val = outputs.get(key)
            if val is not None and isinstance(val, (int, float)):
                if val < lo or val > hi:
                    warnings.append(
                        f"{tool_name}: {key}={val} outside expected range [{lo}, {hi}]"
                    )

        # Range checks on intermediate values
        for key, (lo, hi) in _INPUT_CHECKS.items():
            for source in (intermediate, inputs_used):
                val = source.get(key)
                if val is not None and isinstance(val, (int, float)):
                    if val < lo or val > hi:
                        warnings.append(
                            f"{tool_name}: {key}={val} outside expected range [{lo}, {hi}]"
                        )

        # Tool-specific cross-checks
        if tool_name == "column_buckling_ec3":
            warnings.extend(self._check_column_buckling(result, inputs))
        elif tool_name == "section_classification_ec3":
            warnings.extend(self._check_section_classification(result, inputs))

        return warnings

    def _check_column_buckling(
        self, result: dict[str, Any], inputs: dict[str, Any]
    ) -> list[str]:
        """Cross-check column buckling specifics."""
        warnings: list[str] = []
        inputs_used = result.get("inputs_used", {})
        outputs = result.get("outputs", {})

        # chi should decrease as lambda_bar increases
        chi = outputs.get("chi")
        lam = outputs.get("lambda_bar")
        if chi is not None and lam is not None:
            if lam > 2.0 and chi > 0.5:
                warnings.append(
                    f"column_buckling_ec3: chi={chi:.4f} seems high for lambda_bar={lam:.4f}"
                )

        # Nb_Rd should not exceed squash load
        nb = outputs.get("Nb_Rd_kN")
        area = inputs_used.get("area_cm2")
        fy = inputs_used.get("fy_mpa")
        if nb and area and fy:
            squash = area * 100 * fy / 1000  # A_mm2 * fy / 1000
            if nb > squash * 1.01:  # 1% tolerance
                warnings.append(
                    f"column_buckling_ec3: Nb_Rd={nb:.1f} kN exceeds squash load {squash:.1f} kN"
                )

        return warnings

    def _check_section_classification(
        self, result: dict[str, Any], inputs: dict[str, Any]
    ) -> list[str]:
        """Cross-check section classification specifics."""
        warnings: list[str] = []
        outputs = result.get("outputs", {})

        web_class = outputs.get("web_class")
        flange_class = outputs.get("flange_class")
        governing = outputs.get("governing_class")

        if web_class and flange_class and governing:
            expected = max(web_class, flange_class)
            if governing != expected:
                warnings.append(
                    f"section_classification_ec3: governing_class={governing} "
                    f"but max(web={web_class}, flange={flange_class})={expected}"
                )

        return warnings
