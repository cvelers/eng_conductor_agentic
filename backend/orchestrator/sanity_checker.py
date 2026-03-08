"""End-to-end sanity checker for tool chain outputs.

Cross-validates outputs across multiple tools run in the same query,
flagging inconsistencies before the final answer is composed.

Injected in agent_loop.py after all tasks complete, before Phase 3
finalization.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SanityChecker:
    """Cross-validates tool chain outputs and flags inconsistencies."""

    def check(
        self,
        all_tool_outputs: dict[str, dict[str, Any]],
        tool_trace: list,
    ) -> list[str]:
        warnings: list[str] = []

        # 1. Section class consistency
        classification = all_tool_outputs.get("section_classification_ec3", {})
        cls = classification.get("outputs", {}).get("governing_class")

        for tool_name in ("member_resistance_ec3", "ipe_moment_resistance_ec3", "column_buckling_ec3"):
            other = all_tool_outputs.get(tool_name, {})
            used_cls = other.get("inputs_used", {}).get("section_class")
            if cls and used_cls and cls != used_cls:
                warnings.append(
                    f"Section class mismatch: classification gave {cls}, "
                    f"but {tool_name} used {used_cls}"
                )

        # 2. fy consistency across tools
        fy_values: dict[str, float] = {}
        for tool_name, result in all_tool_outputs.items():
            fy = result.get("inputs_used", {}).get("fy_mpa")
            if fy is not None:
                fy_values[tool_name] = float(fy)
        unique_fy = set(fy_values.values())
        if len(unique_fy) > 1:
            detail = ", ".join(f"{t}={v}" for t, v in fy_values.items())
            warnings.append(f"Inconsistent fy across tools: {detail}")

        # 3. Magnitude sanity for structural resistances
        _magnitude_checks = [
            ("M_Rd_kNm", 5, 10_000),
            ("Nb_Rd_kN", 5, 50_000),
            ("N_t_Rd_kN", 5, 50_000),
            ("N_pl_Rd_kN", 5, 50_000),
            ("V_Rd_kN", 5, 50_000),
        ]
        for tool_name, result in all_tool_outputs.items():
            outputs = result.get("outputs", {})
            for key, lo, hi in _magnitude_checks:
                val = outputs.get(key)
                if val is not None and isinstance(val, (int, float)):
                    if val < lo or val > hi:
                        warnings.append(
                            f"Suspicious {key}={val:.1f} from {tool_name} "
                            f"(expected [{lo}, {hi}])"
                        )

        # 4. Steel grade consistency
        grades: dict[str, str] = {}
        for tool_name, result in all_tool_outputs.items():
            grade = result.get("inputs_used", {}).get("steel_grade")
            if grade:
                grades[tool_name] = str(grade)
        unique_grades = set(grades.values())
        if len(unique_grades) > 1:
            detail = ", ".join(f"{t}={g}" for t, g in grades.items())
            warnings.append(f"Inconsistent steel grades across tools: {detail}")

        return warnings
