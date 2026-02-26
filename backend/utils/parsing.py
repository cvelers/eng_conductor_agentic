from __future__ import annotations

import re
from typing import Any

from backend.config import Settings

SECTION_PATTERN = re.compile(r"\b(IPE\s*\d+|HEA\s*\d+|HEB\s*\d+)\b", re.IGNORECASE)
STEEL_PATTERN = re.compile(r"\bS(235|275|355|420|460)\b", re.IGNORECASE)
LENGTH_PATTERN = re.compile(r"(?:\bL\b|length|span)\s*=?\s*(\d+(?:\.\d+)?)\s*(mm|m|cm)?", re.IGNORECASE)
MED_PATTERN = re.compile(r"(?:M(?:_?ed)?|moment)\s*=?\s*(\d+(?:\.\d+)?)\s*(kNm|knm|Nm|nm)?", re.IGNORECASE)
NED_PATTERN = re.compile(r"(?:N(?:_?ed)?|axial)\s*=?\s*(\d+(?:\.\d+)?)\s*(kN|kn|N|n)?", re.IGNORECASE)
VEd_PATTERN = re.compile(r"(?:V(?:_?ed)?|shear)\s*=?\s*(\d+(?:\.\d+)?)\s*(kN|kn|N|n)?", re.IGNORECASE)
SECTION_CLASS_PATTERN = re.compile(r"\bclass\s*([1-4])\b", re.IGNORECASE)

UDL_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:kN/m|kn/m)\b", re.IGNORECASE)
POINT_LOAD_PATTERN = re.compile(r"(?:point\s*load|P|load)\s*=?\s*(\d+(?:\.\d+)?)\s*(?:kN|kn)\b", re.IGNORECASE)
BOLT_CLASS_PATTERN = re.compile(r"\b(4\.6|4\.8|5\.6|5\.8|6\.8|8\.8|10\.9)\b")
BOLT_DIAMETER_PATTERN = re.compile(r"\bM\s*(\d+)\b", re.IGNORECASE)
N_BOLTS_PATTERN = re.compile(r"(\d+)\s*(?:×|x|bolts?\b)", re.IGNORECASE)
THROAT_PATTERN = re.compile(r"(?:throat|a)\s*=?\s*(\d+(?:\.\d+)?)\s*mm", re.IGNORECASE)
WELD_LENGTH_PATTERN = re.compile(r"(?:weld\s*)?length\s*=?\s*(\d+(?:\.\d+)?)\s*mm", re.IGNORECASE)
DEFLECTION_PATTERN = re.compile(r"(?:deflection|delta)\s*=?\s*(\d+(?:\.\d+)?)\s*mm", re.IGNORECASE)
LIMIT_RATIO_PATTERN = re.compile(r"\bL/(\d+)\b")


def normalize_section_name(raw: str) -> str:
    return raw.upper().replace(" ", "")


def parse_user_inputs(query: str) -> dict[str, Any]:
    found: dict[str, Any] = {}

    if section_match := SECTION_PATTERN.search(query):
        found["section_name"] = normalize_section_name(section_match.group(1))

    if steel_match := STEEL_PATTERN.search(query):
        found["steel_grade"] = f"S{steel_match.group(1)}"

    if length_match := LENGTH_PATTERN.search(query):
        val = float(length_match.group(1))
        unit = (length_match.group(2) or "m").lower()
        if unit == "mm":
            val /= 1000.0
        elif unit == "cm":
            val /= 100.0
        found["length_m"] = val

    if med_match := MED_PATTERN.search(query):
        val = float(med_match.group(1))
        unit = (med_match.group(2) or "kNm").lower()
        if unit == "nm":
            val /= 1000.0
        found["MEd_kNm"] = val

    if ned_match := NED_PATTERN.search(query):
        val = float(ned_match.group(1))
        unit = (ned_match.group(2) or "kN").lower()
        if unit == "n":
            val /= 1000.0
        found["NEd_kN"] = val

    if ved_match := VEd_PATTERN.search(query):
        val = float(ved_match.group(1))
        unit = (ved_match.group(2) or "kN").lower()
        if unit == "n":
            val /= 1000.0
        found["VEd_kN"] = val

    if class_match := SECTION_CLASS_PATTERN.search(query):
        found["section_class"] = int(class_match.group(1))

    if udl_match := UDL_PATTERN.search(query):
        found["load_kn_per_m"] = float(udl_match.group(1))

    if point_match := POINT_LOAD_PATTERN.search(query):
        found["load_kn"] = float(point_match.group(1))

    if bolt_class_match := BOLT_CLASS_PATTERN.search(query):
        found["bolt_class"] = bolt_class_match.group(1)

    if bolt_diam_match := BOLT_DIAMETER_PATTERN.search(query):
        found["bolt_diameter_mm"] = int(bolt_diam_match.group(1))

    if n_bolts_match := N_BOLTS_PATTERN.search(query):
        found["n_bolts"] = int(n_bolts_match.group(1))

    if throat_match := THROAT_PATTERN.search(query):
        found["throat_thickness_mm"] = float(throat_match.group(1))

    if defl_match := DEFLECTION_PATTERN.search(query):
        found["actual_deflection_mm"] = float(defl_match.group(1))

    if limit_match := LIMIT_RATIO_PATTERN.search(query):
        found["limit_ratio"] = f"L/{limit_match.group(1)}"

    return found


def apply_defaults(
    query: str,
    user_inputs: dict[str, Any],
    settings: Settings,
    requires_tools: bool,
) -> tuple[dict[str, Any], list[str]]:
    assumed: dict[str, Any] = {}
    assumptions: list[str] = []

    if not requires_tools:
        return assumed, assumptions

    lowered = query.lower()

    if "steel_grade" not in user_inputs:
        assumed["steel_grade"] = settings.default_steel_grade
        assumptions.append(
            f"Steel grade assumed as {settings.default_steel_grade} (typical structural grade)."
        )

    if "section_name" not in user_inputs:
        assumed["section_name"] = settings.default_section_name
        assumptions.append(
            f"Section assumed as {settings.default_section_name} (common rolled I-section)."
        )

    if "gamma_M0" not in user_inputs:
        assumed["gamma_M0"] = settings.default_gamma_m0
        assumptions.append(f"γ_M0 = {settings.default_gamma_m0:.2f} (standard EC3 value).")

    if (
        "section_class" not in user_inputs
        and any(token in lowered for token in ["class", "classification", "m_rd", "moment resistance", "bending resistance"])
    ):
        assumed["section_class"] = 2
        assumptions.append("Section class assumed as Class 2 (compact — plastic resistance basis).")

    interaction_like = "interaction" in lowered or (
        "combined" in lowered and ("bending" in lowered or "axial" in lowered)
    )
    if interaction_like and "MEd_kNm" not in user_inputs:
        assumed["MEd_kNm"] = settings.default_med_knm
        assumptions.append(f"M_Ed = {settings.default_med_knm:.1f} kNm assumed (typical service value).")

    if interaction_like and "NEd_kN" not in user_inputs:
        assumed["NEd_kN"] = settings.default_ned_kn
        assumptions.append(f"N_Ed = {settings.default_ned_kn:.1f} kN assumed (typical service value).")

    return assumed, assumptions
