"""Registry of eurocodepy capabilities exposed as engineering tools.

Each entry maps a tool name to a eurocodepy function (or adapter) with
its JSON Schema parameters, keywords for search, and clause references.

To add a new tool: append an ``EngToolEntry`` to ``ENGINEERING_TOOL_REGISTRY``.
No other code changes needed — search and dispatcher pick it up automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EngToolEntry:
    name: str
    category: str  # EC3 (more to come)
    subcategory: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    keywords: list[str]
    handler_module: str  # Python import path
    handler_function: str  # function name in that module
    clause_references: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ENGINEERING_TOOL_REGISTRY: list[EngToolEntry] = [
    # ── EC3  Steel — ULS checks ───────────────────────────────────────
    EngToolEntry(
        name="ec3_combined_section_check",
        category="EC3",
        subcategory="ULS_combined",
        description=(
            "EC3 combined section check for axial force, bending moment, and "
            "shear (EN 1993-1-1 §6.2). Returns utilization ratios for N, M, V "
            "and interaction check with pass/fail."
        ),
        parameters={
            "type": "object",
            "properties": {
                "N_Ed": {"type": "number", "description": "Design axial force (N)"},
                "M_Ed": {"type": "number", "description": "Design bending moment (Nmm)"},
                "V_Ed": {"type": "number", "description": "Design shear force (N)"},
                "area": {"type": "number", "description": "Cross-sectional area (mm²)"},
                "area_v": {"type": "number", "description": "Shear area (mm²)"},
                "W_el": {"type": "number", "description": "Elastic section modulus (mm³)"},
                "fy": {"type": "number", "description": "Yield strength (MPa)"},
                "gamma_M0": {"type": "number", "description": "Partial factor (default 1.0)"},
            },
            "required": ["N_Ed", "M_Ed", "V_Ed", "area", "area_v", "W_el", "fy"],
        },
        keywords=[
            "combined", "section check", "axial", "bending", "shear",
            "interaction", "utilization", "N_Ed", "M_Ed", "V_Ed", "steel",
            "ULS", "resistance", "6.2",
        ],
        handler_module="eurocodepy.ec3.uls",
        handler_function="eurocode3_combined_check",
        clause_references=["EN 1993-1-1 §6.2"],
    ),
    EngToolEntry(
        name="ec3_ltb_check",
        category="EC3",
        subcategory="ULS_LTB",
        description=(
            "Lateral-torsional buckling resistance check (EN 1993-1-1 §6.3.2). "
            "Computes Mcr, chi_LT, Mb,Rd and utilization ratio. "
            "Requires the design bending moment M_Ed from loading/actions; do not "
            "substitute a resistance or capacity value such as M_Rd or Mb,Rd."
        ),
        parameters={
            "type": "object",
            "properties": {
                "f_y": {"type": "number", "description": "Yield strength (MPa)"},
                "E": {"type": "number", "description": "Young's modulus (MPa), typically 210000"},
                "G": {"type": "number", "description": "Shear modulus (MPa), typically 81000"},
                "gamma_M1": {"type": "number", "description": "Partial factor (default 1.0)"},
                "I_y": {"type": "number", "description": "Major axis moment of inertia (mm⁴)"},
                "I_z": {"type": "number", "description": "Minor axis moment of inertia (mm⁴)"},
                "W_el_z": {"type": "number", "description": "Elastic section modulus about z-axis (mm³)"},
                "I_w": {"type": "number", "description": "Warping constant (mm⁶)"},
                "I_t": {"type": "number", "description": "Torsion constant (mm⁴)"},
                "L": {"type": "number", "description": "Unbraced length (mm)"},
                "M_Ed": {
                    "type": "number",
                    "description": (
                        "Design bending moment from loading/actions (kNm). "
                        "This is a demand effect, not a resistance/capacity."
                    ),
                },
                "C1": {"type": "number", "description": "Moment distribution factor (default 1.0)"},
                "alpha_LT": {"type": "number", "description": "Imperfection factor (default 0.34)"},
            },
            "required": ["f_y", "E", "G", "gamma_M1", "I_y", "I_z", "W_el_z", "I_w", "I_t", "L", "M_Ed"],
        },
        keywords=[
            "lateral torsional buckling", "LTB", "chi_LT", "Mcr", "Mb,Rd",
            "buckling resistance", "moment", "steel", "6.3.2",
        ],
        handler_module="eurocodepy.ec3.uls",
        handler_function="check_ltb_resistance",
        clause_references=["EN 1993-1-1 §6.3.2"],
    ),
    EngToolEntry(
        name="ec3_flexural_buckling_check",
        category="EC3",
        subcategory="ULS_buckling",
        description=(
            "Flexural buckling resistance check for compression members "
            "(EN 1993-1-1 §6.3.1). Returns Nb,Rd and utilization."
        ),
        parameters={
            "type": "object",
            "properties": {
                "N_Ed": {"type": "number", "description": "Design axial compression (N)"},
                "A": {"type": "number", "description": "Cross-sectional area (mm²)"},
                "fy": {"type": "number", "description": "Yield strength (MPa)"},
                "L_cr": {"type": "number", "description": "Buckling length (mm)"},
                "i": {"type": "number", "description": "Radius of gyration (mm) for the relevant axis"},
                "buckling_curve": {
                    "type": "string",
                    "description": "Buckling curve: 'a0', 'a', 'b', 'c', 'd' (default 'b')",
                },
                "gamma_M1": {"type": "number", "description": "Partial factor (default 1.0)"},
            },
            "required": ["N_Ed", "A", "fy", "L_cr", "i"],
        },
        keywords=[
            "flexural buckling", "column buckling", "compression", "Nb,Rd",
            "chi", "buckling curve", "slenderness", "6.3.1", "strut",
        ],
        handler_module="backend.eurocodepy.adapters",
        handler_function="ec3_flexural_buckling",
        clause_references=["EN 1993-1-1 §6.3.1"],
    ),
    EngToolEntry(
        name="ec3_elastic_critical_force",
        category="EC3",
        subcategory="ULS_buckling",
        description=(
            "Euler elastic critical force Ncr for flexural buckling. "
            "Ncr = π²·E·I / L²."
        ),
        parameters={
            "type": "object",
            "properties": {
                "E": {"type": "number", "description": "Young's modulus (MPa)"},
                "I": {"type": "number", "description": "Moment of inertia (mm⁴)"},
                "L": {"type": "number", "description": "Buckling length (mm)"},
                "K": {"type": "number", "description": "Effective length factor (default 1.0)"},
            },
            "required": ["E", "I", "L"],
        },
        keywords=[
            "Ncr", "Euler", "critical force", "elastic critical", "buckling load",
        ],
        handler_module="eurocodepy.ec3.uls",
        handler_function="calc_Ncr",
        clause_references=["EN 1993-1-1 §6.3.1"],
    ),

    # ── EC3  Steel — Profile & material lookups ───────────────────────
    EngToolEntry(
        name="ec3_profile_i_lookup",
        category="EC3",
        subcategory="profiles",
        description=(
            "Look up I-section profile properties (IPE, HEA, HEB, HEM). "
            "Returns full geometric and design properties: A, Iy, Iz, Wel, "
            "Wpl, IT, Iw, Av_z, fyd, Npl_Rd, Mpl_Rd, etc."
        ),
        parameters={
            "type": "object",
            "properties": {
                "profile_name": {
                    "type": "string",
                    "description": "Profile designation, e.g. 'IPE300', 'HEA200', 'HEB300', 'HEM200'.",
                },
            },
            "required": ["profile_name"],
        },
        keywords=[
            "IPE", "HEA", "HEB", "HEM", "I-section", "profile", "section properties",
            "area", "moment of inertia", "section modulus", "geometric",
        ],
        handler_module="backend.eurocodepy.adapters",
        handler_function="lookup_i_profile",
        clause_references=[],
    ),
    EngToolEntry(
        name="ec3_profile_chs_lookup",
        category="EC3",
        subcategory="profiles",
        description=(
            "Look up Circular Hollow Section (CHS) profile properties. "
            "162 profiles available. Provide designation like 'CHS139_7x5_0'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "profile_name": {
                    "type": "string",
                    "description": "CHS designation, e.g. 'CHS139_7x5_0'. Use dots replaced by underscores.",
                },
            },
            "required": ["profile_name"],
        },
        keywords=[
            "CHS", "circular hollow", "tube", "pipe", "hollow section", "profile",
        ],
        handler_module="backend.eurocodepy.adapters",
        handler_function="lookup_chs_profile",
        clause_references=[],
    ),
    EngToolEntry(
        name="ec3_profile_rhs_lookup",
        category="EC3",
        subcategory="profiles",
        description=(
            "Look up Rectangular Hollow Section (RHS) profile properties. "
            "125 profiles available. Provide designation like 'RHS200x100x5'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "profile_name": {
                    "type": "string",
                    "description": "RHS designation, e.g. 'RHS200x100x5'.",
                },
            },
            "required": ["profile_name"],
        },
        keywords=[
            "RHS", "rectangular hollow", "box section", "hollow section", "profile",
        ],
        handler_module="backend.eurocodepy.adapters",
        handler_function="lookup_rhs_profile",
        clause_references=[],
    ),
    EngToolEntry(
        name="ec3_profile_shs_lookup",
        category="EC3",
        subcategory="profiles",
        description=(
            "Look up Square Hollow Section (SHS) profile properties. "
            "123 profiles available. Provide designation like 'SHS100x100x5'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "profile_name": {
                    "type": "string",
                    "description": "SHS designation, e.g. 'SHS100x100x5'.",
                },
            },
            "required": ["profile_name"],
        },
        keywords=[
            "SHS", "square hollow", "box section", "hollow section", "profile",
        ],
        handler_module="backend.eurocodepy.adapters",
        handler_function="lookup_shs_profile",
        clause_references=[],
    ),
    EngToolEntry(
        name="ec3_steel_grade_lookup",
        category="EC3",
        subcategory="materials",
        description=(
            "Look up structural steel grade properties. Returns fy, fu, E, "
            "gamma_M0/M1/M2. Grades: S235, S275, S355, S420, S460."
        ),
        parameters={
            "type": "object",
            "properties": {
                "grade": {
                    "type": "string",
                    "description": "Steel grade, e.g. 'S355', 'S235', 'S275'.",
                },
            },
            "required": ["grade"],
        },
        keywords=[
            "steel grade", "S235", "S275", "S355", "S420", "S460",
            "yield strength", "fy", "fu", "material", "EN 10025",
        ],
        handler_module="backend.eurocodepy.adapters",
        handler_function="lookup_steel_grade",
        clause_references=["EN 1993-1-1 Table 3.1"],
    ),
    EngToolEntry(
        name="ec3_bolt_lookup",
        category="EC3",
        subcategory="connections",
        description=(
            "Look up bolt properties: area, thread area, fub, fyb. "
            "Diameters: M12-M36. Grades: 4.6, 4.8, 5.6, 5.8, 6.8, 8.8, 10.9."
        ),
        parameters={
            "type": "object",
            "properties": {
                "diameter": {
                    "type": "string",
                    "description": "Bolt diameter, e.g. 'M20', 'M24'.",
                },
                "grade": {
                    "type": "string",
                    "description": "Bolt grade, e.g. '8.8', '10.9'.",
                },
            },
            "required": ["diameter", "grade"],
        },
        keywords=[
            "bolt", "M20", "M24", "fub", "fyb", "thread area", "connection",
            "8.8", "10.9", "fastener",
        ],
        handler_module="backend.eurocodepy.adapters",
        handler_function="lookup_bolt",
        clause_references=["EN 1993-1-8 Table 3.1"],
    ),
]

# Quick index for dispatcher
TOOL_INDEX: dict[str, EngToolEntry] = {e.name: e for e in ENGINEERING_TOOL_REGISTRY}
