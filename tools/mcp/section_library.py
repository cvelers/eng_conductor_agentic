from __future__ import annotations

from tools.mcp.ipe_section_library import load_ipe_sections

# Minimal built-in sample properties for demo/testing.
SECTION_LIBRARY = {
    "IPE300": {
        "h_mm": 300.0,
        "b_mm": 150.0,
        "tw_mm": 7.1,
        "tf_mm": 10.7,
        "r_mm": 15.0,
        "area_cm2": 53.8,
        "I_y_cm4": 8356.0,
        "I_z_cm4": 604.0,
        "wpl_y_cm3": 628.0,
        "wel_y_cm3": 557.0,
        "av_z_cm2": 27.5,
    },
    "HEA200": {
        "h_mm": 190.0,
        "b_mm": 200.0,
        "tw_mm": 6.5,
        "tf_mm": 10.0,
        "r_mm": 18.0,
        "area_cm2": 53.8,
        "I_y_cm4": 3692.0,
        "I_z_cm4": 1336.0,
        "wpl_y_cm3": 429.5,
        "wel_y_cm3": 388.6,
        "av_z_cm2": 31.0,
    },
}

_ipe_sections, _ipe_source = load_ipe_sections()
for _name, _row in _ipe_sections.items():
    _merged = dict(_row)
    _merged.setdefault("av_z_cm2", round(float(_merged.get("area_cm2", 0.0)) * 0.5, 3))
    SECTION_LIBRARY.setdefault(_name, _merged)


def steel_grade_to_fy(steel_grade: str, thickness_mm: float | None = None) -> float:
    """Return fy in MPa, accounting for thickness-dependent reduction per EC3 Table 3.1."""
    from tools.mcp.steel_grade_properties import EC3_TABLE_3_1

    grade = steel_grade.strip().upper()
    if grade in EC3_TABLE_3_1:
        row = EC3_TABLE_3_1[grade]
        t = thickness_mm or 16.0  # default assumes t ≤ 40 mm
        if t <= 40.0:
            return float(row["fy_t_le_40"])
        else:
            return float(row["fy_40_lt_t_le_80"])
    # Fallback for non-standard grades
    if grade.startswith("S") and grade[1:].isdigit():
        return float(int(grade[1:]))
    raise ValueError(f"Unsupported steel grade '{steel_grade}'. Use formats like S355.")
