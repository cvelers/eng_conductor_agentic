"""Look up geometric properties for standard rolled steel profiles.

Supports IPE, HEA, HEB, and HEM families. Returns area, moments of
inertia, section moduli, and other geometric properties needed for
Eurocode calculations.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from tools.mcp.cli import run_cli
from tools.mcp.section_library import SECTION_LIBRARY

TOOL_NAME = "section_properties"

# ── HEM sections (not in the base library) ─────────────────────────

_HEM_SECTIONS: dict[str, dict[str, float]] = {
    "HEM100": {
        "h_mm": 120.0, "b_mm": 106.0, "tw_mm": 12.0, "tf_mm": 20.0, "r_mm": 12.0,
        "area_cm2": 53.2, "I_y_cm4": 1143.0, "I_z_cm4": 399.0,
        "wel_y_cm3": 190.4, "wpl_y_cm3": 235.8,
    },
    "HEM120": {
        "h_mm": 140.0, "b_mm": 126.0, "tw_mm": 12.5, "tf_mm": 21.0, "r_mm": 12.0,
        "area_cm2": 66.4, "I_y_cm4": 2018.0, "I_z_cm4": 703.0,
        "wel_y_cm3": 288.3, "wpl_y_cm3": 350.6,
    },
    "HEM140": {
        "h_mm": 160.0, "b_mm": 146.0, "tw_mm": 13.0, "tf_mm": 22.0, "r_mm": 12.0,
        "area_cm2": 80.6, "I_y_cm4": 3291.0, "I_z_cm4": 1144.0,
        "wel_y_cm3": 411.4, "wpl_y_cm3": 496.3,
    },
    "HEM160": {
        "h_mm": 180.0, "b_mm": 166.0, "tw_mm": 14.0, "tf_mm": 23.0, "r_mm": 15.0,
        "area_cm2": 97.1, "I_y_cm4": 5098.0, "I_z_cm4": 1759.0,
        "wel_y_cm3": 566.4, "wpl_y_cm3": 674.6,
    },
    "HEM180": {
        "h_mm": 200.0, "b_mm": 186.0, "tw_mm": 14.5, "tf_mm": 24.0, "r_mm": 15.0,
        "area_cm2": 113.3, "I_y_cm4": 7483.0, "I_z_cm4": 2580.0,
        "wel_y_cm3": 748.3, "wpl_y_cm3": 883.3,
    },
    "HEM200": {
        "h_mm": 220.0, "b_mm": 206.0, "tw_mm": 15.0, "tf_mm": 25.0, "r_mm": 18.0,
        "area_cm2": 131.3, "I_y_cm4": 10640.0, "I_z_cm4": 3651.0,
        "wel_y_cm3": 967.3, "wpl_y_cm3": 1135.0,
    },
    "HEM220": {
        "h_mm": 240.0, "b_mm": 226.0, "tw_mm": 15.5, "tf_mm": 26.0, "r_mm": 18.0,
        "area_cm2": 149.4, "I_y_cm4": 14600.0, "I_z_cm4": 5012.0,
        "wel_y_cm3": 1217.0, "wpl_y_cm3": 1419.0,
    },
    "HEM240": {
        "h_mm": 270.0, "b_mm": 248.0, "tw_mm": 18.0, "tf_mm": 32.0, "r_mm": 21.0,
        "area_cm2": 199.6, "I_y_cm4": 24290.0, "I_z_cm4": 8153.0,
        "wel_y_cm3": 1799.0, "wpl_y_cm3": 2117.0,
    },
    "HEM260": {
        "h_mm": 290.0, "b_mm": 268.0, "tw_mm": 18.0, "tf_mm": 32.5, "r_mm": 24.0,
        "area_cm2": 220.0, "I_y_cm4": 31310.0, "I_z_cm4": 10450.0,
        "wel_y_cm3": 2159.0, "wpl_y_cm3": 2524.0,
    },
    "HEM280": {
        "h_mm": 310.0, "b_mm": 288.0, "tw_mm": 18.5, "tf_mm": 33.0, "r_mm": 24.0,
        "area_cm2": 240.2, "I_y_cm4": 39550.0, "I_z_cm4": 13160.0,
        "wel_y_cm3": 2551.0, "wpl_y_cm3": 2966.0,
    },
    "HEM300": {
        "h_mm": 340.0, "b_mm": 310.0, "tw_mm": 21.0, "tf_mm": 39.0, "r_mm": 27.0,
        "area_cm2": 303.1, "I_y_cm4": 59200.0, "I_z_cm4": 19400.0,
        "wel_y_cm3": 3482.0, "wpl_y_cm3": 4078.0,
    },
    "HEM320": {
        "h_mm": 359.0, "b_mm": 309.0, "tw_mm": 21.0, "tf_mm": 40.0, "r_mm": 27.0,
        "area_cm2": 312.0, "I_y_cm4": 68130.0, "I_z_cm4": 19710.0,
        "wel_y_cm3": 3796.0, "wpl_y_cm3": 4435.0,
    },
    "HEM340": {
        "h_mm": 377.0, "b_mm": 309.0, "tw_mm": 21.0, "tf_mm": 40.0, "r_mm": 27.0,
        "area_cm2": 316.0, "I_y_cm4": 76370.0, "I_z_cm4": 19710.0,
        "wel_y_cm3": 4052.0, "wpl_y_cm3": 4718.0,
    },
    "HEM360": {
        "h_mm": 395.0, "b_mm": 308.0, "tw_mm": 21.0, "tf_mm": 40.0, "r_mm": 27.0,
        "area_cm2": 318.8, "I_y_cm4": 84870.0, "I_z_cm4": 19520.0,
        "wel_y_cm3": 4296.0, "wpl_y_cm3": 4989.0,
    },
    "HEM400": {
        "h_mm": 432.0, "b_mm": 307.0, "tw_mm": 21.0, "tf_mm": 40.0, "r_mm": 27.0,
        "area_cm2": 325.8, "I_y_cm4": 104100.0, "I_z_cm4": 19340.0,
        "wel_y_cm3": 4820.0, "wpl_y_cm3": 5571.0,
    },
    "HEM450": {
        "h_mm": 478.0, "b_mm": 307.0, "tw_mm": 21.0, "tf_mm": 40.0, "r_mm": 27.0,
        "area_cm2": 335.4, "I_y_cm4": 131500.0, "I_z_cm4": 19340.0,
        "wel_y_cm3": 5501.0, "wpl_y_cm3": 6331.0,
    },
    "HEM500": {
        "h_mm": 524.0, "b_mm": 306.0, "tw_mm": 21.0, "tf_mm": 40.0, "r_mm": 27.0,
        "area_cm2": 344.2, "I_y_cm4": 161900.0, "I_z_cm4": 19150.0,
        "wel_y_cm3": 6180.0, "wpl_y_cm3": 7094.0,
    },
    "HEM550": {
        "h_mm": 572.0, "b_mm": 306.0, "tw_mm": 21.0, "tf_mm": 40.0, "r_mm": 27.0,
        "area_cm2": 354.2, "I_y_cm4": 198000.0, "I_z_cm4": 19160.0,
        "wel_y_cm3": 6923.0, "wpl_y_cm3": 7933.0,
    },
    "HEM600": {
        "h_mm": 620.0, "b_mm": 305.0, "tw_mm": 21.0, "tf_mm": 40.0, "r_mm": 27.0,
        "area_cm2": 363.7, "I_y_cm4": 237400.0, "I_z_cm4": 18980.0,
        "wel_y_cm3": 7660.0, "wpl_y_cm3": 8772.0,
    },
}

# HEA sections not in base library
_HEA_SECTIONS: dict[str, dict[str, float]] = {
    "HEA100": {
        "h_mm": 96.0, "b_mm": 100.0, "tw_mm": 5.0, "tf_mm": 8.0, "r_mm": 12.0,
        "area_cm2": 21.2, "I_y_cm4": 349.0, "I_z_cm4": 134.0,
        "wel_y_cm3": 72.8, "wpl_y_cm3": 83.0,
    },
    "HEA120": {
        "h_mm": 114.0, "b_mm": 120.0, "tw_mm": 5.0, "tf_mm": 8.0, "r_mm": 12.0,
        "area_cm2": 25.3, "I_y_cm4": 606.0, "I_z_cm4": 231.0,
        "wel_y_cm3": 106.3, "wpl_y_cm3": 119.5,
    },
    "HEA140": {
        "h_mm": 133.0, "b_mm": 140.0, "tw_mm": 5.5, "tf_mm": 8.5, "r_mm": 12.0,
        "area_cm2": 31.4, "I_y_cm4": 1033.0, "I_z_cm4": 389.0,
        "wel_y_cm3": 155.4, "wpl_y_cm3": 173.5,
    },
    "HEA160": {
        "h_mm": 152.0, "b_mm": 160.0, "tw_mm": 6.0, "tf_mm": 9.0, "r_mm": 15.0,
        "area_cm2": 38.8, "I_y_cm4": 1673.0, "I_z_cm4": 616.0,
        "wel_y_cm3": 220.1, "wpl_y_cm3": 245.1,
    },
    "HEA180": {
        "h_mm": 171.0, "b_mm": 180.0, "tw_mm": 6.0, "tf_mm": 9.5, "r_mm": 15.0,
        "area_cm2": 45.3, "I_y_cm4": 2510.0, "I_z_cm4": 925.0,
        "wel_y_cm3": 293.6, "wpl_y_cm3": 324.8,
    },
    "HEA260": {
        "h_mm": 250.0, "b_mm": 260.0, "tw_mm": 7.5, "tf_mm": 12.5, "r_mm": 24.0,
        "area_cm2": 86.8, "I_y_cm4": 10450.0, "I_z_cm4": 3668.0,
        "wel_y_cm3": 836.4, "wpl_y_cm3": 919.8,
    },
    "HEA280": {
        "h_mm": 270.0, "b_mm": 280.0, "tw_mm": 8.0, "tf_mm": 13.0, "r_mm": 24.0,
        "area_cm2": 97.3, "I_y_cm4": 13670.0, "I_z_cm4": 4763.0,
        "wel_y_cm3": 1013.0, "wpl_y_cm3": 1112.0,
    },
    "HEA300": {
        "h_mm": 290.0, "b_mm": 300.0, "tw_mm": 8.5, "tf_mm": 14.0, "r_mm": 27.0,
        "area_cm2": 112.5, "I_y_cm4": 18260.0, "I_z_cm4": 6310.0,
        "wel_y_cm3": 1260.0, "wpl_y_cm3": 1383.0,
    },
    "HEA320": {
        "h_mm": 310.0, "b_mm": 300.0, "tw_mm": 9.0, "tf_mm": 15.5, "r_mm": 27.0,
        "area_cm2": 124.4, "I_y_cm4": 22930.0, "I_z_cm4": 6985.0,
        "wel_y_cm3": 1479.0, "wpl_y_cm3": 1628.0,
    },
    "HEA340": {
        "h_mm": 330.0, "b_mm": 300.0, "tw_mm": 9.5, "tf_mm": 16.5, "r_mm": 27.0,
        "area_cm2": 133.5, "I_y_cm4": 27690.0, "I_z_cm4": 7436.0,
        "wel_y_cm3": 1678.0, "wpl_y_cm3": 1850.0,
    },
    "HEA360": {
        "h_mm": 350.0, "b_mm": 300.0, "tw_mm": 10.0, "tf_mm": 17.5, "r_mm": 27.0,
        "area_cm2": 142.8, "I_y_cm4": 33090.0, "I_z_cm4": 7887.0,
        "wel_y_cm3": 1891.0, "wpl_y_cm3": 2088.0,
    },
    "HEA400": {
        "h_mm": 390.0, "b_mm": 300.0, "tw_mm": 11.0, "tf_mm": 19.0, "r_mm": 27.0,
        "area_cm2": 159.0, "I_y_cm4": 45070.0, "I_z_cm4": 8564.0,
        "wel_y_cm3": 2311.0, "wpl_y_cm3": 2562.0,
    },
    "HEA450": {
        "h_mm": 440.0, "b_mm": 300.0, "tw_mm": 11.5, "tf_mm": 21.0, "r_mm": 27.0,
        "area_cm2": 178.0, "I_y_cm4": 63720.0, "I_z_cm4": 9465.0,
        "wel_y_cm3": 2896.0, "wpl_y_cm3": 3216.0,
    },
    "HEA500": {
        "h_mm": 490.0, "b_mm": 300.0, "tw_mm": 12.0, "tf_mm": 23.0, "r_mm": 27.0,
        "area_cm2": 197.5, "I_y_cm4": 86970.0, "I_z_cm4": 10370.0,
        "wel_y_cm3": 3550.0, "wpl_y_cm3": 3949.0,
    },
    "HEA550": {
        "h_mm": 540.0, "b_mm": 300.0, "tw_mm": 12.5, "tf_mm": 24.0, "r_mm": 27.0,
        "area_cm2": 211.8, "I_y_cm4": 111900.0, "I_z_cm4": 10820.0,
        "wel_y_cm3": 4146.0, "wpl_y_cm3": 4622.0,
    },
    "HEA600": {
        "h_mm": 590.0, "b_mm": 300.0, "tw_mm": 13.0, "tf_mm": 25.0, "r_mm": 27.0,
        "area_cm2": 226.5, "I_y_cm4": 141200.0, "I_z_cm4": 11270.0,
        "wel_y_cm3": 4787.0, "wpl_y_cm3": 5350.0,
    },
}

# HEB sections not in base library
_HEB_SECTIONS: dict[str, dict[str, float]] = {
    "HEB100": {
        "h_mm": 100.0, "b_mm": 100.0, "tw_mm": 6.0, "tf_mm": 10.0, "r_mm": 12.0,
        "area_cm2": 26.0, "I_y_cm4": 450.0, "I_z_cm4": 167.0,
        "wel_y_cm3": 89.9, "wpl_y_cm3": 104.2,
    },
    "HEB120": {
        "h_mm": 120.0, "b_mm": 120.0, "tw_mm": 6.5, "tf_mm": 11.0, "r_mm": 12.0,
        "area_cm2": 34.0, "I_y_cm4": 864.0, "I_z_cm4": 318.0,
        "wel_y_cm3": 144.1, "wpl_y_cm3": 165.2,
    },
    "HEB140": {
        "h_mm": 140.0, "b_mm": 140.0, "tw_mm": 7.0, "tf_mm": 12.0, "r_mm": 12.0,
        "area_cm2": 43.0, "I_y_cm4": 1509.0, "I_z_cm4": 550.0,
        "wel_y_cm3": 215.6, "wpl_y_cm3": 245.4,
    },
    "HEB160": {
        "h_mm": 160.0, "b_mm": 160.0, "tw_mm": 8.0, "tf_mm": 13.0, "r_mm": 15.0,
        "area_cm2": 54.3, "I_y_cm4": 2492.0, "I_z_cm4": 889.0,
        "wel_y_cm3": 311.5, "wpl_y_cm3": 354.0,
    },
    "HEB180": {
        "h_mm": 180.0, "b_mm": 180.0, "tw_mm": 8.5, "tf_mm": 14.0, "r_mm": 15.0,
        "area_cm2": 65.3, "I_y_cm4": 3831.0, "I_z_cm4": 1363.0,
        "wel_y_cm3": 426.0, "wpl_y_cm3": 481.4,
    },
    "HEB220": {
        "h_mm": 220.0, "b_mm": 220.0, "tw_mm": 9.5, "tf_mm": 16.0, "r_mm": 18.0,
        "area_cm2": 91.0, "I_y_cm4": 8091.0, "I_z_cm4": 2843.0,
        "wel_y_cm3": 735.5, "wpl_y_cm3": 827.0,
    },
    "HEB240": {
        "h_mm": 240.0, "b_mm": 240.0, "tw_mm": 10.0, "tf_mm": 17.0, "r_mm": 21.0,
        "area_cm2": 106.0, "I_y_cm4": 11260.0, "I_z_cm4": 3923.0,
        "wel_y_cm3": 938.3, "wpl_y_cm3": 1053.0,
    },
    "HEB260": {
        "h_mm": 260.0, "b_mm": 260.0, "tw_mm": 10.0, "tf_mm": 17.5, "r_mm": 24.0,
        "area_cm2": 118.4, "I_y_cm4": 14920.0, "I_z_cm4": 5135.0,
        "wel_y_cm3": 1148.0, "wpl_y_cm3": 1283.0,
    },
    "HEB280": {
        "h_mm": 280.0, "b_mm": 280.0, "tw_mm": 10.5, "tf_mm": 18.0, "r_mm": 24.0,
        "area_cm2": 131.4, "I_y_cm4": 19270.0, "I_z_cm4": 6595.0,
        "wel_y_cm3": 1376.0, "wpl_y_cm3": 1534.0,
    },
    "HEB320": {
        "h_mm": 320.0, "b_mm": 300.0, "tw_mm": 11.5, "tf_mm": 20.5, "r_mm": 27.0,
        "area_cm2": 161.3, "I_y_cm4": 30820.0, "I_z_cm4": 9239.0,
        "wel_y_cm3": 1926.0, "wpl_y_cm3": 2149.0,
    },
    "HEB340": {
        "h_mm": 340.0, "b_mm": 300.0, "tw_mm": 12.0, "tf_mm": 21.5, "r_mm": 27.0,
        "area_cm2": 170.9, "I_y_cm4": 36660.0, "I_z_cm4": 9690.0,
        "wel_y_cm3": 2156.0, "wpl_y_cm3": 2408.0,
    },
    "HEB360": {
        "h_mm": 360.0, "b_mm": 300.0, "tw_mm": 12.5, "tf_mm": 22.5, "r_mm": 27.0,
        "area_cm2": 180.6, "I_y_cm4": 43190.0, "I_z_cm4": 10140.0,
        "wel_y_cm3": 2400.0, "wpl_y_cm3": 2683.0,
    },
    "HEB400": {
        "h_mm": 400.0, "b_mm": 300.0, "tw_mm": 13.5, "tf_mm": 24.0, "r_mm": 27.0,
        "area_cm2": 197.8, "I_y_cm4": 57680.0, "I_z_cm4": 10820.0,
        "wel_y_cm3": 2884.0, "wpl_y_cm3": 3232.0,
    },
    "HEB450": {
        "h_mm": 450.0, "b_mm": 300.0, "tw_mm": 14.0, "tf_mm": 26.0, "r_mm": 27.0,
        "area_cm2": 218.0, "I_y_cm4": 79890.0, "I_z_cm4": 11720.0,
        "wel_y_cm3": 3551.0, "wpl_y_cm3": 3982.0,
    },
    "HEB500": {
        "h_mm": 500.0, "b_mm": 300.0, "tw_mm": 14.5, "tf_mm": 28.0, "r_mm": 27.0,
        "area_cm2": 238.6, "I_y_cm4": 107200.0, "I_z_cm4": 12620.0,
        "wel_y_cm3": 4287.0, "wpl_y_cm3": 4815.0,
    },
    "HEB550": {
        "h_mm": 550.0, "b_mm": 300.0, "tw_mm": 15.0, "tf_mm": 29.0, "r_mm": 27.0,
        "area_cm2": 254.1, "I_y_cm4": 136700.0, "I_z_cm4": 13080.0,
        "wel_y_cm3": 4971.0, "wpl_y_cm3": 5591.0,
    },
    "HEB600": {
        "h_mm": 600.0, "b_mm": 300.0, "tw_mm": 15.5, "tf_mm": 30.0, "r_mm": 27.0,
        "area_cm2": 270.0, "I_y_cm4": 171000.0, "I_z_cm4": 13530.0,
        "wel_y_cm3": 5701.0, "wpl_y_cm3": 6425.0,
    },
}

# Merge all into a combined lookup
_ALL_SECTIONS: dict[str, dict[str, float]] = {}
_ALL_SECTIONS.update(SECTION_LIBRARY)
_ALL_SECTIONS.update(_HEM_SECTIONS)
for name, props in _HEA_SECTIONS.items():
    _ALL_SECTIONS.setdefault(name, props)
for name, props in _HEB_SECTIONS.items():
    _ALL_SECTIONS.setdefault(name, props)

# Organized family listing
_FAMILIES: dict[str, list[str]] = {}
for name in sorted(_ALL_SECTIONS.keys()):
    import re as _re
    match = _re.match(r"^([A-Z]+)", name)
    if match:
        family = match.group(1)
        _FAMILIES.setdefault(family, []).append(name)


# ── Pydantic models ──────────────────────────────────────────────────

class SectionPropertiesInput(BaseModel):
    section_name: str = Field(
        description="Standard section designation, e.g. 'IPE300', 'HEA200', 'HEB300', 'HEM200'. "
        "Case-insensitive, spaces are stripped."
    )
    properties: Optional[List[str]] = Field(
        default=None,
        description="Optional list of specific properties to return. "
        "If omitted, all available properties are returned. "
        "Available: h_mm, b_mm, tw_mm, tf_mm, r_mm, area_cm2, "
        "I_y_cm4, I_z_cm4, wel_y_cm3, wpl_y_cm3, av_z_cm2",
    )


def lookup(inp: SectionPropertiesInput) -> dict:
    """Look up geometric properties for a standard rolled profile."""
    key = inp.section_name.upper().replace(" ", "")

    if key not in _ALL_SECTIONS:
        # Find close matches in the same family
        match = _re.match(r"^([A-Z]+)", key)
        family = match.group(1) if match else ""
        available = _FAMILIES.get(family, [])
        if not available:
            available_families = sorted(_FAMILIES.keys())
            raise ValueError(
                f"Unknown section '{inp.section_name}'. "
                f"Available families: {', '.join(available_families)}. "
                f"Example sections: IPE200, IPE300, HEA200, HEB300, HEM200."
            )
        raise ValueError(
            f"Section '{inp.section_name}' not found in {family} family. "
            f"Available {family} sections: {', '.join(available)}"
        )

    row = _ALL_SECTIONS[key]

    # Filter properties if requested
    if inp.properties:
        outputs = {}
        for p in inp.properties:
            if p in row:
                outputs[p] = row[p]
            else:
                outputs[p] = None  # Signal not available
    else:
        outputs = dict(row)

    # Build family list for context
    match = _re.match(r"^([A-Z]+)", key)
    family = match.group(1) if match else "Unknown"

    return {
        "inputs_used": {
            "section_name": key,
            "family": family,
        },
        "outputs": outputs,
        "clause_references": [],
        "notes": [
            f"Section {key} properties from {family} series.",
            f"Available {family} sections: {', '.join(_FAMILIES.get(family, []))}",
        ],
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=SectionPropertiesInput, handler=lookup)
