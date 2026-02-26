from __future__ import annotations

from pydantic import BaseModel, Field

from tools.mcp.cli import run_cli

TOOL_NAME = "unit_converter"

CONVERSION_TABLE: dict[str, dict[str, float]] = {
    "mm":   {"mm": 1, "cm": 0.1, "m": 0.001, "in": 1/25.4, "ft": 1/304.8},
    "cm":   {"mm": 10, "cm": 1, "m": 0.01, "in": 1/2.54, "ft": 1/30.48},
    "m":    {"mm": 1000, "cm": 100, "m": 1, "in": 39.3701, "ft": 3.28084},
    "in":   {"mm": 25.4, "cm": 2.54, "m": 0.0254, "in": 1, "ft": 1/12},
    "ft":   {"mm": 304.8, "cm": 30.48, "m": 0.3048, "in": 12, "ft": 1},

    "N":    {"N": 1, "kN": 0.001, "MN": 1e-6, "kgf": 1/9.80665, "lbf": 0.224809},
    "kN":   {"N": 1000, "kN": 1, "MN": 0.001, "kgf": 1000/9.80665, "lbf": 224.809},
    "MN":   {"N": 1e6, "kN": 1000, "MN": 1, "kgf": 1e6/9.80665, "lbf": 224809},
    "kgf":  {"N": 9.80665, "kN": 0.00980665, "MN": 9.80665e-6, "kgf": 1, "lbf": 2.20462},
    "lbf":  {"N": 4.44822, "kN": 0.00444822, "MN": 4.44822e-6, "kgf": 0.453592, "lbf": 1},

    "Nm":   {"Nm": 1, "kNm": 0.001, "kgfm": 1/9.80665, "lbft": 0.737562},
    "kNm":  {"Nm": 1000, "kNm": 1, "kgfm": 1000/9.80665, "lbft": 737.562},
    "kgfm": {"Nm": 9.80665, "kNm": 0.00980665, "kgfm": 1, "lbft": 7.23301},
    "lbft": {"Nm": 1.35582, "kNm": 0.00135582, "kgfm": 0.138255, "lbft": 1},

    "Pa":   {"Pa": 1, "kPa": 0.001, "MPa": 1e-6, "GPa": 1e-9, "psi": 0.000145038, "N/mm2": 1e-6},
    "kPa":  {"Pa": 1000, "kPa": 1, "MPa": 0.001, "GPa": 1e-6, "psi": 0.145038, "N/mm2": 0.001},
    "MPa":  {"Pa": 1e6, "kPa": 1000, "MPa": 1, "GPa": 0.001, "psi": 145.038, "N/mm2": 1},
    "GPa":  {"Pa": 1e9, "kPa": 1e6, "MPa": 1000, "GPa": 1, "psi": 145038, "N/mm2": 1000},
    "psi":  {"Pa": 6894.76, "kPa": 6.89476, "MPa": 0.00689476, "GPa": 6.89476e-6, "psi": 1, "N/mm2": 0.00689476},
    "N/mm2": {"Pa": 1e6, "kPa": 1000, "MPa": 1, "GPa": 0.001, "psi": 145.038, "N/mm2": 1},

    "mm2":  {"mm2": 1, "cm2": 0.01, "m2": 1e-6, "in2": 1/645.16},
    "cm2":  {"mm2": 100, "cm2": 1, "m2": 1e-4, "in2": 100/645.16},
    "m2":   {"mm2": 1e6, "cm2": 1e4, "m2": 1, "in2": 1550.0031},
    "in2":  {"mm2": 645.16, "cm2": 6.4516, "m2": 6.4516e-4, "in2": 1},

    "mm4":  {"mm4": 1, "cm4": 1e-4, "m4": 1e-12},
    "cm4":  {"mm4": 1e4, "cm4": 1, "m4": 1e-8},
    "m4":   {"mm4": 1e12, "cm4": 1e8, "m4": 1},

    "mm3":  {"mm3": 1, "cm3": 0.001, "m3": 1e-9},
    "cm3":  {"mm3": 1000, "cm3": 1, "m3": 1e-6},
    "m3":   {"mm3": 1e9, "cm3": 1e6, "m3": 1},
}

ALL_UNITS = sorted(set(CONVERSION_TABLE.keys()))


class UnitConverterInput(BaseModel):
    value: float = Field(description="Numeric value to convert")
    from_unit: str = Field(description="Source unit (e.g. mm, kN, MPa, cm4)")
    to_unit: str = Field(description="Target unit (e.g. m, N, GPa, mm4)")


def convert(inp: UnitConverterInput) -> dict:
    from_u = inp.from_unit.strip()
    to_u = inp.to_unit.strip()

    if from_u not in CONVERSION_TABLE:
        raise ValueError(f"Unknown source unit '{from_u}'. Available: {ALL_UNITS}")

    conversions = CONVERSION_TABLE[from_u]
    if to_u not in conversions:
        compatible = list(conversions.keys())
        raise ValueError(f"Cannot convert '{from_u}' to '{to_u}'. Compatible units for {from_u}: {compatible}")

    factor = conversions[to_u]
    result = inp.value * factor

    return {
        "inputs_used": {"value": inp.value, "from_unit": from_u, "to_unit": to_u},
        "outputs": {
            "result": result,
            "formatted": f"{result:g} {to_u}",
            "factor": factor,
        },
        "clause_references": [],
        "notes": [f"{inp.value:g} {from_u} = {result:g} {to_u} (factor: {factor:g})"],
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=UnitConverterInput, handler=convert)
