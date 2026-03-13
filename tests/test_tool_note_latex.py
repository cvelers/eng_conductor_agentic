from tools.mcp.bolt_shear import BoltShearInput, calculate as calc_bolt_shear
from tools.mcp.effective_length import EffectiveLengthInput, calculate as calc_effective_length


def test_bolt_shear_returns_latex_note_objects() -> None:
    result = calc_bolt_shear(
        BoltShearInput(
            bolt_class="8.8",
            bolt_diameter_mm=20,
            n_shear_planes=1,
            n_bolts=4,
        )
    )

    assert isinstance(result["notes"][0], dict)
    assert "F_{v,Rd}" in result["notes"][0]["latex"]


def test_effective_length_returns_latex_formula_note() -> None:
    result = calc_effective_length(
        EffectiveLengthInput(
            support_conditions="fixed-pinned",
            system_length_m=4.5,
        )
    )

    assert isinstance(result["notes"][1], dict)
    assert "L_{cr}" in result["notes"][1]["latex"]
