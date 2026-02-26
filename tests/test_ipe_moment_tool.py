import pytest

from tools.mcp.ipe_moment_resistance import (
    IPEMomentResistanceInput,
    compute_ipe_moment_resistance,
)


def test_ipe_moment_resistance_happy_path() -> None:
    payload = IPEMomentResistanceInput.model_validate(
        {
            "section_name": "IPE300",
            "steel_grade": "S355",
            "section_class": 2,
            "gamma_M0": 1.0,
        }
    )

    result = compute_ipe_moment_resistance(payload)
    assert result["outputs"]["M_Rd_kNm"] > 0
    assert "library_source" in result["section_properties"]


def test_ipe_moment_resistance_missing_section_fails() -> None:
    payload = IPEMomentResistanceInput.model_validate(
        {
            "section_name": "IPE999",
            "steel_grade": "S355",
        }
    )

    with pytest.raises(ValueError):
        compute_ipe_moment_resistance(payload)
