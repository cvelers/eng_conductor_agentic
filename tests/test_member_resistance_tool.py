import pytest
from pydantic import ValidationError

from tools.mcp.member_resistance import MemberResistanceInput, compute_resistance


def test_member_resistance_happy_path() -> None:
    payload = MemberResistanceInput.model_validate(
        {
            "section_name": "IPE300",
            "steel_grade": "S355",
            "section_class": 2,
            "gamma_M0": 1.0,
        }
    )

    result = compute_resistance(payload)
    outputs = result["outputs"]

    assert outputs["M_Rd_kNm"] > 0
    assert outputs["N_Rd_kN"] > 0
    assert outputs["V_Rd_kN"] > 0


def test_member_resistance_fails_without_section_properties() -> None:
    with pytest.raises(ValidationError):
        MemberResistanceInput.model_validate(
            {
                "steel_grade": "S355",
                "section_class": 2,
                "gamma_M0": 1.0,
            }
        )
