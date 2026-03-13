from tools.mcp.math_calculator import Equation, MathCalculatorInput, calculate


def test_math_calculator_emits_latex_for_each_step() -> None:
    payload = MathCalculatorInput(
        variables={
            "pi": 3.141592653589793,
            "E": 210000,
            "Iz": 6038000,
            "L": 5000,
            "Iw": 124260000000,
            "G": 81000,
            "It": 197500,
            "Wpl_y": 628400,
            "fy": 355,
        },
        equations=[
            Equation(
                name="M_cr_Nmm",
                expression="(pi**2 * E * Iz / L**2) * sqrt(Iw / Iz + (L**2 * G * It) / (pi**2 * E * Iz))",
                unit="Nmm",
            ),
            Equation(
                name="lambda_LT_bar",
                expression="sqrt((Wpl_y * fy) / M_cr_Nmm)",
            ),
        ],
    )

    result = calculate(payload)
    steps = result["intermediate"]["steps"]

    assert len(steps) == 2
    assert "\\pi" in steps[0]["latex"]
    assert "\\sqrt{" in steps[0]["latex"]
    assert "\\mathrm{Nmm}" in steps[0]["latex"]
    assert "\\bar{\\lambda}" in steps[1]["latex"]
    assert steps[1]["latex"].count("=") == 2


def test_math_calculator_formats_ec3_identifiers_cleanly() -> None:
    payload = MathCalculatorInput(
        variables={
            "W_pl_y_cm3": 628.4,
            "fy_MPa": 355,
            "gamma_M0": 1.0,
        },
        equations=[
            Equation(
                name="M_c_Rd_kNm",
                expression="(W_pl_y_cm3 * 1000 * fy_MPa) / (gamma_M0 * 1000000)",
                unit="kNm",
            ),
        ],
    )

    result = calculate(payload)
    latex = result["intermediate"]["steps"][0]["latex"]

    assert latex == (
        r"M_{c,Rd} = \frac{W_{pl,y} \cdot 1000 \cdot f_y}{\gamma_{M0} \cdot 1000000} = 223.082\,\mathrm{kNm}"
    )
