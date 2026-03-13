"""Safe AST-based math expression evaluator for Eurocode calculations.

Takes a sequential list of named equations (with units) and a variables dict.
Evaluates them in order so later equations can reference earlier results.
Only allows safe math operations — no exec/eval of arbitrary code.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Optional

from pydantic import BaseModel, Field

from tools.mcp.cli import run_cli

TOOL_NAME = "math_calculator"

# ── Safe math environment ────────────────────────────────────────────

_SAFE_FUNCTIONS: dict[str, Any] = {
    "sqrt": math.sqrt,
    "pow": pow,
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "radians": math.radians,
    "degrees": math.degrees,
    "pi": math.pi,
    "e": math.e,
}

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_COMPARE = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


class _SafeEvaluator(ast.NodeVisitor):
    """Walk an AST expression tree and evaluate with restricted operations."""

    def __init__(self, namespace: dict[str, Any]) -> None:
        self.namespace = namespace

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    def visit_Name(self, node: ast.Name) -> Any:
        name = node.id
        if name in self.namespace:
            return self.namespace[name]
        raise ValueError(
            f"Unknown variable '{name}'. "
            f"Available: {sorted(k for k in self.namespace if not k.startswith('_'))}"
        )

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op_func = _SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return op_func(left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        op_func = _SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(self.visit(node.operand))

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed (e.g., sqrt(x)).")
        func_name = node.func.id
        if func_name not in _SAFE_FUNCTIONS:
            raise ValueError(
                f"Function '{func_name}' is not allowed. "
                f"Available: {sorted(k for k, v in _SAFE_FUNCTIONS.items() if callable(v))}"
            )
        func = _SAFE_FUNCTIONS[func_name]
        if not callable(func):
            raise ValueError(f"'{func_name}' is not callable.")
        args = [self.visit(a) for a in node.args]
        return func(*args)

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        test = self.visit(node.test)
        return self.visit(node.body) if test else self.visit(node.orelse)

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            op_func = _SAFE_COMPARE.get(type(op))
            if op_func is None:
                raise ValueError(f"Unsupported comparison: {type(op).__name__}")
            right = self.visit(comparator)
            if not op_func(left, right):
                return False
            left = right
        return True

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        if isinstance(node.op, ast.And):
            result = True
            for val in node.values:
                result = self.visit(val)
                if not result:
                    return result
            return result
        elif isinstance(node.op, ast.Or):
            result = False
            for val in node.values:
                result = self.visit(val)
                if result:
                    return result
            return result
        raise ValueError(f"Unsupported boolean op: {type(node.op).__name__}")

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        # Allow simple dict/list subscript access: namespace["key"]
        value = self.visit(node.value)
        if isinstance(node.slice, ast.Constant):
            key = node.slice.value
        else:
            key = self.visit(node.slice)
        return value[key]

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(
            f"Unsupported expression node: {type(node).__name__}. "
            "Only arithmetic, comparisons, function calls, and conditionals are allowed."
        )


def safe_eval(expression: str, namespace: dict[str, Any]) -> Any:
    """Evaluate a math expression safely using AST parsing.

    Only allows: arithmetic, comparisons, safe math functions, conditionals.
    No imports, no attribute access, no exec/eval.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc

    evaluator = _SafeEvaluator(namespace)
    return evaluator.visit(tree)


_LATEX_GREEK = {
    "alpha": r"\alpha",
    "beta": r"\beta",
    "gamma": r"\gamma",
    "phi": r"\phi",
    "Phi": r"\Phi",
    "chi": r"\chi",
    "lambda": r"\lambda",
    "psi": r"\psi",
    "eta": r"\eta",
    "mu": r"\mu",
    "rho": r"\rho",
    "epsilon": r"\epsilon",
}

_LATEX_IDENT_OVERRIDES = {
    "Mcr": r"M_{cr}",
    "Mpl": r"M_{pl}",
    "Mel": r"M_{el}",
    "Mb": r"M_b",
    "Mc": r"M_c",
    "Ncr": r"N_{cr}",
    "Npl": r"N_{pl}",
    "Vpl": r"V_{pl}",
    "Wpl": r"W_{pl}",
    "Wel": r"W_{el}",
    "fy": r"f_y",
    "fu": r"f_u",
}

_LATEX_NAME_OVERRIDES = {
    "M_c_Rd": r"M_{c,Rd}",
    "Mc_Rd": r"M_{c,Rd}",
    "M_b_Rd": r"M_{b,Rd}",
    "Mb_Rd": r"M_{b,Rd}",
    "M_pl_Rd": r"M_{pl,Rd}",
    "Mpl_Rd": r"M_{pl,Rd}",
    "M_el_Rd": r"M_{el,Rd}",
    "Mel_Rd": r"M_{el,Rd}",
    "M_Rd": r"M_{Rd}",
    "M_Ed": r"M_{Ed}",
    "M_cr": r"M_{cr}",
    "N_pl_Rd": r"N_{pl,Rd}",
    "Npl_Rd": r"N_{pl,Rd}",
    "N_b_Rd": r"N_{b,Rd}",
    "N_t_Rd": r"N_{t,Rd}",
    "N_Rd": r"N_{Rd}",
    "N_Ed": r"N_{Ed}",
    "N_cr": r"N_{cr}",
    "V_pl_Rd": r"V_{pl,Rd}",
    "Vpl_Rd": r"V_{pl,Rd}",
    "V_Rd": r"V_{Rd}",
    "V_Ed": r"V_{Ed}",
    "W_pl_y": r"W_{pl,y}",
    "Wpl_y": r"W_{pl,y}",
    "W_el_y": r"W_{el,y}",
    "Wel_y": r"W_{el,y}",
    "W_pl_z": r"W_{pl,z}",
    "Wpl_z": r"W_{pl,z}",
    "W_el_z": r"W_{el,z}",
    "Wel_z": r"W_{el,z}",
    "I_y": r"I_y",
    "I_z": r"I_z",
    "I_w": r"I_w",
    "I_t": r"I_t",
    "f_y": r"f_y",
    "f_u": r"f_u",
    "gamma_M0": r"\gamma_{M0}",
    "gamma_M1": r"\gamma_{M1}",
    "gamma_M2": r"\gamma_{M2}",
    "alpha_LT": r"\alpha_{LT}",
    "phi_LT": r"\phi_{LT}",
    "Phi_LT": r"\Phi_{LT}",
    "chi_LT": r"\chi_{LT}",
    "lambda_LT": r"\lambda_{LT}",
    "lambda_LT_bar": r"\bar{\lambda}_{LT}",
}

_LATEX_UNIT_OVERRIDES = {
    "nmm": r"\mathrm{Nmm}",
    "knm": r"\mathrm{kNm}",
    "kn": r"\mathrm{kN}",
    "mpa": r"\mathrm{MPa}",
    "gpa": r"\mathrm{GPa}",
    "mm": r"\mathrm{mm}",
    "mm2": r"\mathrm{mm}^{2}",
    "mm3": r"\mathrm{mm}^{3}",
    "mm4": r"\mathrm{mm}^{4}",
    "mm6": r"\mathrm{mm}^{6}",
    "cm2": r"\mathrm{cm}^{2}",
    "cm3": r"\mathrm{cm}^{3}",
    "cm4": r"\mathrm{cm}^{4}",
    "cm6": r"\mathrm{cm}^{6}",
    "m": r"\mathrm{m}",
    "m2": r"\mathrm{m}^{2}",
    "rad": r"\mathrm{rad}",
    "deg": r"^\circ",
}


def _normalize_unit_token(token: str) -> str:
    return (
        str(token or "")
        .replace(" ", "")
        .replace("^", "")
        .replace("\u00b2", "2")
        .replace("\u00b3", "3")
        .replace("\u2074", "4")
        .replace("\u2076", "6")
        .lower()
    )


def _unit_to_latex(unit: str) -> str:
    raw = str(unit or "").strip()
    if not raw:
        return ""
    return _LATEX_UNIT_OVERRIDES.get(_normalize_unit_token(raw), rf"\mathrm{{{raw}}}")


def _identifier_token_to_latex(token: str) -> str:
    if token in _LATEX_IDENT_OVERRIDES:
        return _LATEX_IDENT_OVERRIDES[token]
    if token in _LATEX_GREEK:
        return _LATEX_GREEK[token]
    if token.isalpha() and len(token) == 1:
        return token
    if token.isupper() or token.islower():
        return rf"\mathrm{{{token}}}"
    return rf"\mathrm{{{token}}}"


def _identifier_to_latex(name: str, unit: str | None = None) -> str:
    parts = [p for p in str(name or "").split("_") if p]
    if not parts:
        return r"\mathrm{?}"

    normalized_unit = _normalize_unit_token(unit or "")
    trailing_unit = _normalize_unit_token(parts[-1]) if parts else ""
    if normalized_unit and parts and trailing_unit == normalized_unit:
        parts = parts[:-1]
    elif trailing_unit in _LATEX_UNIT_OVERRIDES:
        parts = parts[:-1]
    if not parts:
        return r"\mathrm{?}"

    joined_name = "_".join(parts)
    if joined_name in _LATEX_NAME_OVERRIDES:
        return _LATEX_NAME_OVERRIDES[joined_name]

    add_bar = parts[-1].lower() == "bar"
    if add_bar:
        parts = parts[:-1]
    if not parts:
        return r"\mathrm{?}"

    base = _identifier_token_to_latex(parts[0])
    if add_bar:
        base = rf"\bar{{{base}}}"

    if len(parts) == 1:
        return base

    subs = ",".join(_identifier_token_to_latex(part) for part in parts[1:])
    return rf"{base}_{{{subs}}}"


def _format_scalar_for_latex(value: Any) -> str:
    if isinstance(value, bool):
        return r"\mathrm{true}" if value else r"\mathrm{false}"
    return str(value)


def _expr_to_latex(node: ast.AST, parent_prec: int = 0) -> str:
    if isinstance(node, ast.Expression):
        return _expr_to_latex(node.body, parent_prec)
    if isinstance(node, ast.Constant):
        return _format_scalar_for_latex(node.value)
    if isinstance(node, ast.Name):
        name = node.id
        if name == "pi":
            return r"\pi"
        if name == "e":
            return "e"
        return _identifier_to_latex(name)
    if isinstance(node, ast.UnaryOp):
        operand = _expr_to_latex(node.operand, 4)
        if isinstance(node.op, ast.USub):
            text = rf"-{operand}"
        elif isinstance(node.op, ast.UAdd):
            text = rf"+{operand}"
        else:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return rf"\left({text}\right)" if parent_prec > 4 else text
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            prec = 1
            text = rf"{_expr_to_latex(node.left, prec)} + {_expr_to_latex(node.right, prec)}"
        elif isinstance(node.op, ast.Sub):
            prec = 1
            text = rf"{_expr_to_latex(node.left, prec)} - {_expr_to_latex(node.right, prec + 1)}"
        elif isinstance(node.op, ast.Mult):
            prec = 2
            text = rf"{_expr_to_latex(node.left, prec)} \cdot {_expr_to_latex(node.right, prec)}"
        elif isinstance(node.op, ast.Div):
            prec = 2
            text = rf"\frac{{{_expr_to_latex(node.left)}}}{{{_expr_to_latex(node.right)}}}"
        elif isinstance(node.op, ast.Pow):
            prec = 3
            text = rf"{_expr_to_latex(node.left, prec)}^{{{_expr_to_latex(node.right)}}}"
        elif isinstance(node.op, ast.Mod):
            prec = 2
            text = rf"{_expr_to_latex(node.left, prec)} \bmod {_expr_to_latex(node.right, prec)}"
        elif isinstance(node.op, ast.FloorDiv):
            prec = 2
            text = rf"\left\lfloor \frac{{{_expr_to_latex(node.left)}}}{{{_expr_to_latex(node.right)}}} \right\rfloor"
        else:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return rf"\left({text}\right)" if parent_prec > prec else text
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed.")
        func_name = node.func.id
        args = [_expr_to_latex(arg) for arg in node.args]
        if func_name == "sqrt" and len(args) == 1:
            return rf"\sqrt{{{args[0]}}}"
        if func_name == "abs" and len(args) == 1:
            return rf"\left|{args[0]}\right|"
        if func_name in {"min", "max"} and args:
            return rf"\{func_name}\left({', '.join(args)}\right)"
        return rf"\operatorname{{{func_name}}}\left({', '.join(args)}\right)"
    if isinstance(node, ast.Compare):
        left = _expr_to_latex(node.left)
        pieces: list[str] = []
        current = left
        for op, comparator in zip(node.ops, node.comparators):
            right = _expr_to_latex(comparator)
            op_text = {
                ast.Lt: "<",
                ast.LtE: r"\le",
                ast.Gt: ">",
                ast.GtE: r"\ge",
                ast.Eq: "=",
                ast.NotEq: r"\ne",
            }.get(type(op))
            if op_text is None:
                raise ValueError(f"Unsupported comparison: {type(op).__name__}")
            pieces.append(rf"{current} {op_text} {right}")
            current = right
        return r" \land ".join(pieces)
    if isinstance(node, ast.BoolOp):
        op_text = r"\land" if isinstance(node.op, ast.And) else r"\lor"
        return f" {op_text} ".join(_expr_to_latex(val) for val in node.values)
    if isinstance(node, ast.IfExp):
        return rf"\begin{{cases}}{_expr_to_latex(node.body)}, & \text{{if }} {_expr_to_latex(node.test)} \\ {_expr_to_latex(node.orelse)}, & \text{{otherwise}}\end{{cases}}"
    if isinstance(node, ast.Subscript):
        return rf"{_expr_to_latex(node.value)}\left[{_expr_to_latex(node.slice)}\right]"
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def expression_to_latex(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    return _expr_to_latex(tree)


def step_to_latex(name: str, expression: str, value: Any, unit: str | None = None) -> str:
    lhs = _identifier_to_latex(name, unit)
    rhs = expression_to_latex(expression)
    unit_latex = _unit_to_latex(unit or "")
    value_latex = _format_scalar_for_latex(value)
    if unit_latex:
        return rf"{lhs} = {rhs} = {value_latex}\,{unit_latex}"
    return rf"{lhs} = {rhs} = {value_latex}"


# ── Pydantic models ──────────────────────────────────────────────────


class Equation(BaseModel):
    name: str = Field(
        description="Variable name for the result (e.g. 'A_net', 'N_t_Rd'). "
        "Later equations can reference this name."
    )
    expression: str = Field(
        description="Math expression to evaluate. Can reference variables and "
        "earlier equation results by name. Supports: +, -, *, /, **, sqrt(), "
        "min(), max(), abs(), round(), trig functions, pi, e. "
        "Example: 'A - n_holes * d0 * t'"
    )
    unit: Optional[str] = Field(
        default=None,
        description="Unit of the result (e.g. 'mm²', 'kN', 'MPa'). For documentation.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description (e.g. 'Net cross-section area per EC3 6.2.2.2').",
    )


class MathCalculatorInput(BaseModel):
    equations: list[Equation] = Field(
        description="Ordered list of equations to evaluate sequentially. "
        "Each equation's result is available to subsequent equations by name."
    )
    variables: dict[str, float] = Field(
        description="Input variables with their numeric values. "
        "Example: {'A': 5380, 'n_holes': 2, 'd0': 22, 't': 10.7}"
    )


# ── Handler ──────────────────────────────────────────────────────────


def calculate(inp: MathCalculatorInput) -> dict:
    """Evaluate equations sequentially, building up a namespace."""

    # Build initial namespace: user variables + safe math functions/constants
    namespace: dict[str, Any] = dict(_SAFE_FUNCTIONS)
    namespace.update(inp.variables)

    results: list[dict[str, Any]] = []
    outputs: dict[str, Any] = {}

    for eq in inp.equations:
        try:
            value = safe_eval(eq.expression, namespace)
        except Exception as exc:
            raise ValueError(
                f"Error evaluating equation '{eq.name}' "
                f"(expression: {eq.expression}): {exc}"
            ) from exc

        # Round floats for clean output
        if isinstance(value, float):
            # Keep up to 6 significant figures
            rounded = round(value, 6)
            # But if it's a "clean" number, simplify
            if abs(rounded - round(rounded, 2)) < 1e-9:
                rounded = round(value, 2)
            value = rounded

        # Store result in namespace for subsequent equations
        namespace[eq.name] = value

        step = {
            "name": eq.name,
            "expression": eq.expression,
            "value": value,
            "latex": step_to_latex(eq.name, eq.expression, value, eq.unit),
        }
        if eq.unit:
            step["unit"] = eq.unit
        if eq.description:
            step["description"] = eq.description

        results.append(step)
        outputs[eq.name] = value

    return {
        "inputs_used": {
            "variables": inp.variables,
            "equation_count": len(inp.equations),
        },
        "intermediate": {
            "steps": results,
        },
        "outputs": outputs,
        "clause_references": [],
        "notes": [
            f"{r['name']} = {r['expression']} = {r['value']}"
            + (f" {r['unit']}" if r.get('unit') else "")
            for r in results
        ],
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=MathCalculatorInput, handler=calculate)
