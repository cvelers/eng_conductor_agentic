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
