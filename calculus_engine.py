"""
Math engine for the Calculus Python Tutor.

Deliberately free of Streamlit imports so every function here can be unit
tested with plain Python. All UI code lives in calculus_tutor.py.

Improvements over the original single-file draft:
  * Student-friendly parser (accepts `2x`, `x^2`, `x²`, `sin^2(x)`, `|x|`,
    `3sin(x)`) with error messages that say what to fix.
  * Implicit differentiation really differentiates, instead of hard-coding
    the answer into the question.
  * Two-sided limit analysis that can honestly report "does not exist".
  * Rule-by-rule derivative walkthrough built from the expression structure.
  * Graphs with adaptive y-limits and broken lines at asymptotes, so `1/x`
    and `tan(x)` do not draw fake vertical connectors.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import sympy as sp
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe: no display needed
import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

# --------------------------------------------------------------------------
# Symbols
# --------------------------------------------------------------------------
X = sp.Symbol("x", real=True)
Y = sp.Symbol("y", real=True)
H = sp.Symbol("h", real=True)
DYDX = sp.Symbol("dy/dx")

ALLOWED_SYMBOLS = {X, Y, H}

LOCALS = {
    "x": X,
    "y": Y,
    "h": H,
    "pi": sp.pi,
    "e": sp.E,
    "ln": sp.log,
    "log": sp.log,
    "log10": sp.log,
    "sqrt": sp.sqrt,
    "cbrt": sp.cbrt,
    "abs": sp.Abs,
    "sec": sp.sec,
    "csc": sp.csc,
    "cot": sp.cot,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "arcsin": sp.asin,
    "arccos": sp.acos,
    "arctan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
}

TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication_application)

# Unicode that students paste in from notes and web pages.
UNICODE_FIXES = {
    "−": "-", "–": "-", "—": "-", "‐": "-",
    "×": "*", "·": "*", "∗": "*",
    "÷": "/",
    "（": "(", "）": ")", "［": "[", "］": "]",
    "＝": "=", "：": ":", "，": ",",
    "√": "sqrt", "π": "pi",
    "⁰": "**0", "¹": "**1", "²": "**2", "³": "**3", "⁴": "**4",
    "⁵": "**5", "⁶": "**6", "⁷": "**7", "⁸": "**8", "⁹": "**9",
    "⁻": "**-",
    "‘": "'", "’": "'",
}


class ExpressionError(ValueError):
    """A parsing problem we can explain to a student in plain English."""

    def __init__(self, friendly: str, technical: str = ""):
        super().__init__(friendly)
        self.friendly = friendly
        self.technical = technical


def _clean(raw: str, *, strip_lhs: bool = True) -> str:
    """Normalise notation before handing the string to SymPy."""
    text = str(raw).strip()
    for bad, good in UNICODE_FIXES.items():
        text = text.replace(bad, good)

    # Drop a leading definition such as `f(x) =` or `y =` — but never when we
    # are parsing a whole equation, where the left side is meaningful.
    if strip_lhs:
        text = re.sub(
            r"^\s*(?:[fFgG]\s*\(\s*[xX]\s*\)|[fFgG]\s*[xX]|[yY])\s*=\s*", "", text
        )
    return text.strip()


def _pipes_to_abs(text: str) -> str:
    """Turn |...| into Abs(...) so the parser understands it."""
    out, depth, i = [], 0, 0
    while i < len(text):
        ch = text[i]
        if ch == "|":
            if depth == 0:
                out.append("Abs(")
                depth += 1
            else:
                out.append(")")
                depth -= 1
        else:
            out.append(ch)
        i += 1
    if depth:
        raise ExpressionError(
            "The vertical bars | | must come in pairs, for example `abs(x - 1)` or `|x|`.",
            "unbalanced absolute-value bars",
        )
    return "".join(out)


def parse_expression(raw: str, *, allow_y: bool = True) -> sp.Expr:
    """Parse a student expression into a SymPy expression.

    Raises ExpressionError with a readable message when the input cannot work.
    """
    text = _clean(raw)
    if not text:
        raise ExpressionError("Type a function first, for example `x**2 - 4` or `sin(x)/x`.")

    try:
        text = _pipes_to_abs(text)
    except ExpressionError:
        raise

    # Parentheses must balance, otherwise the message below is more useful
    # than SymPy's SyntaxError.
    if text.count("(") != text.count(")"):
        left = text.count("(")
        right = text.count(")")
        raise ExpressionError(
            f"You have {left} opening brackets and {right} closing brackets. "
            "Add the missing one, for example `(x**2 - 1)/(x - 1)`."
        )
    if text.count("[") != text.count("]"):
        raise ExpressionError("Use round brackets ( ) for grouping, not square brackets [ ].")

    local = dict(LOCALS)
    if not allow_y:
        local.pop("y", None)

    try:
        expr = parse_expr(text, local_dict=local, transformations=TRANSFORMS, evaluate=True)
    except Exception as err:  # noqa: BLE001 - any parse failure becomes friendly text
        raise ExpressionError(_friendly_parse_error(err, text), str(err)) from err

    if not isinstance(expr, sp.Expr):
        raise ExpressionError(
            f"`{raw.strip()}` did not come out as a formula. Try something like `x**2 + 3*x`.",
            repr(expr),
        )

    unknown = {s for s in expr.free_symbols if s not in ALLOWED_SYMBOLS}
    if unknown:
        names = ", ".join(sorted(str(s) for s in unknown))
        raise ExpressionError(
            f"I do not know the name(s) {names}. The tutor uses `x` (and `y` for implicit "
            "equations). Known functions are sin, cos, tan, sec, exp, ln, sqrt, abs.",
            f"free symbols: {names}",
        )
    if not allow_y and Y in expr.free_symbols:
        raise ExpressionError("This box takes a function of x only. Remove any `y`.")
    return expr


def _friendly_parse_error(err: Exception, text: str) -> str:
    msg = str(err)
    lowered = msg.lower()
    if "invalid syntax" in lowered:
        return (
            f"I could not read `{text}`. Usual causes: a missing `*` between a number and a "
            "letter (`2*x`, not `2 x` in some places), a missing `**` for powers (`x**2`), "
            "or two expressions with no operator between them."
        )
    if "unknown" in lowered or "not defined" in lowered:
        return f"`{text}` uses a name or function the tutor does not know. Try sin, cos, tan, exp, ln, sqrt."
    return f"I could not read `{text}`. Check the brackets and operators. ({msg[:160]})"


def parse_equation(raw: str) -> tuple[sp.Expr, sp.Expr]:
    """Parse `left = right` into two expressions, keeping variables symbolic."""
    text = _clean(raw, strip_lhs=False)
    if "=" not in text:
        raise ExpressionError(
            "An equation needs an equals sign, for example `x**2 + y**2 = 25`."
        )
    if text.count("=") != 1:
        raise ExpressionError("Please write one equation with a single `=` sign.")
    lhs_raw, rhs_raw = text.split("=")
    return parse_expression(lhs_raw, allow_y=True), parse_expression(rhs_raw, allow_y=True)


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------
def to_text(expr, limit: int = 90) -> str:
    try:
        out = sp.sstr(sp.nsimplify(expr) if isinstance(expr, sp.Expr) else expr)
    except Exception:  # noqa: BLE001
        out = str(expr)
    out = out.replace("\n", " ")
    return out if len(out) <= limit else out[: limit - 1] + "…"


def to_latex(expr, limit: int = 240) -> str:
    try:
        out = sp.latex(expr)
    except Exception:  # noqa: BLE001
        out = str(expr)
    out = out.replace("\n", " ").strip()
    return out if len(out) <= limit else out[: limit - 1] + r"\;\text{…}"


def number_text(value) -> str:
    """Compact display for a SymPy number; keeps sqrt(2) style exact forms."""
    if isinstance(value, (int, float, np.floating)):
        if value == int(value):
            return str(int(value))
        return f"{float(value):.4g}"
    if value is sp.zoo:
        return "undefined (infinite)"
    if value is sp.oo:
        return "+∞"
    if value is -sp.oo:
        return "−∞"
    if value is sp.nan:
        return "undefined"
    if getattr(value, "is_number", False):
        approx = sp.N(value)
        if approx.is_real and abs(approx - round(float(approx))) > 1e-9:
            return f"{to_text(value)} ≈ {float(approx):.4g}"
        return to_text(value)
    return to_text(value)


# --------------------------------------------------------------------------
# Domain and continuity
# --------------------------------------------------------------------------
@dataclass
class DomainReport:
    latex: str
    text: str
    restrictions: list[str] = field(default_factory=list)
    ok: bool = True


def domain_of(expr: sp.Expr, var: sp.Symbol = X) -> DomainReport:
    """Find where a real function of one variable is defined."""
    restrictions: list[str] = []

    for denom in _denominators(expr, var):
        try:
            bad = sp.solveset(sp.Eq(denom, 0), var, domain=sp.S.Reals)
        except Exception:  # noqa: BLE001
            continue
        for point in _sample_points(bad, 6):
            restrictions.append(f"{to_text(var)} = {number_text(point)} (makes a denominator zero)")

    for inner in _radicands(expr, var, even_only=True):
        try:
            ok_region = sp.solveset(sp.Ge(inner, 0), var, domain=sp.S.Reals)
        except Exception:  # noqa: BLE001
            continue
        restrictions.append(f"inside the square root needs {to_latex(inner)} ≥ 0, i.e. {to_text(ok_region)}")

    for arg in _log_args(expr, var):
        try:
            ok_region = sp.solveset(sp.Gt(arg, 0), var, domain=sp.S.Reals)
        except Exception:  # noqa: BLE001
            continue
        restrictions.append(f"inside a logarithm needs {to_latex(arg)} > 0, i.e. {to_text(ok_region)}")

    try:
        dom = sp.calculus.util.continuous_domain(sp.S.Zero if expr == 0 else expr, var, sp.S.Reals)
        dom_latex, dom_text = to_latex(dom), to_text(dom)
    except Exception:  # noqa: BLE001
        dom_latex = dom_text = ""

    if not restrictions and not dom_text:
        return DomainReport("", "Defined for all real numbers (no denominator, root or log to worry about).", ok=True)

    return DomainReport(dom_latex, dom_text or "See the restrictions below.", restrictions, ok=True)


def _walk(expr):
    yield expr
    for arg in getattr(expr, "args", ()):
        yield from _walk(arg)


def _denominators(expr, var):
    """Every sub-expression that must not be zero, without duplicates."""
    out: list[sp.Expr] = []
    seen: set[str] = set()

    def add(denom):
        if not denom.has(var):
            return
        key = sp.sstr(sp.factor(denom))
        if key not in seen:
            seen.add(key)
            out.append(denom)

    try:
        _, denom = sp.fraction(sp.together(sp.sympify(expr)))
        add(denom)
    except Exception:  # noqa: BLE001
        pass
    for node in _walk(expr):
        # Only negative powers are division; logarithm arguments are handled
        # separately, because they need > 0 rather than != 0.
        if node.func is sp.Pow and node.args[1].is_number and node.args[1] < 0:
            add(node.args[0])
    return out


def _radicands(expr, var, even_only: bool = False):
    out = []
    for node in _walk(expr):
        if node.func is sp.Pow and node.args[1].is_number:
            power = node.args[1]
            if power == sp.Rational(1, 2):
                out.append(node.args[0])
            elif not even_only and power.is_Rational and power.p == 1 and power.q % 2 == 0:
                out.append(node.args[0])
    return [r for r in out if r.has(var)]


def _log_args(expr, var):
    out = []
    for node in _walk(expr):
        if node.func is sp.log and node.args[0].has(var):
            out.append(node.args[0])
    return out


def _sample_points(sol, count: int):
    """Best-effort numeric list from a solveset answer."""
    try:
        if isinstance(sol, sp.FiniteSet):
            return list(sol)[:count]
        if isinstance(sol, sp.Complement):
            bad = sol.complement
            if isinstance(bad, sp.FiniteSet):
                return list(bad)[:count]
        if isinstance(sol, sp.imageset.ImageSet):
            base = sol.lamda.expr
            counter = sol.lamda.variables[0]
            return [sp.simplify(base.subs(counter, k)) for k in range(-2, 2)][:count]
    except Exception:  # noqa: BLE001
        pass
    return []


# --------------------------------------------------------------------------
# Limits
# --------------------------------------------------------------------------
@dataclass
class LimitReport:
    point: object
    two_sided: object
    left: object
    right: object
    exists: bool
    kind: str  # finite | infinite | does-not-exist
    note: str


def limit_at(expr: sp.Expr, point, var: sp.Symbol = X) -> LimitReport:
    direct = None
    try:
        direct = expr.subs(var, point)
    except Exception:  # noqa: BLE001
        direct = None

    left = right = two = None
    try:
        two = sp.limit(expr, var, point, "+-")
    except Exception:  # noqa: BLE001
        two = None
    for direction, store in (("-", "left"), ("+", "right")):
        try:
            value = sp.limit(expr, var, point, direction)
        except Exception:  # noqa: BLE001
            value = None
        if store == "left":
            left = value
        else:
            right = value

    direct_ok = direct is not None and direct.is_finite is not None and str(direct) != "nan"
    if direct_ok and not (direct.has(sp.zoo) or direct.has(sp.nan) or direct.has(sp.oo)):
        note = f"Direct substitution worked: f({number_text(point)}) = {number_text(direct)}."
        return LimitReport(point, two if two is not None else direct, left, right, True, "finite", note)

    if left is not None and right is not None:
        if left == right:
            kind = "infinite" if (left.has(sp.oo) or left.has(sp.zoo) or left.has(-sp.oo)) else "finite"
            note = (
                "Direct substitution gives an indeterminate form, so we simplify first "
                "(factor, cancel, or use a known limit). Both sides agree."
            )
            return LimitReport(point, left, left, right, True, kind, note)
        note = (
            f"Left-hand limit is {number_text(left)} but right-hand limit is {number_text(right)}. "
            "Because they differ, the two-sided limit does not exist."
        )
        return LimitReport(point, None, left, right, False, "does-not-exist", note)

    note = "SymPy could not decide the limit from this input; try simplifying the expression."
    return LimitReport(point, two, left, right, two is not None, "unknown", note)


# --------------------------------------------------------------------------
# Derivatives: structure-aware walkthrough
# --------------------------------------------------------------------------
def derivative_walkthrough(expr: sp.Expr, var: sp.Symbol = X) -> dict:
    """Name the rules a student should apply, then verify with SymPy.

    The final answer always comes from sp.diff, so the walkthrough can never
    disagree with the correct result.
    """
    expr = sp.sympify(expr)
    result = sp.diff(expr, var)
    steps: list[dict] = []

    terms = expr.as_ordered_terms()
    if len(terms) > 1:
        steps.append({
            "rule": "Sum / difference rule",
            "formula": r"(f \pm g)' = f' \pm g'",
            "detail": f"Treat each of the {len(terms)} terms separately, then add the results.",
        })
        work_terms = terms
    else:
        work_terms = [expr]

    for term in work_terms:
        steps.extend(_term_steps(term, var))

    steps.append({
        "rule": "Collect like terms",
        "formula": "",
        "detail": f"f'({to_text(var)}) = {to_text(result)}",
    })

    check = numeric_derivative_check(expr, var)
    return {"steps": steps, "result": result, "check": check}


def _term_steps(term: sp.Expr, var: sp.Symbol) -> list[dict]:
    out: list[dict] = []

    if term.is_number:
        return [{"rule": "Constant rule", "formula": r"(c)' = 0",
                 "detail": f"{to_text(term)} is a constant, so its derivative is 0."}]

    if term.func is sp.Pow and term.args[1].is_number:
        base, power = term.args
        if base == var:
            return [{
                "rule": "Power rule",
                "formula": r"(x^n)' = n\,x^{n-1}",
                "detail": f"n = {number_text(power)} → {number_text(power)}·{to_text(var)}^({number_text(power - 1)})",
            }]
        out.append({
            "rule": "Chain rule (power outside)",
            "formula": r"([u^n])' = n\,u^{\,n-1}\cdot u'",
            "detail": f"inside u = {to_text(base)}",
        })
        out.extend(_term_steps(base, var))
        return out

    if term.func is sp.Mul:
        factors = list(term.args)
        plain = [f for f in factors if f.is_number]
        real = [f for f in factors if not f.is_number]
        if len(real) >= 2:
            out.append({
                "rule": "Product rule",
                "formula": r"(fg)' = f'g + fg'",
                "detail": f"{len(factors)} factors: " + " · ".join(to_text(f) for f in factors),
            })
            for f in real:
                out.append({"rule": "  differentiate", "formula": "",
                            "detail": f"{to_text(f)} → {to_text(sp.diff(f, var))}"})
        if plain:
            out.append({
                "rule": "Constant multiple rule",
                "formula": r"(c\,f)' = c\,f'",
                "detail": f"keep {number_text(plain[0])} in front.",
            })

    num, den = sp.fraction(sp.together(term))
    if den.has(var):
        out.insert(0, {
            "rule": "Quotient rule",
            "formula": r"(f/g)' = (f'g - fg')/g^2",
            "detail": f"f = {to_text(num)},  g = {to_text(den)}",
        })

    for func in (sp.sin, sp.cos, sp.tan, sp.exp, sp.log, sp.sec, sp.csc, sp.cot,
                 sp.asin, sp.acos, sp.atan, sp.sinh, sp.cosh, sp.tanh):
        if term.has(func) and term.func is not func:
            continue
        if term.func is func:
            inner = term.args[0]
            table = {
                sp.sin: r"(\sin u)' = \cos u \cdot u'",
                sp.cos: r"(\cos u)' = -\sin u \cdot u'",
                sp.tan: r"(\tan u)' = \sec^2 u \cdot u'",
                sp.sec: r"(\sec u)' = \sec u\tan u \cdot u'",
                sp.csc: r"(\csc u)' = -\csc u\cot u \cdot u'",
                sp.cot: r"(\cot u)' = -\csc^2 u \cdot u'",
                sp.exp: r"(e^u)' = e^u \cdot u'",
                sp.log: r"(\ln u)' = u'/u",
                sp.asin: r"(\arcsin u)' = u'/\sqrt{1-u^2}",
                sp.acos: r"(\arccos u)' = -u'/\sqrt{1-u^2}",
                sp.atan: r"(\arctan u)' = u'/(1+u^2)",
                sp.sinh: r"(\sinh u)' = \cosh u \cdot u'",
                sp.cosh: r"(\cosh u)' = \sinh u \cdot u'",
                sp.tanh: r"(\tanh u)' = \operatorname{sech}^2 u \cdot u'",
            }
            step = {
                "rule": f"{func.__name__.upper()} rule",
                "formula": table.get(func, ""),
                "detail": f"inside u = {to_text(inner)}",
            }
            out.append(step)
            if inner != var:
                out.append({
                    "rule": "Chain rule",
                    "formula": r"\text{multiply by } u'",
                    "detail": f"u' = {to_text(sp.diff(inner, var))}",
                })
            return out

    if not out:
        out.append({"rule": "Basic rules", "formula": "", "detail": f"differentiate {to_text(term)} directly."})
    return out


def numeric_derivative_check(expr: sp.Expr, var: sp.Symbol = X, at=sp.Rational(3, 2)) -> dict:
    """Compare the symbolic derivative with a symmetric difference quotient."""
    try:
        exact = sp.diff(expr, var).subs(var, at)
        f = sp.lambdify(var, expr, modules=["numpy"])
        delta = 1e-5
        approx = (float(f(float(at) + delta)) - float(f(float(at) - delta))) / (2 * delta)
        exact_f = float(sp.N(exact))
        if not (math.isfinite(approx) and math.isfinite(exact_f)):
            agree = None
            note = (f"Numerical check skipped: the slope at {number_text(at)} is not finite "
                    "(vertical tangent or the point sits outside the domain).")
        else:
            # Relative tolerance: an absolute 1e-3 threshold would wrongly flag
            # steep-but-correct curves such as (5x-1)^10.
            scale = max(1.0, abs(exact_f))
            agree = abs(approx - exact_f) / scale < 2e-4
            note = (
                f"f'({number_text(at)}) = {number_text(exact)} agrees with the numerical slope "
                f"{approx:.5g} from [f(a+h) − f(a−h)] / 2h — the two routes to the answer match."
                if agree else
                f"Symbolic f'({number_text(at)}) = {number_text(exact)} vs numerical slope {approx:.5g}. "
                "They differ, which usually means a corner, cusp or discontinuity at that point — "
                "try evaluating elsewhere."
            )
        return {"at": at, "exact": exact, "approx": approx, "agree": agree, "note": note}
    except Exception as err:  # noqa: BLE001
        return {"at": at, "exact": None, "approx": None, "agree": None, "note": f"Check skipped: {err}"}


# --------------------------------------------------------------------------
# Tangent line
# --------------------------------------------------------------------------
def tangent_line(expr: sp.Expr, point, var: sp.Symbol = X) -> dict:
    slope = sp.diff(expr, var).subs(var, point)
    height = expr.subs(var, point)
    line = sp.expand(height + slope * (var - point))
    normal_slope = -1 / slope if slope != 0 else sp.oo
    return {
        "point": (point, height),
        "slope": slope,
        "height": height,
        "line": line,
        "point_slope_form": sp.Symbol("y") - height - slope * (var - point),
        "normal_slope": normal_slope,
        "exists": slope.is_finite is not False,
    }


# --------------------------------------------------------------------------
# Implicit differentiation  (the honest version)
# --------------------------------------------------------------------------
def implicit_derivative(equation_text: str, at: tuple | None = None) -> dict:
    """Differentiate F(x, y) = G(x, y) with respect to x and solve for dy/dx.

    The student's equation is differentiated term by term; nothing is
    hard-coded, so changing the input changes the answer.
    """
    lhs, rhs = parse_equation(equation_text)
    y_of_x = sp.Function("y")(X)

    # Temporarily map the student's `y` to y(x) so the chain rule fires.
    lhs_t = lhs.subs(Y, y_of_x)
    rhs_t = rhs.subs(Y, y_of_x)
    combined = sp.expand(lhs_t - rhs_t)

    derived = sp.diff(combined, X)
    dy = sp.Derivative(y_of_x, X)
    try:
        solved = sp.solve(sp.Eq(derived, 0), dy, dict=True)
    except Exception as err:  # noqa: BLE001
        raise ExpressionError(
            "I differentiated both sides but could not isolate dy/dx. Try an equation where y "
            "appears to the first power, for example `x**2 + y**2 = 25`.",
            str(err),
        ) from err
    if not solved:
        raise ExpressionError(
            "The equation has no y in it, so there is nothing to differentiate implicitly. "
            "Use the Python calculus lab instead."
        )

    dydx = sp.simplify(list(solved[0].values())[0])
    # Rewrite y(x) back to plain y, and Derivative(y(x), x) to dy/dx, for display.
    pretty = dydx.subs(dy, DYDX).replace(y_of_x, Y)
    derived_pretty = derived.subs(dy, DYDX).replace(y_of_x, Y)

    result = {
        "lhs": lhs,
        "rhs": rhs,
        "differentiated": derived_pretty,
        "dydx": pretty,
        "raw": dydx,
        "_y_of_x": y_of_x,
    }

    if at is not None:
        xv, yv = at
        try:
            slope = sp.simplify(dydx.subs({X: xv, y_of_x: yv}))
            result["slope"] = slope
            residual = sp.simplify(lhs.subs({X: xv, Y: yv}) - rhs.subs({X: xv, Y: yv}))
            result["on_curve"] = bool(residual == 0)
        except Exception as err:  # noqa: BLE001
            result["slope_error"] = str(err)
    return result


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def _finite_samples(expr: sp.Expr, x_min: float, x_max: float, n: int = 900):
    fn = sp.lambdify(X, expr, modules=["numpy"])
    xs = np.linspace(x_min, x_max, n)
    with np.errstate(all="ignore", divide="ignore", invalid="ignore", over="ignore"):
        ys = np.asarray(fn(xs), dtype=float)
    if ys.ndim == 0:
        ys = np.full(xs.shape, float(ys))
    if ys.shape[0] != xs.shape[0]:
        ys = np.resize(ys, xs.shape)
    ys = ys.astype(float, copy=True)
    ys[~np.isfinite(ys)] = np.nan
    return xs, ys


def _robust_ylim(ys: np.ndarray, pad: float = 0.12) -> tuple[float, float] | None:
    good = ys[np.isfinite(ys)]
    if good.size < 4:
        return None
    lo, hi = np.percentile(good, [2, 98])
    if hi - lo < 1e-9:
        mid = float(hi)
        lo, hi = mid - 1.0, mid + 1.0
    span = hi - lo
    return lo - pad * span, hi + pad * span


def _break_at_asymptotes(xs: np.ndarray, ys: np.ndarray, ylim: tuple[float, float]) -> np.ndarray:
    """Insert NaN gaps where the curve jumps across the frame (true asymptote)."""
    out = ys.copy()
    span = abs(ylim[1] - ylim[0])
    step = xs[1] - xs[0]
    for i in range(1, len(out)):
        a, b = out[i - 1], out[i]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        jump = abs(b - a)
        # A jump far bigger than the visible height, with a sign change, is a pole.
        if jump > 6 * span and (a > ylim[1] or a < ylim[0] or b > ylim[1] or b < ylim[0]):
            out[i] = np.nan
        elif jump / max(step, 1e-12) > 250 * span and a * b < 0:
            out[i] = np.nan
    return out


def plot_series(
    series: list[tuple[sp.Expr, str]],
    x_min: float = -5.0,
    x_max: float = 5.0,
    title: str = "",
    markers: list[tuple[float, float, str]] | None = None,
    height: float = 4.0,
    width: float = 8.0,
):
    """Draw one or more functions of x with sane axis limits.

    Returns a matplotlib Figure. Caller must close it (see show_figure).
    """
    series = [(sp.sympify(e), label) for e, label in series if e is not None]
    if not series:
        raise ExpressionError("Nothing to plot yet — enter a function of x.")

    fig, ax = plt.subplots(figsize=(width, height), dpi=110)
    all_ys = []
    plotted = []
    for expr, label in series:
        try:
            xs, ys = _finite_samples(expr, x_min, x_max)
        except Exception:  # noqa: BLE001
            continue
        all_ys.append(ys)
        plotted.append((xs, ys, label))

    if not plotted:
        ax.set_title(title or "No plottable values in this range")
        return fig

    candidate = np.concatenate([y[np.isfinite(y)] for _, y, _ in plotted]) if any(
        np.isfinite(y).any() for _, y, _ in plotted) else np.array([0.0])
    ylim = _robust_ylim(candidate)
    if ylim is None:
        ylim = (-1.0, 1.0)

    for xs, ys, label in plotted:
        clipped = ys.copy()
        clipped[(clipped > ylim[1] * 3 + 10) | (clipped < ylim[0] * 3 - 10)] = np.nan
        broken = _break_at_asymptotes(xs, clipped, ylim)
        ax.plot(xs, broken, linewidth=1.6, label=label)

    ax.set_ylim(*ylim)
    ax.set_xlim(x_min, x_max)
    ax.axhline(0, color="#94a3b8", linewidth=0.9)
    ax.axvline(0, color="#94a3b8", linewidth=0.9)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)

    for mx, my, label in markers or []:
        if ylim[0] <= my <= ylim[1]:
            ax.scatter([mx], [my], zorder=5, color="#ef4444", s=45)
            ax.annotate(label, (mx, my), textcoords="offset points", xytext=(8, 8), fontsize=9)

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    return fig


def show_figure(fig) -> None:
    """Close a figure after Streamlit has rendered it."""
    plt.close(fig)


def parse_number(raw: str, default: float | None = None, name: str = "value"):
    text = _clean(str(raw))
    if not text:
        if default is None:
            raise ExpressionError(f"Enter a number for {name}, for example `2` or `pi/2`.")
        return sp.nsimplify(default)
    try:
        value = parse_expression(text, allow_y=False)
    except ExpressionError as err:
        raise ExpressionError(f"{name} should be a number or a simple exact value like `pi/2`. {err.friendly}") from err
    if value.free_symbols:
        raise ExpressionError(f"{name} must be a specific number, not an expression containing x.")
    return value


def to_float(value, default: float) -> float:
    try:
        out = float(sp.N(value))
        return out if math.isfinite(out) else default
    except Exception:  # noqa: BLE001
        return default
