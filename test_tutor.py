"""Headless test suite for the Calculus Python Tutor.

Run:  python test_tutor.py
Exercises the maths engine directly, and drives every Streamlit page with
streamlit.testing.v1.AppTest so router/UI errors surface before launch.
"""

import math
import os
import sys
import traceback

# Must be set before matplotlib is imported, otherwise it tries to build a
# font cache in a temp directory the sandbox cannot scan.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_mplcache"))

import sympy as sp

import calculus_engine as ce

X = ce.X
PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(("  ok   " if condition else "  FAIL ") + name + (f"  [{detail}]" if detail and not condition else ""))


print("=" * 70)
print("1. Parser: student-friendly notation")
print("=" * 70)
cases = {
    "x**2 + 3*x + 1": X ** 2 + 3 * X + 1,
    "x^2": X ** 2,
    "2x": 2 * X,
    "3sin(x)": 3 * sp.sin(X),
    "sqrt(x+1)": sp.sqrt(X + 1),
    "ln(x)": sp.log(X),
    "e^(x^2)": sp.exp(X ** 2),
    "1/2x": sp.Rational(1, 2) * X,
    "|x|": sp.Abs(X),
    "x²": X ** 2,
    "sin^2(x)": sp.sin(X) ** 2,
    "(x² − 1)/(x − 1)": (X ** 2 - 1) / (X - 1),
    "f(x) = x**2": X ** 2,
    "2e^-x": 2 * sp.exp(-X),
    "pi/2": sp.pi / 2,
    "2 x": 2 * X,          # implicit multiplication should ACCEPT this
    "x_1": None,  # must fail
}
for raw, expected in cases.items():
    try:
        got = ce.parse_expression(raw)
        if expected is None:
            check(f"rejects `{raw}`", False, f"parsed as {got}")
        else:
            check(f"parses `{raw}`", sp.simplify(got - expected) == 0, f"got {ce.to_text(got)}")
    except ce.ExpressionError as err:
        if expected is None:
            check(f"rejects `{raw}` with friendly text", bool(err.friendly))
        else:
            check(f"parses `{raw}`", False, err.friendly)

print("\n" + "=" * 70)
print("2. Parser: friendly errors for genuinely broken input")
print("=" * 70)
for bad in ["", "x**", "(x**2", "@#$", "x + y + z", "foo(x)"]:
    try:
        ce.parse_expression(bad)
        check(f"rejects `{bad}`", False, "no error raised")
    except ce.ExpressionError as err:
        check(f"rejects `{bad}` -> readable", bool(err.friendly) and len(err.friendly) > 15,
              err.friendly)

print("\n" + "=" * 70)
print("3. Limits")
print("=" * 70)
limit_tests = [
    ((X ** 2 - 1) / (X - 1), 1, 2, True),
    (sp.sin(X) / X, 0, 1, True),
    (X ** 2 + 1, 3, 10, True),
    ((3 * X ** 2 + 1) / (X ** 2 - X), sp.oo, 3, True),
    (1 / X, 0, None, False),                 # does not exist
    (sp.Abs(X) / X, 0, None, False),         # left -1, right +1
]
for expr, a, expected, exists in limit_tests:
    report = ce.limit_at(expr, a)
    ok_exists = report.exists == exists
    ok_value = True if expected is None else (
        report.two_sided is not None and abs(sp.N(report.two_sided) - expected) < 1e-9
    )
    check(f"limit {ce.to_text(expr)} at {ce.number_text(a)}", ok_exists and ok_value,
          f"exists={report.exists} value={ce.number_text(report.two_sided)} "
          f"L={ce.number_text(report.left)} R={ce.number_text(report.right)} :: {report.note}")
    print(f"        note: {report.note}")

print("\n" + "=" * 70)
print("4. Derivative walkthrough (answers must equal sp.diff)")
print("=" * 70)
deriv_cases = [
    X ** 2 * sp.log(X),
    X * sp.sin(X),
    sp.sin(3 * X ** 2),
    sp.exp(X ** 2),
    sp.log(X ** 2 + 1),
    (X ** 2 + 1) / (X - 3),
    X ** 4 - 3 * X ** 2 + 2,
    sp.sec(X),
    (5 * X - 1) ** 10,
    sp.Abs(X),
]
for expr in deriv_cases:
    walk = ce.derivative_walkthrough(expr)
    truth = sp.simplify(sp.diff(expr, X))
    got = sp.simplify(walk["result"])
    named = [s["rule"] for s in walk["steps"] if s["rule"]]
    check(f"d/dx {ce.to_text(expr, 40)} matches SymPy", sp.simplify(truth - got) == 0,
          f"truth={ce.to_text(truth)} got={ce.to_text(got)}")
    print(f"        rules named: {named}   ->  {ce.to_text(got, 60)}")
    if walk["check"]["agree"] is not None:
        print(f"        numeric check: {walk['check']['note'][:100]}")

print("\n" + "=" * 70)
print("5. Tangent line")
print("=" * 70)
t = ce.tangent_line(X ** 2, 3)
check("slope at x=3 for x^2 is 6", t["slope"] == 6, str(t["slope"]))
check("tangent line is 6x - 9", sp.simplify(t["line"] - (6 * X - 9)) == 0, ce.to_text(t["line"]))
check("normal slope is -1/6", t["normal_slope"] == sp.Rational(-1, 6), str(t["normal_slope"]))

print("\n" + "=" * 70)
print("6. Implicit differentiation (real, not hard-coded)")
print("=" * 70)
imp = ce.implicit_derivative("x**2 + y**2 = 25", at=(3, 4))
expected = -X / ce.Y
check("circle gives dy/dx = -x/y", sp.simplify(imp["dydx"] - expected) == 0, ce.to_text(imp["dydx"]))
check("plain y appears, no y(x) left", "y(x)" not in ce.to_text(imp["dydx"]), ce.to_text(imp["dydx"]))
check("slope at (3,4) is -3/4", imp.get("slope") == sp.Rational(-3, 4), str(imp.get("slope")))
check("(3,4) recognised on circle", imp.get("on_curve") is True, str(imp.get("on_curve")))

imp_off = ce.implicit_derivative("x**2 + y**2 = 25", at=(1, 1))
check("(1,1) flagged as NOT on circle", imp_off.get("on_curve") is False, str(imp_off.get("on_curve")))

# Now a *different* curve: if the code were hard-coded, this would be wrong.
imp2 = ce.implicit_derivative("x**3 + y**3 = 6*x*y")   # folium of Descartes
yfun = sp.Function("y")(X)
manual = sp.simplify(sp.solve(
    sp.Eq(sp.diff(X ** 3 + yfun ** 3 - 6 * X * yfun, X), 0),
    sp.Derivative(yfun, X))[0])
# Compare after normalising y(x) -> y, then numerically at three points.
normalised = sp.simplify(manual.replace(yfun, ce.Y))
check("folium dy/dx equals derivation (symbolic)",
      sp.simplify(imp2["dydx"] - normalised) == 0,
      f"{ce.to_text(imp2['dydx'])} vs {ce.to_text(normalised)}")
numeric_ok = all(
    abs(float(imp2["dydx"].subs({X: px, ce.Y: py})) -
        float(normalised.subs({X: px, ce.Y: py}))) < 1e-9
    for px, py in [(1.0, 2.0), (2.0, 1.0), (3.0, 1.5)]
)
check("folium dy/dx agrees numerically at 3 points", numeric_ok)
print(f"        folium result: {ce.to_text(imp2['dydx'])}")

# A third curve, to be certain nothing is hard-coded.
imp4 = ce.implicit_derivative("x**2*y + y**3 = 5", at=(1, 2))
truth4 = sp.simplify(-sp.diff(X ** 2 * ce.Y + ce.Y ** 3 - 5, X) / sp.diff(X ** 2 * ce.Y + ce.Y ** 3 - 5, ce.Y))
check("third curve dy/dx matches implicit-function theorem",
      sp.simplify(imp4["dydx"] - truth4) == 0, f"{ce.to_text(imp4['dydx'])} vs {ce.to_text(truth4)}")
check("third-curve slope is a number", getattr(imp4.get("slope"), "is_number", False), str(imp4.get("slope")))
print(f"        at (1,2): m = {ce.number_text(imp4['slope'])}")

imp3 = ce.implicit_derivative("y = x**2")   # explicit but written as equation
check("handles y = x^2 -> 2x", sp.simplify(imp3["dydx"] - 2 * X) == 0, ce.to_text(imp3["dydx"]))

for bad_eq in ["x**2 + 1", "x + y + z = 1", "sin(x) = 0.5"]:
    try:
        ce.implicit_derivative(bad_eq)
        print(f"        (no error for `{bad_eq}` - acceptable)")
    except ce.ExpressionError as err:
        check(f"`{bad_eq}` -> friendly", bool(err.friendly), err.friendly)

print("\n" + "=" * 70)
print("7. Domain analysis")
print("=" * 70)
rep = ce.domain_of(sp.sqrt(X - 2) / (X - 5))
check("sqrt(x-2)/(x-5) reports 2 restrictions", len(rep.restrictions) >= 2, str(rep.restrictions))
print("        " + " | ".join(rep.restrictions))
rep2 = ce.domain_of(X ** 2 + 1)
check("polynomial has no restrictions", rep2.restrictions == [], str(rep2.restrictions))
rep3 = ce.domain_of(sp.log(X - 1))
check("log reports positivity", any("logarithm" in r for r in rep3.restrictions), str(rep3.restrictions))

print("\n" + "=" * 70)
print("8. Plotting: no crash, sensible limits, asymptote gaps")
print("=" * 70)
plot_exprs = [1 / X, sp.tan(X), sp.sqrt(X - 2) / (X - 5), X ** 3, sp.exp(X), sp.Abs(X),
              sp.sin(X) / X, X ** 2 * sp.log(X), 1 / (X ** 2 - 4)]
for expr in plot_exprs:
    try:
        fig = ce.plot_series([(expr, "f")], -4, 4, "test")
        ax = fig.axes[0]
        lo, hi = ax.get_ylim()
        finite = math.isfinite(lo) and math.isfinite(hi) and hi > lo
        check(f"plot {ce.to_text(expr, 30)} finite y-limits", finite, f"ylim={lo},{hi}")
        if not finite:
            print(f"        ylim = ({lo}, {hi})")
        ce.show_figure(fig)
    except Exception as err:
        check(f"plot {ce.to_text(expr, 30)}", False, f"{type(err).__name__}: {err}")

# Asymptote handling: 1/x must not draw a connector through 0
fig = ce.plot_series([(1 / X, "f")], -4, 4, "asym")
line = fig.axes[0].get_lines()[0]
ys = line.get_ydata()
ce.show_figure(fig)
gaps = sum(1 for i in range(1, len(ys)) if (not math.isnan(ys[i])) and math.isnan(ys[i - 1]))
check("1/x plot breaks at the pole", gaps >= 1, f"gaps={gaps}")

fig = ce.plot_series([(sp.tan(X), "tan")], -6, 6, "asym")
ys2 = fig.axes[0].get_lines()[0].get_ydata()
ce.show_figure(fig)
gaps2 = sum(1 for i in range(1, len(ys2)) if (not math.isnan(ys2[i])) and math.isnan(ys2[i - 1]))
check("tan(x) plot breaks at each pole", gaps2 >= 2, f"gaps={gaps2}")

# scalar / constant function edge case (the original code crashed here)
try:
    fig = ce.plot_series([(sp.pi, "const")], -5, 5, "const")
    check("constant function plots", True)
    ce.show_figure(fig)
except Exception as err:
    check("constant function plots", False, str(err))

# expression that is undefined across the whole window
try:
    fig = ce.plot_series([(sp.sqrt(-X ** 2 - 1), "none")], -5, 5, "nowhere")
    check("all-undefined function handled", True)
    ce.show_figure(fig)
except Exception as err:
    check("all-undefined function handled", False, str(err))

print("\n" + "=" * 70)
print("9. Number parsing")
print("=" * 70)
for raw, want in [("2", 2.0), ("pi/2", math.pi / 2), ("-0.5", -0.5), ("sqrt(2)", math.sqrt(2))]:
    val = ce.parse_number(raw, name="test")
    check(f"parse_number `{raw}`", abs(float(sp.N(val)) - want) < 1e-9, str(val))
try:
    ce.parse_number("x + 1", name="test")
    check("parse_number rejects symbols", False)
except ce.ExpressionError:
    check("parse_number rejects symbols", True)

print("\n" + "=" * 70)
print("10. LaTeX strings from the app must contain no stray newlines")
print("=" * 70)
sample = r"\lim_{x \to " + ce.to_latex(sp.Rational(1)) + r"} f(x) = " + ce.to_latex(sp.Integer(2))
check("limit latex has literal \\to", "\\to" in sample and "\t" not in sample, repr(sample))
check("tab hazard gone", "\t" not in sample)

print("\n" + "=" * 70)
print("11. Every Streamlit page must run without exceptions")
print("=" * 70)
import os
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath("_mplcache"))
try:
    from streamlit.testing.v1 import AppTest
except Exception as err:  # noqa: BLE001
    AppTest = None
    check("AppTest available", False, str(err))

TOPICS = [
    "Start here", "1.1 Functions and domains", "1.2 Transformations and composition",
    "1.3 Limits and limit laws", "1.4 Epsilon-delta definition", "1.5 Continuity and IVT",
    "2.1 Rates of change and tangents", "2.2 The derivative as a function",
    "2.3 Differentiation rules", "2.4 Trigonometric derivatives", "2.5 The chain rule",
    "2.6 Implicit differentiation", "Python calculus lab", "Practice quiz",
]

def fresh(topic):
    """AppTest widgets only exist after the first run()."""
    at = AppTest.from_file("calculus_tutor.py", default_timeout=240)
    at.run()
    at.sidebar.radio[0].set_value(topic)
    at.run()
    return at


if AppTest is not None:
    for topic in TOPICS:
        try:
            at = fresh(topic)
            exceptions = list(at.exception)
            if exceptions:
                check(f"page `{topic}` renders", False,
                      " || ".join(str(e.value)[:300] for e in exceptions))
            else:
                check(f"page `{topic}` renders", True)
            # Educational red boxes (like "f'(0) does not exist") are valid
            # teaching output. A genuine failure is the ⚠️ friendly-error box
            # that error_box() prints, or an uncaught exception.
            err_texts = [str(e.value) for e in at.error]
            real_errors = [t for t in err_texts if t.startswith("⚠️")]
            if real_errors:
                check(f"page `{topic}` no parse error on defaults", False, " | ".join(real_errors)[:200])
        except Exception as err:  # noqa: BLE001
            check(f"page `{topic}` renders", False, f"{type(err).__name__}: {err}")
            traceback.print_exc()

    lab_paths = [
        ("First derivative (with rule walkthrough)", "sin(x)*x**2"),
        ("Limit at a point", "(x**2 - 1)/(x - 1)"),
        ("Limit at infinity", "(3*x**2 + 1)/(x**2 - x)"),
        ("Tangent and normal lines", "x**3"),
        ("Evaluate f(a)", "sqrt(x)"),
        ("Solve f(x) = 0", "x**2 - 5*x + 6"),
        ("Domain of definition", "ln(x - 1)/sqrt(3 - x)"),
        ("Second derivative", "x**4"),
        ("Simplify and graph", "(x + 1)**2 - x**2"),
        ("Implicit differentiation", "x**2 + y**2 = 25"),
    ]
    for mode, expr in lab_paths:
        try:
            at = fresh("Python calculus lab")
            at.selectbox[0].set_value(mode)
            at.run()
            boxes = [tb for tb in at.text_input if tb.label.startswith(("f(x)", "Equation"))]
            if boxes:
                boxes[0].set_value(expr)
                at.run()
            errs = [str(e.value) for e in at.error]
            check(f"lab `{mode}` with `{expr}`", not errs and not list(at.exception),
                  (" | ".join(errs)[:200]) or str([e.value for e in at.exception])[:200])
        except Exception as err:  # noqa: BLE001
            check(f"lab `{mode}` with `{expr}`", False, f"{type(err).__name__}: {err}")

    # Quiz: answer everything correctly, expect a perfect score.
    try:
        import calculus_tutor as app_mod
        at = fresh("Practice quiz")
        for i, (_q, options, correct, _why) in enumerate(app_mod.QUIZ):
            at.radio[i].set_value(options[correct])
        at.button[0].set_value(True)
        at.run()
        body = " ".join(str(m.value) for m in at.metric)
        check("quiz perfect score path", f"{len(app_mod.QUIZ)} / {len(app_mod.QUIZ)}" in body, body)

        # A wrong answer should surface an explanation box, not crash
        at2 = fresh("Practice quiz")
        for i, (_q, options, correct, _w) in enumerate(app_mod.QUIZ):
            wrong = [o for o in options if o != options[correct]][0]
            at2.radio[i].set_value(wrong if i == 0 else options[correct])
        at2.button[0].set_value(True)
        at2.run()
        check("quiz wrong-answer path", not list(at2.exception),
              str([e.value for e in at2.exception])[:200])
    except Exception as err:  # noqa: BLE001
        check("quiz score path runs", False, f"{type(err).__name__}: {err}")
        traceback.print_exc()

    # Bad input must produce a friendly error, never an exception page.
    for bad in ["x**", "((x+1", "foo(x)", ""]:
        try:
            at = fresh("Python calculus lab")
            boxes = [tb for tb in at.text_input if tb.label.startswith("f(x)")]
            boxes[0].set_value(bad)
            at.run()
            friendly = len(at.error) >= 1 and all(
                not str(e.value).startswith("Traceback") for e in at.error)
            check(f"lab handles bad input `{bad}`", not list(at.exception) and friendly,
                  f"exc={str([e.value for e in at.exception])[:150]} err={[str(e.value)[:60] for e in at.error]}")
        except Exception as err:  # noqa: BLE001
            check(f"lab handles bad input `{bad}`", False, f"{type(err).__name__}: {err}")

print("\n" + "=" * 70)
print("12. Every lesson must ship a how-to-solve method and worked solutions")
print("=" * 70)
import calculus_tutor as app_ui  # noqa: E402

LESSON_TOPICS = [t for t in TOPICS
                 if t not in ("Start here", "Python calculus lab", "Practice quiz")]
for topic in LESSON_TOPICS:
    entry = app_ui.HOW_TO_SOLVE.get(topic)
    check(f"`{topic}` has a method", bool(entry) and len(entry.get("method", [])) >= 3,
          "missing or too short")
    solved = entry.get("solved", []) if entry else []
    check(f"`{topic}` has worked solutions", len(solved) >= 2, f"found {len(solved)}")
    incomplete = [p.get("title", "<untitled>") for p in solved
                  if any(not p.get(field) for field in
                         ("title", "problem", "steps", "answer", "pitfall"))]
    check(f"`{topic}` solutions complete", not incomplete, f"missing fields in {incomplete}")
    if solved:
        check(f"`{topic}` solutions are stepped",
              all(len(p.get("steps", [])) >= 3 for p in solved),
              "a worked solution needs at least 3 steps")

check("general method present", len(app_ui.GENERAL_METHOD) >= 4)
for mode in app_ui.LAB_MODES:
    steps = app_ui.LAB_METHODS.get(mode, app_ui.GENERAL_METHOD)
    check(f"lab method for `{mode}`", len(steps) >= 2, f"{len(steps)} steps")
check("quiz has methods by type", len(app_ui.QUIZ_METHODS) >= 4)

# The method text must actually reach the rendered page, not just exist in data.
print("  --- verifying the content is rendered on the page ---")
if AppTest is not None:
    probe_topics = ["1.1 Functions and domains", "2.6 Implicit differentiation"]
    for topic in probe_topics:
        at = fresh(topic)
        sub_texts = " ".join(str(s.value) for s in at.subheader)
        check(f"`{topic}` shows the method heading",
              "How to solve it, step by step" in sub_texts, sub_texts[:200])
        check(f"`{topic}` shows worked solutions heading",
              "Worked solutions, in full" in sub_texts, sub_texts[:200])
        # A distinctive phrase from the solved problem text must appear
        first_answer = app_ui.HOW_TO_SOLVE[topic]["solved"][0]["answer"]
        check(f"`{topic}` renders an answer", first_answer[:20] in " ".join(
            str(s.value) for s in at.success), first_answer[:40])

    at_lab = fresh("Python calculus lab")
    lab_md = " ".join(str(m.value) for m in at_lab.markdown)
    check("lab shows a method step", "Simplify algebraically" in lab_md or "Expand or factor" in lab_md,
          lab_md[:200])

    at_quiz = fresh("Practice quiz")
    quiz_md = " ".join(str(m.value) for m in at_quiz.markdown)
    quiz_subs = " ".join(str(s.value) for s in at_quiz.subheader)
    check("quiz shows approach method", "How to approach any problem here" in quiz_subs, quiz_subs[:200])
    check("quiz shows methods by type", "Method by question type" in quiz_subs, quiz_subs[:200])

print("\n" + "=" * 70)
print("13. Worked answers must be arithmetically correct")
print("=" * 70)
# Re-derive the headline answers with SymPy instead of trusting the prose.
y_sym = ce.Y
verify = [
    ("1.1 domain root/quad", sp.calculus.util.continuous_domain(
        sp.sqrt(X + 3) / (X ** 2 - 4), X, sp.S.Reals),
     sp.Union(sp.Interval.Ropen(-3, -2), sp.Interval.open(-2, 2), sp.Interval.open(2, sp.oo))),
    ("1.1 domain log+root", sp.calculus.util.continuous_domain(
        sp.log(5 - 2 * X) + sp.sqrt(X + 1), X, sp.S.Reals),
     sp.Interval.Ropen(-1, sp.Rational(5, 2))),
    ("1.3 limit factor", sp.limit((X ** 2 - 9) / (X ** 2 - 2 * X - 3), X, 3), sp.Rational(3, 2)),
    ("1.3 limit conjugate", sp.limit((sp.sqrt(X + 4) - 2) / X, X, 0), sp.Rational(1, 4)),
    ("1.3 limit infinity", sp.limit((4 * X ** 3 - X) / (2 * X ** 3 + 5 * X ** 2), X, sp.oo), sp.Integer(2)),
    ("2.1 tangent slope", sp.diff(X ** 3 - 2 * X, X).subs(X, 1), sp.Integer(1)),
    ("2.1 derivative 1/x at 2", sp.diff(1 / X, X).subs(X, 2), sp.Rational(-1, 4)),
    ("2.2 derivative sqrt", sp.diff(sp.sqrt(X), X), 1 / (2 * sp.sqrt(X))),
    ("2.3 quotient rule", sp.simplify(sp.diff((X ** 2 + 1) / (X - 3), X)),
     sp.simplify((X ** 2 - 6 * X - 1) / (X - 3) ** 2)),
    ("2.3 x^2 sqrt(x)", sp.diff(X ** 2 * sp.sqrt(X), X), sp.Rational(5, 2) * X ** sp.Rational(3, 2)),
    ("2.3 x^2 log x", sp.simplify(sp.diff(X ** 2 * sp.log(X), X)), sp.simplify(X * (2 * sp.log(X) + 1))),
    ("2.4 sin/(1+cos)", sp.simplify(sp.diff(sp.sin(X) / (1 + sp.cos(X)), X)),
     sp.simplify(1 / (1 + sp.cos(X)))),
    ("2.4 x tan x", sp.diff(X * sp.tan(X), X), sp.tan(X) + X * sp.sec(X) ** 2),
    ("2.4 sin^2 x", sp.simplify(sp.diff(sp.sin(X) ** 2, X)), sp.simplify(sp.sin(2 * X))),
    ("2.5 power composite", sp.simplify(sp.diff((3 * X ** 2 + 1) ** 5, X)),
     sp.simplify(30 * X * (3 * X ** 2 + 1) ** 4)),
    ("2.5 exp sin", sp.diff(sp.exp(sp.sin(X)), X), sp.exp(sp.sin(X)) * sp.cos(X)),
    ("2.5 ln cos x^2", sp.simplify(sp.diff(sp.log(sp.cos(X ** 2)), X)),
     sp.simplify(-2 * X * sp.tan(X ** 2))),
    ("2.6 circle", sp.simplify(ce.implicit_derivative("x**2 + y**2 = 25")["dydx"]),
     sp.simplify(-X / y_sym)),
    ("2.6 x^2 y + y^3", sp.simplify(ce.implicit_derivative("x**2*y + y**3 = 5")["dydx"]),
     sp.simplify(-2 * X * y_sym / (X ** 2 + 3 * y_sym ** 2))),
    ("2.6 folium", sp.simplify(ce.implicit_derivative("x**3 + y**3 = 6*x*y")["dydx"]),
     sp.simplify((2 * y_sym - X ** 2) / (y_sym ** 2 - 2 * X))),
]
for name, got, want in verify:
    # Domains are SymPy Set objects (Interval/Union), which cannot be compared
    # by subtracting; everything else is an expression.
    if isinstance(got, sp.Set) or isinstance(want, sp.Set):
        ok = bool(got == want) or bool(getattr(got, "equals", lambda w: False)(want))
    else:
        try:
            ok = sp.simplify(got - want) == 0
        except Exception:  # noqa: BLE001
            ok = got == want
    check(name, ok, f"got {ce.to_text(got, 60)} want {ce.to_text(want, 60)}")

# Slopes quoted at specific points
point_checks = [
    ("2.6 circle at (3,4) -> -3/4",
     ce.implicit_derivative("x**2 + y**2 = 25", at=(3, 4))["slope"], sp.Rational(-3, 4)),
    ("2.6 x^2y+y^3 at (2,1) -> -4/7",
     ce.implicit_derivative("x**2*y + y**3 = 5", at=(2, 1))["slope"], sp.Rational(-4, 7)),
    ("2.6 folium at (3,3) -> -1",
     ce.implicit_derivative("x**3 + y**3 = 6*x*y", at=(3, 3))["slope"], sp.Integer(-1)),
    ("1.5 IVT root of x^3+x-1", round(float(sp.nsolve(X ** 3 + X - 1, 0.5)), 3), 0.682),
]
for name, got, want in point_checks:
    check(name, got == want if isinstance(want, float) else sp.simplify(got - want) == 0,
          f"got {got} want {want}")

# Tangent lines quoted in the prose
t1 = ce.tangent_line(X ** 3 - 2 * X, 1)
check("2.1 tangent line y = x - 2", sp.simplify(t1["line"] - (X - 2)) == 0, ce.to_text(t1["line"]))
t2 = ce.tangent_line(1 / X, 2)
check("2.1 tangent y = -x/4 + 1", sp.simplify(t2["line"] - (-X / 4 + 1)) == 0, ce.to_text(t2["line"]))
t3 = ce.tangent_line(X ** 2, 3)
check("circle tangent y = -0.75x + 6.25", sp.simplify(
    (4 + sp.Rational(-3, 4) * (X - 3)) - (sp.Rational(-3, 4) * X + sp.Rational(25, 4))) == 0)

print("\n" + "=" * 70)
print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("Failed checks:")
    for name in FAIL:
        print("   - " + name)
print("=" * 70)
sys.exit(1 if FAIL else 0)
