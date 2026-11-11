"""
Calculus Python Tutor — interactive Streamlit app.

Covers Chapters 1–2 of the uploaded calculus notes: functions and domains,
transformations and composition, limits and limit laws, epsilon-delta and
continuity, rates of change, the derivative as a function, differentiation
rules, trigonometric derivatives, the chain rule, implicit differentiation.

Run (from this folder):
    python -m pip install -r requirements.txt
    python -m streamlit run calculus_tutor.py
"""

from __future__ import annotations

import streamlit as st
import sympy as sp

import calculus_engine as ce

st.set_page_config(
    page_title="Calculus Python Tutor",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

TOPICS = [
    "Start here",
    "1.1 Functions and domains",
    "1.2 Transformations and composition",
    "1.3 Limits and limit laws",
    "1.4 Epsilon-delta definition",
    "1.5 Continuity and IVT",
    "2.1 Rates of change and tangents",
    "2.2 The derivative as a function",
    "2.3 Differentiation rules",
    "2.4 Trigonometric derivatives",
    "2.5 The chain rule",
    "2.6 Implicit differentiation",
    "Python calculus lab",
    "Practice quiz",
]

X = ce.X
Y = ce.Y


# --------------------------------------------------------------------------
# Shared UI helpers
# --------------------------------------------------------------------------
def latex(expr: str | sp.Expr) -> None:
    """Render LaTeX without letting a bad expression kill the page."""
    if isinstance(expr, sp.Expr):
        expr = ce.to_latex(expr)
    try:
        st.latex(expr)
    except Exception:  # noqa: BLE001
        st.code(str(expr), language="tex")


def formula(inline: str) -> None:
    try:
        st.markdown(f"$ {inline} $")
    except Exception:  # noqa: BLE001
        st.write(inline)


def show(fig) -> None:
    st.pyplot(fig)
    ce.show_figure(fig)


def rule_list(rules: list[str]) -> None:
    st.subheader("Key rules")
    for rule in rules:
        st.markdown(f"- {rule}")


def error_box(err: Exception) -> None:
    friendly = getattr(err, "friendly", str(err))
    st.error(f"⚠️ {friendly}")
    technical = getattr(err, "technical", "")
    if technical:
        with st.expander("Technical detail (for debugging)"):
            st.code(technical)


def lesson_header(title: str, bigidea: str, formula_tex: str = "") -> None:
    st.header(title)
    st.markdown(f"**Big idea.** {bigidea}")
    if formula_tex:
        latex(formula_tex)


def try_expr(raw: str, *, allow_y: bool = True) -> sp.Expr:
    return ce.parse_expression(raw, allow_y=allow_y)


def render_method(steps: list[str], heading: str = "How to solve it, step by step") -> None:
    st.subheader(heading)
    for i, step in enumerate(steps, 1):
        st.markdown(f"{i}. {step}")


def render_solved(problems: list[dict]) -> None:
    st.subheader("Worked solutions, in full")
    for problem in problems:
        with st.container(border=True):
            st.markdown(f"**{problem['title']}**")
            st.markdown(f"*Problem:* {problem['problem']}")
            st.markdown("**Solution**")
            for i, step in enumerate(problem["steps"], 1):
                st.markdown(f"{i}. {step}")
            st.success(f"Answer: {problem['answer']}")
            st.caption(f"Common mistake — {problem['pitfall']}")


# --------------------------------------------------------------------------
# Lesson content
# --------------------------------------------------------------------------
LESSONS: dict[str, dict] = {
    "1.1 Functions and domains": {
        "bigidea": "A function assigns exactly one output to each allowed input. The domain is "
                   "the set of inputs the rule is actually allowed to receive.",
        "formula": r"\text{domain}(f)=\{\,x\in\mathbb{R}: f(x)\ \text{is defined}\,\}",
        "rules": [
            "A denominator may not equal zero.",
            "The expression inside an even root must be ≥ 0.",
            "The argument of a logarithm must be > 0.",
            "Inverse trig functions are limited: arcsin and arccos need −1 ≤ x ≤ 1.",
            "Vertical Line Test: if any vertical line crosses a graph more than once, "
            "that graph is **not** a function.",
        ],
        "worked": ("For f(x) = √(x − 2) / (x − 5): the root needs x − 2 ≥ 0, so x ≥ 2; "
                   "the denominator needs x − 5 ≠ 0, so x ≠ 5. Domain: [2, 5) ∪ (5, ∞)."),
    },
    "1.2 Transformations and composition": {
        "bigidea": "A transformation moves, flips or stretches a known graph. Composition puts one "
                   "function inside another — and the order matters.",
        "formula": r"(f\circ g)(x)=f\big(g(x)\big)\ \neq\ g\big(f(x)\big)\ \text{in general}",
        "rules": [
            "f(x) + c moves the graph **up** c; f(x) − c moves it down c.",
            "f(x − c) moves the graph **right** c; f(x + c) moves it left c. "
            "The horizontal direction looks backwards — that is normal.",
            "−f(x) reflects across the x-axis; f(−x) reflects across the y-axis.",
            "f(kx) compresses horizontally by k; f(x/k) stretches horizontally by k.",
            "A function has an inverse only when it is one-to-one (passes the Horizontal Line Test).",
        ],
        "worked": ("With f(x) = x² and g(x) = x + 1: f(g(x)) = (x + 1)², but g(f(x)) = x² + 1. "
                   "Same two letters, different answers — so order always matters."),
    },
    "1.3 Limits and limit laws": {
        "bigidea": "A limit asks what value f(x) is *approaching* as x closes in on a point. "
                   "What happens exactly at the point can be different, or undefined.",
        "formula": r"\lim_{x\to a} f(x)=L \iff f(x)\ \text{gets arbitrarily close to } L\ \text{as } x\to a",
        "rules": [
            "Polynomials: substitute directly.",
            "Sum, difference, constant multiple, product, power and root laws let you split a hard "
            "limit into easy ones.",
            "Quotient law needs a non-zero denominator limit.",
            "0/0 means *indeterminate*, not *no answer*: factor, cancel, multiply by a conjugate, "
            "or combine fractions and try again.",
            "Squeeze Theorem: if g(x) ≤ f(x) ≤ h(x) and g, h share a limit, so does f.",
            "Radian limits: lim(x→0) sin x / x = 1 and lim(x→0) (1 − cos x)/x = 0.",
            "At infinity for rational functions, compare the leading powers of x.",
        ],
        "worked": ("lim(x→1) (x² − 1)/(x − 1): substituting gives 0/0. Factor to get "
                   "(x − 1)(x + 1)/(x − 1), cancel the common factor, then substitute: the limit is 2. "
                   "Note f(1) itself is undefined — the limit still exists."),
    },
    "1.4 Epsilon-delta definition": {
        "bigidea": "The formal meaning of “gets arbitrarily close”: whatever tolerance ε you demand "
                   "around L, we can find a window δ around a that keeps f(x) inside it.",
        "formula": r"\forall \varepsilon>0\ \exists\,\delta>0:\ 0<|x-a|<\delta \Longrightarrow |f(x)-L|<\varepsilon",
        "rules": [
            "ε (epsilon) is the allowed error in the **output**; δ (delta) is how close x must stay.",
            "The condition 0 < |x − a| deliberately excludes x = a itself — that is why a hole in the "
            "graph can still have a limit.",
            "For linear f(x) = mx + b, work out |f(x) − L| = |m|·|x − a| and choose δ = ε/|m|.",
            "To *disprove* a limit, show one ε for which no δ works.",
        ],
        "worked": ("Prove lim(x→3) (2x + 1) = 7. Compute |(2x + 1) − 7| = |2x − 6| = 2|x − 3|. "
                   "We want this < ε, so require |x − 3| < ε/2. Choosing δ = ε/2 finishes the proof."),
    },
    "1.5 Continuity and IVT": {
        "bigidea": "Continuous means no hole, jump or break: the limit exists, the function value "
                   "exists, and they are equal.",
        "formula": r"f\ \text{continuous at } a \iff \lim_{x\to a} f(x)=f(a)",
        "rules": [
            "Three checks at a: f(a) defined, lim(x→a) f(x) exists, and the two agree.",
            "Continuity from the right uses lim(x→a⁺); from the left uses lim(x→a⁻).",
            "Polynomials are continuous everywhere; rational functions are continuous wherever the "
            "denominator is non-zero; sqrt is continuous on its domain.",
            "Removable discontinuity: a single missing point (a hole). Jump discontinuity: the two "
            "sided limits differ. Infinite discontinuity: a vertical asymptote.",
            "Intermediate Value Theorem: if f is continuous on [a, b] and k lies between f(a) and f(b), "
            "then some c in (a, b) has f(c) = k. This is how we prove roots exist.",
            "Extreme Value Theorem: a continuous function on a closed interval [a, b] must attain both "
            "an absolute maximum and an absolute minimum.",
        ],
        "worked": ("Show x³ − 3x + 1 has a root between 0 and 1. f(0) = 1 > 0 and f(1) = −1 < 0. "
                   "f is a polynomial, hence continuous, and 0 lies between 1 and −1, so IVT guarantees "
                   "some c ∈ (0, 1) with f(c) = 0."),
    },
    "2.1 Rates of change and tangents": {
        "bigidea": "Average rate of change is the slope of a secant line between two points. Let the "
                   "gap shrink to zero and you get the tangent line — the instantaneous rate.",
        "formula": r"f'(a)=\lim_{h\to 0}\frac{f(a+h)-f(a)}{h}=\lim_{x\to a}\frac{f(x)-f(a)}{x-a}",
        "rules": [
            "Average rate of change on [a, b]: [f(b) − f(a)] / (b − a).",
            "Tangent line at x = a: y − f(a) = f′(a)(x − a).",
            "Normal line: perpendicular to the tangent, slope −1/f′(a).",
            "Motion: position s(t), velocity v(t) = s′(t), acceleration a(t) = s″(t).",
            "Units: the derivative's units are (output units) ÷ (input units).",
        ],
        "worked": ("For f(x) = x² at a = 3: f′(x) = 2x, so the slope is 6 and f(3) = 9. "
                   "Tangent line: y − 9 = 6(x − 3), i.e. y = 6x − 9."),
    },
    "2.2 The derivative as a function": {
        "bigidea": "Differentiating produces a new function f′ that reports the slope of f at every x "
                   "where the slope exists.",
        "formula": r"f'(x)=\lim_{h\to 0}\frac{f(x+h)-f(x)}{h}",
        "rules": [
            "f′(x) > 0 means f is increasing there; f′(x) < 0 means decreasing.",
            "f″(x) > 0 means concave up (∪); f″(x) < 0 means concave down (∩).",
            "Differentiability implies continuity — but continuity does **not** imply differentiability.",
            "f′ fails to exist at: a corner or cusp (|x| at 0), a vertical tangent (x^(1/3) at 0), "
            "or any discontinuity.",
            "Higher derivatives: f′, f″, f‴, f⁗, then dⁿf/dxⁿ.",
            "Estimating from a graph: read the slope of the curve at the point of interest.",
        ],
        "worked": ("For f(x) = x⁴ − 3x² + 2: f′(x) = 4x³ − 6x and f″(x) = 12x² − 6. "
                   "f″ changes sign, so this curve switches concavity — visible in the graph below."),
    },
    "2.3 Differentiation rules": {
        "bigidea": "Rules replace the limit definition for routine work. Learn which rule applies by "
                   "recognising the *structure* of the expression.",
        "formula": r"(fg)'=f'g+fg' \qquad \left(\frac{f}{g}\right)'=\frac{f'g-fg'}{g^{2}}",
        "rules": [
            "Constant: d(c)/dx = 0. Power: d(xⁿ)/dx = n·xⁿ⁻¹ (works for negative and fractional n).",
            "Constant multiple: d(c·f)/dx = c·f′.",
            "Sum/difference: (f ± g)′ = f′ ± g′.",
            "Product: (fg)′ = f′g + fg′.",
            "Quotient: (f/g)′ = (f′g − fg′)/g² — mind the order, the numerator is not symmetric.",
            "Exponential and log: (eˣ)′ = eˣ, (aˣ)′ = aˣ ln a, (ln x)′ = 1/x.",
        ],
        "worked": ("For f(x) = x² ln x, use the product rule: f′ = 2x·ln x + x²·(1/x) = 2x ln x + x "
                   "= x(2 ln x + 1)."),
    },
    "2.4 Trigonometric derivatives": {
        "bigidea": "Trig derivatives describe waves, rotation and oscillation. Calculus formulas assume "
                   "**radians** — degrees introduce a factor of π/180 and break them.",
        "formula": r"\frac{d}{dx}\sin x=\cos x,\quad \frac{d}{dx}\cos x=-\sin x,\quad \frac{d}{dx}\tan x=\sec^{2}x",
        "rules": [
            "(sin x)′ = cos x and (cos x)′ = −sin x — note the minus appears only on cosine.",
            "(tan x)′ = sec²x, (sec x)′ = sec x tan x, (csc x)′ = −csc x cot x, (cot x)′ = −csc²x.",
            "A handy pattern: every cofunction (cos, csc, cot) derivative carries a minus sign.",
            "Inverse trig: (arcsin x)′ = 1/√(1 − x²), (arctan x)′ = 1/(1 + x²), "
            "(arcsec x)′ = 1/(|x|√(x² − 1)).",
            "These come from the radian limit sin(x)/x → 1, so radians are not a convention here — "
            "they are a requirement.",
        ],
        "worked": ("For f(x) = x sin x, apply the product rule: f′ = 1·sin x + x·cos x = "
                   "sin x + x cos x."),
    },
    "2.5 The chain rule": {
        "bigidea": "For a function inside a function, differentiate the outer layer, keep the inner "
                   "expression untouched, then multiply by the inner derivative.",
        "formula": r"\frac{dy}{dx}=\frac{dy}{du}\cdot\frac{du}{dx}\qquad\text{or}\qquad (f\circ g)'(x)=f'(g(x))\,g'(x)",
        "rules": [
            "Ask: what is the *outermost* operation? That determines the first rule to apply.",
            "sin(3x²) → cos(3x²) · 6x = 6x cos(3x²).",
            "e^(x²) → e^(x²) · 2x = 2x e^(x²).",
            "ln(x² + 1) → (2x)/(x² + 1).",
            "(5x − 1)¹⁰ → 10(5x − 1)⁹ · 5 = 50(5x − 1)⁹.",
            "Most common error: forgetting to multiply by the inner derivative. Nesting can require the "
            "chain rule more than once.",
        ],
        "worked": ("For y = sin(3x): outer derivative cos(3x), inner derivative 3, so y′ = 3 cos(3x). "
                   "Check at x = 0: y′(0) = 3, which matches the steep rise of sin(3x) through the origin."),
    },
    "2.6 Implicit differentiation": {
        "bigidea": "When y is not isolated (x² + y² = 25, for instance), differentiate both sides with "
                   "respect to x, treating y as a hidden function of x.",
        "formula": r"\frac{d}{dx}F(x,y)=\frac{d}{dx}G(x,y)\ \Longrightarrow\ \frac{dy}{dx}=\text{solve for }y'",
        "rules": [
            "Differentiate every term with respect to x on both sides.",
            "Any term involving y gets an extra factor dy/dx, because y = y(x): d(y²)/dx = 2y·y′.",
            "Whenever a product mixes x and y (such as xy), use the product rule *and* the chain rule.",
            "Collect all y′ terms on one side, factor y′ out, then divide.",
            "To find a slope at a specific point, substitute the coordinates **after** solving for y′.",
        ],
        "worked": ("For x² + y² = 25: 2x + 2y·y′ = 0 → y′ = −x/y. At (3, 4) the slope is −3/4, so the "
                   "tangent line is y − 4 = −(3/4)(x − 3), i.e. y = −0.75x + 6.25."),
    },
}


# --------------------------------------------------------------------------
# How to solve: general procedure + fully worked problems, per topic
#
# Every step is written out because the point of a worked solution is the
# method, not just the final number. Unicode maths (x², √, ≥) is used instead
# of $...$ so the text survives markdown parsing everywhere.
# --------------------------------------------------------------------------
HOW_TO_SOLVE: dict[str, dict] = {
    "1.1 Functions and domains": {
        "method": [
            "List every operation the rule performs on x. Each restricted operation contributes a condition.",
            "Every even root (√, fourth root): set the inside ≥ 0 and solve the inequality.",
            "Every denominator: set it ≠ 0 and solve. A quotient can also be written as a negative power.",
            "Every logarithm: set the argument > 0 (strictly greater, never equal).",
            "arcsin and arccos: require −1 ≤ input ≤ 1.",
            "Intersect all the conditions — the domain must satisfy every one of them at once.",
            "Convert to interval notation and check each endpoint: √ allows the endpoint, ln and denominators do not.",
        ],
        "solved": [
            {
                "title": "Root over a quadratic denominator",
                "problem": "Find the domain of f(x) = √(x + 3) / (x² − 4).",
                "steps": [
                    "Two restricted operations: a square root and a denominator.",
                    "Root condition: x + 3 ≥ 0, so x ≥ −3.",
                    "Denominator condition: x² − 4 ≠ 0 → (x − 2)(x + 2) ≠ 0, so x ≠ 2 and x ≠ −2.",
                    "Intersect: start from x ≥ −3, then remove −2 and 2.",
                    "Both −2 and 2 are ≥ −3, so both holes are real and must be cut out.",
                ],
                "answer": "[−3, −2) ∪ (−2, 2) ∪ (2, ∞)",
                "pitfall": "Writing x ≥ −3 and forgetting the two holes. A square bracket at −3 is correct "
                           "because √0 = 0 is defined; parentheses at ±2 are correct because division by zero is not.",
            },
            {
                "title": "Log plus root together",
                "problem": "Find the domain of f(x) = ln(5 − 2x) + √(x + 1).",
                "steps": [
                    "Log condition: 5 − 2x > 0 → −2x > −5 → x < 5/2 (the inequality flips when dividing by −2).",
                    "Root condition: x + 1 ≥ 0 → x ≥ −1.",
                    "Intersect the two: x must satisfy both, so −1 ≤ x < 5/2.",
                ],
                "answer": "[−1, 5/2)",
                "pitfall": "Using ≥ for the logarithm. ln(0) is undefined, so the right endpoint stays open.",
            },
        ],
    },
    "1.2 Transformations and composition": {
        "method": [
            "Rewrite the function in the standard form a·f(b(x − h)) + k.",
            "Read the horizontal changes from inside the brackets first: h shifts right, b compresses by a factor of b, "
            "and a negative b reflects across the y-axis.",
            "Read the vertical changes from outside: a stretches vertically (negative a reflects across the x-axis), "
            "k shifts up.",
            "Apply horizontal changes before vertical ones, then track one anchor point (a vertex or intercept) to confirm.",
            "For composition f(g(x)), work inside-out: evaluate g first and feed its output into f.",
            "The domain of f(g(x)) is restricted by g's domain, then further restricted by any x that g maps outside f's domain.",
        ],
        "solved": [
            {
                "title": "Describe a transformation",
                "problem": "Describe how y = −2(x − 3)² + 5 comes from y = x².",
                "steps": [
                    "Match to a·f(b(x − h)) + k with f(x) = x², a = −2, b = 1, h = 3, k = 5.",
                    "h = 3 → shift right 3.",
                    "a = −2 → vertical stretch by 2, and the minus sign reflects the graph across the x-axis (opens downward).",
                    "k = 5 → shift up 5.",
                    "Check with the anchor point: the vertex (0, 0) of x² moves to (3, 5). "
                    "Substituting x = 3 gives y = −2·0 + 5 = 5. ✓",
                ],
                "answer": "Shift right 3, stretch vertically by 2 and reflect over the x-axis, then shift up 5; vertex (3, 5).",
                "pitfall": "Reading (x − 3) as a leftward shift. Inside the brackets, the sign is the opposite of what you expect.",
            },
            {
                "title": "Composition and its domain",
                "problem": "For f(x) = x² − 1 and g(x) = √x, find f(g(x)) and g(f(x)) with their domains.",
                "steps": [
                    "f(g(x)) means apply g first: f(√x) = (√x)² − 1 = x − 1.",
                    "But √x only accepts x ≥ 0, so the domain of f∘g is x ≥ 0, not all reals.",
                    "g(f(x)) means apply f first: g(x² − 1) = √(x² − 1).",
                    "Now the outer √ restricts us: x² − 1 ≥ 0 → x ≤ −1 or x ≥ 1.",
                    "Notice the two compositions give different formulas AND different domains.",
                ],
                "answer": "f(g(x)) = x − 1 with domain [0, ∞);  g(f(x)) = √(x² − 1) with domain (−∞, −1] ∪ [1, ∞).",
                "pitfall": "Cancelling (√x)² to x and then reporting the domain as all real numbers. "
                           "The inner function's domain always survives into the composition.",
            },
        ],
    },
    "1.3 Limits and limit laws": {
        "method": [
            "Always start by substituting. If you get a finite number, that is the answer — stop.",
            "If substitution gives 0/0, that is indeterminate, not 'no answer'. Simplify: factor and cancel, "
            "multiply by the conjugate, combine fractions, or use a trig identity.",
            "If substitution gives (non-zero)/0, the limit is infinite. Check the sign of the denominator from each side.",
            "Compute the left-hand and right-hand limits separately. If they differ, the two-sided limit does not exist.",
            "At infinity for a rational function, divide numerator and denominator by the highest power of x in the denominator.",
            "For a bounded function times something going to 0, use the Squeeze Theorem.",
            "After simplifying, substitute again — the simplified expression agrees with the original everywhere except "
            "possibly at the point itself, which is exactly what a limit ignores.",
        ],
        "solved": [
            {
                "title": "0/0 by factoring",
                "problem": "Evaluate lim(x→3) (x² − 9)/(x² − 2x − 3).",
                "steps": [
                    "Substitute x = 3: (9 − 9)/(9 − 6 − 3) = 0/0. Indeterminate, so simplify.",
                    "Factor numerator: x² − 9 = (x − 3)(x + 3).",
                    "Factor denominator: x² − 2x − 3 = (x − 3)(x + 1).",
                    "Cancel the common factor (x − 3), which is legal because x approaches 3 but never equals it: "
                    "the expression becomes (x + 3)/(x + 1).",
                    "Substitute x = 3: (3 + 3)/(3 + 1) = 6/4 = 3/2.",
                ],
                "answer": "3/2",
                "pitfall": "Stopping at 0/0 and declaring the limit does not exist. 0/0 means 'more work needed'.",
            },
            {
                "title": "0/0 by conjugate",
                "problem": "Evaluate lim(x→0) (√(x + 4) − 2)/x.",
                "steps": [
                    "Substitute x = 0: (√4 − 2)/0 = 0/0. Factoring will not help a radical — rationalise instead.",
                    "Multiply top and bottom by the conjugate √(x + 4) + 2.",
                    "Numerator becomes (√(x+4))² − 2² = (x + 4) − 4 = x.",
                    "So the fraction is x / [x(√(x + 4) + 2)].",
                    "Cancel x: 1/(√(x + 4) + 2).",
                    "Substitute x = 0: 1/(2 + 2) = 1/4.",
                ],
                "answer": "1/4",
                "pitfall": "Multiplying only the numerator by the conjugate. Whatever you multiply by must go on both "
                           "top and bottom, or you have changed the expression.",
            },
            {
                "title": "Limit at infinity",
                "problem": "Evaluate lim(x→∞) (4x³ − x)/(2x³ + 5x²).",
                "steps": [
                    "Highest power in the denominator is x³. Divide every term by x³.",
                    "Numerator: 4 − 1/x². Denominator: 2 + 5/x.",
                    "As x → ∞, both 1/x² and 5/x → 0.",
                    "Substitute: (4 − 0)/(2 + 0) = 2.",
                ],
                "answer": "2",
                "pitfall": "Assuming every limit at infinity is 0. Equal leading powers give the ratio of leading "
                           "coefficients; only a higher power in the denominator gives 0.",
            },
        ],
    },
    "1.4 Epsilon-delta definition": {
        "method": [
            "Write down the target: make |f(x) − L| smaller than ε.",
            "Substitute f(x) and L, then simplify until the expression contains a factor of |x − a|.",
            "Solve the inequality for |x − a| to discover what δ has to be.",
            "If the expression is not linear, first restrict δ ≤ 1 to bound the other factor, "
            "then set δ = min(1, ε/bound).",
            "Write the proof in the standard order: 'Let ε > 0. Choose δ = … . Suppose 0 < |x − a| < δ. Then … < ε.'",
            "To disprove a limit, exhibit one specific ε for which no δ can work (usually at a jump).",
        ],
        "solved": [
            {
                "title": "Linear function",
                "problem": "Prove that lim(x→2) (3x − 1) = 5.",
                "steps": [
                    "Target: |(3x − 1) − 5| < ε.",
                    "Simplify: |3x − 6| = 3|x − 2|.",
                    "We want 3|x − 2| < ε, so |x − 2| < ε/3.",
                    "That tells us the choice: δ = ε/3.",
                    "Proof: Let ε > 0 and choose δ = ε/3. If 0 < |x − 2| < δ, then "
                    "|(3x − 1) − 5| = 3|x − 2| < 3δ = 3(ε/3) = ε. ∎",
                ],
                "answer": "δ = ε/3 works, so the limit is proved to be 5.",
                "pitfall": "Presenting the scratch work as the proof. The δ you discover comes from working backwards; "
                           "the written proof must run forwards from 'let ε > 0'.",
            },
            {
                "title": "Quadratic — needs a bound",
                "problem": "Prove that lim(x→1) x² = 1.",
                "steps": [
                    "Target: |x² − 1| < ε.",
                    "Factor: |x² − 1| = |x − 1|·|x + 1|.",
                    "The |x + 1| factor varies, so bound it: temporarily require δ ≤ 1.",
                    "Then |x − 1| < 1 gives 0 < x < 2, so |x + 1| < 3.",
                    "Now |x² − 1| < 3|x − 1|, which is below ε whenever |x − 1| < ε/3.",
                    "Choose δ = min(1, ε/3).",
                    "Proof: Let ε > 0, choose δ = min(1, ε/3). If 0 < |x − 1| < δ, then |x + 1| < 3 and "
                    "|x² − 1| = |x − 1||x + 1| < (ε/3)·3 = ε. ∎",
                ],
                "answer": "δ = min(1, ε/3) works.",
                "pitfall": "Choosing δ = ε/3 alone. Without the δ ≤ 1 restriction you have not bounded |x + 1|, "
                           "and the chain of inequalities breaks.",
            },
        ],
    },
    "1.5 Continuity and IVT": {
        "method": [
            "To classify a discontinuity at a: first compute f(a). Undefined means discontinuous — then ask which type.",
            "Compute the left-hand limit lim(x→a⁻) and the right-hand limit lim(x→a⁺).",
            "Left ≠ right → jump discontinuity. Left = right but ≠ f(a) (or f(a) undefined) → removable (a hole). "
            "Either side infinite → infinite discontinuity (vertical asymptote).",
            "All three equal → continuous at a.",
            "To prove a root exists with IVT: confirm f is continuous on the closed interval, then compute f(a) and f(b).",
            "If f(a) and f(b) have opposite signs, IVT guarantees some c in (a, b) with f(c) = 0.",
            "Locate c by bisection: evaluate the midpoint, keep the half where the sign still changes, repeat.",
        ],
        "solved": [
            {
                "title": "Classify a discontinuity",
                "problem": "Classify the discontinuity of f(x) = (x² − x − 6)/(x − 3) at x = 3.",
                "steps": [
                    "f(3) = 0/0 form — undefined, so f is not continuous at 3.",
                    "Factor the numerator: x² − x − 6 = (x − 3)(x + 2).",
                    "Cancel for x ≠ 3: f(x) = x + 2.",
                    "So lim(x→3) f(x) = 5, and the left and right limits agree.",
                    "The limit exists but f(3) does not → removable discontinuity, a hole at (3, 5).",
                ],
                "answer": "Removable discontinuity (hole at the point (3, 5)). Defining f(3) = 5 would make f continuous.",
                "pitfall": "Calling every 0/0 case an asymptote. Cancel first — if the factor cancels it is a hole, "
                           "if it survives in the denominator it is a vertical asymptote.",
            },
            {
                "title": "IVT plus bisection",
                "problem": "Show f(x) = x³ + x − 1 has a root in (0, 1) and locate it to three decimals.",
                "steps": [
                    "f is a polynomial, so it is continuous everywhere — IVT applies.",
                    "f(0) = −1 (negative) and f(1) = 1 (positive). Signs differ, so a root exists in (0, 1).",
                    "Bisect: f(0.5) = 0.125 + 0.5 − 1 = −0.375, still negative → root in (0.5, 1).",
                    "f(0.75) = 0.4219 + 0.75 − 1 = +0.172, positive → root in (0.5, 0.75).",
                    "f(0.625) = 0.2441 + 0.625 − 1 = −0.131, negative → root in (0.625, 0.75).",
                    "f(0.6875) = 0.3250 + 0.6875 − 1 = +0.012, positive → root in (0.625, 0.6875).",
                    "The bracket is now narrower than 0.001, so the root rounds to 0.682.",
                ],
                "answer": "A root exists in (0, 1); numerically c ≈ 0.682.",
                "pitfall": "Claiming IVT gives the exact root. It only guarantees existence — you still need "
                           "bisection or Newton's method to find the number.",
            },
        ],
    },
    "2.1 Rates of change and tangents": {
        "method": [
            "Average rate of change on [a, b]: substitute into [f(b) − f(a)] / (b − a).",
            "Instantaneous rate: form [f(a + h) − f(a)] / h, simplify until the h cancels, then let h → 0.",
            "Tangent line at x = a: find the point (a, f(a)), find the slope f′(a), then use "
            "y − f(a) = f′(a)(x − a) and solve for y.",
            "Normal line: same point, slope −1/f′(a). If f′(a) = 0 the tangent is horizontal and the normal is vertical.",
            "Motion problems: position s(t) → velocity v(t) = s′(t) → acceleration a(t) = s″(t). "
            "Speed is |v(t)|.",
            "Attach units to every answer: derivative units are output units per input unit.",
        ],
        "solved": [
            {
                "title": "Tangent line",
                "problem": "Find the tangent line to f(x) = x³ − 2x at x = 1.",
                "steps": [
                    "Point: f(1) = 1 − 2 = −1, so the point is (1, −1).",
                    "Derivative: f′(x) = 3x² − 2.",
                    "Slope at x = 1: f′(1) = 3 − 2 = 1.",
                    "Point-slope form: y − (−1) = 1·(x − 1), i.e. y + 1 = x − 1.",
                    "Solve for y: y = x − 2.",
                ],
                "answer": "y = x − 2",
                "pitfall": "Using the slope in place of the point, or forgetting the sign in y − f(a). "
                           "Check by substituting x = 1: y = −1 ✓.",
            },
            {
                "title": "From the limit definition",
                "problem": "Use the definition of the derivative to find f′(2) for f(x) = 1/x, then write the tangent line.",
                "steps": [
                    "Form the difference quotient: [1/(2 + h) − 1/2] / h.",
                    "Combine the two fractions in the numerator: [2 − (2 + h)] / [2(2 + h)] = −h / [2(2 + h)].",
                    "Divide by h: −1 / [2(2 + h)].",
                    "Let h → 0: f′(2) = −1/[2·2] = −1/4.",
                    "Point: f(2) = 1/2, so (2, 1/2).",
                    "Tangent: y − 1/2 = −(1/4)(x − 2) → y = −x/4 + 1.",
                ],
                "answer": "f′(2) = −1/4, tangent line y = −x/4 + 1",
                "pitfall": "Substituting h = 0 before the h cancels — that gives 0/0. Simplify first, then substitute.",
            },
        ],
    },
    "2.2 The derivative as a function": {
        "method": [
            "Write [f(x + h) − f(x)] / h with x kept symbolic, not a number.",
            "Expand the numerator fully; every h must cancel with the denominator.",
            "Substitute h = 0 to obtain f′(x).",
            "State the domain of f′ — it can be smaller than the domain of f.",
            "Check where f′ fails to exist: corners and cusps (one-sided derivatives differ), vertical tangents "
            "(the derivative blows up), and any discontinuity of f.",
            "Interpret the sign: f′ > 0 increasing, f′ < 0 decreasing, f″ > 0 concave up, f″ < 0 concave down.",
        ],
        "solved": [
            {
                "title": "Differentiate from the definition",
                "problem": "Find f′(x) for f(x) = √x using the limit definition.",
                "steps": [
                    "Set up: [√(x + h) − √x] / h.",
                    "h will not cancel as written — multiply top and bottom by the conjugate √(x + h) + √x.",
                    "Numerator becomes (x + h) − x = h.",
                    "So the quotient is h / [h(√(x + h) + √x)].",
                    "Cancel h: 1 / (√(x + h) + √x).",
                    "Let h → 0: f′(x) = 1/(2√x).",
                    "Domain: x > 0. At x = 0 the denominator is 0, which corresponds to a vertical tangent.",
                ],
                "answer": "f′(x) = 1/(2√x), valid for x > 0",
                "pitfall": "Reporting the domain as x ≥ 0. The function is defined at 0 but the derivative is not.",
            },
            {
                "title": "Continuous but not differentiable",
                "problem": "Show that f(x) = |x| is continuous at 0 but not differentiable there.",
                "steps": [
                    "Continuity: lim(x→0) |x| = 0 and f(0) = 0. They agree, so f is continuous at 0.",
                    "Differentiability: form [|h| − 0]/h = |h|/h.",
                    "From the right (h > 0): |h|/h = h/h = 1.",
                    "From the left (h < 0): |h|/h = −h/h = −1.",
                    "The one-sided derivatives are 1 and −1 — they differ, so the limit does not exist.",
                ],
                "answer": "Continuous at 0, but f′(0) does not exist because the graph has a corner there.",
                "pitfall": "Assuming continuity implies differentiability. It only goes one way: "
                           "differentiable ⇒ continuous.",
            },
        ],
    },
    "2.3 Differentiation rules": {
        "method": [
            "Simplify algebraically first. Expanding, splitting a fraction, or rewriting √x as x^(1/2) often "
            "removes the need for the product or quotient rule entirely.",
            "Identify the outermost structure: a sum, a product, a quotient, or a power of a function?",
            "Apply the matching rule, then recurse into any inner piece that still needs differentiating.",
            "Simplify the result and factor it — factored form is what makes later work (critical points) easy.",
            "Sanity check with a number: compare your formula against the numerical slope from the Python lab.",
        ],
        "solved": [
            {
                "title": "Quotient rule",
                "problem": "Differentiate y = (x² + 1)/(x − 3).",
                "steps": [
                    "Identify f = x² + 1 and g = x − 3, so f′ = 2x and g′ = 1.",
                    "Quotient rule: y′ = (f′g − fg′)/g².",
                    "Substitute: [2x(x − 3) − (x² + 1)(1)] / (x − 3)².",
                    "Expand the numerator: 2x² − 6x − x² − 1 = x² − 6x − 1.",
                    "The numerator does not factor nicely, so leave it expanded.",
                ],
                "answer": "y′ = (x² − 6x − 1)/(x − 3)²",
                "pitfall": "Reversing the numerator to (fg′ − f′g). The order matters — the correct form starts with "
                           "the derivative of the top.",
            },
            {
                "title": "Simplify first",
                "problem": "Differentiate y = x²·√x.",
                "steps": [
                    "The product rule would work, but rewriting is faster: √x = x^(1/2).",
                    "So y = x²·x^(1/2) = x^(5/2).",
                    "Power rule: y′ = (5/2)x^(3/2).",
                    "Optionally rewrite: (5/2)x√x.",
                ],
                "answer": "y′ = (5/2)x^(3/2)",
                "pitfall": "Reaching for the product rule out of habit. One extra line of algebra often saves "
                           "three lines of differentiation.",
            },
            {
                "title": "Product with a logarithm",
                "problem": "Differentiate y = x²·ln x.",
                "steps": [
                    "Genuine product: f = x², g = ln x, so f′ = 2x and g′ = 1/x.",
                    "Product rule: y′ = f′g + fg′ = 2x·ln x + x²·(1/x).",
                    "Simplify the second term: x²/x = x.",
                    "Factor: y′ = x(2 ln x + 1).",
                    "Domain note: ln x needs x > 0, so the derivative is only meaningful for x > 0.",
                ],
                "answer": "y′ = 2x·ln x + x = x(2 ln x + 1)",
                "pitfall": "Differentiating each factor separately and multiplying, giving 2x·(1/x) = 2. "
                           "That is not the product rule.",
            },
        ],
    },
    "2.4 Trigonometric derivatives": {
        "method": [
            "Confirm the angle is measured in radians. Every formula on this page assumes radians.",
            "Identify the structure: a bare trig function, a product, a quotient, or a composition?",
            "Apply the base rule, then multiply by the inner derivative if the argument is not simply x.",
            "Simplify with identities: sin²u + cos²u = 1, and 1 + tan²u = sec²u.",
            "Watch the signs: the derivatives of cos, csc and cot all carry a minus sign.",
        ],
        "solved": [
            {
                "title": "Quotient that collapses",
                "problem": "Differentiate y = sin x / (1 + cos x).",
                "steps": [
                    "f = sin x, g = 1 + cos x, so f′ = cos x and g′ = −sin x.",
                    "Quotient rule: y′ = [cos x(1 + cos x) − sin x(−sin x)] / (1 + cos x)².",
                    "Expand the numerator: cos x + cos²x + sin²x.",
                    "Apply sin²x + cos²x = 1: the numerator becomes cos x + 1.",
                    "Cancel with the denominator: y′ = (1 + cos x)/(1 + cos x)² = 1/(1 + cos x).",
                ],
                "answer": "y′ = 1/(1 + cos x)",
                "pitfall": "Stopping at the expanded numerator. Trig quotient problems almost always collapse "
                           "once you use an identity — if the answer looks messy, look again.",
            },
            {
                "title": "Product",
                "problem": "Differentiate y = x·tan x.",
                "steps": [
                    "f = x, g = tan x, so f′ = 1 and g′ = sec²x.",
                    "Product rule: y′ = 1·tan x + x·sec²x.",
                    "Nothing cancels, so this is the final form.",
                ],
                "answer": "y′ = tan x + x·sec²x",
                "pitfall": "Writing (tan x)′ as sec x rather than sec²x.",
            },
            {
                "title": "Squared sine",
                "problem": "Differentiate y = sin²x.",
                "steps": [
                    "Read this as (sin x)² — a composition, so the chain rule applies.",
                    "Outer: differentiate u² to get 2u, keeping u = sin x → 2 sin x.",
                    "Inner: derivative of sin x is cos x.",
                    "Multiply: y′ = 2 sin x cos x.",
                    "Recognise the double-angle identity: y′ = sin 2x.",
                ],
                "answer": "y′ = 2 sin x cos x = sin 2x",
                "pitfall": "Writing sin²x as sin(x²). The exponent attaches to the function value, not the angle.",
            },
        ],
    },
    "2.5 The chain rule": {
        "method": [
            "Ask what you would do LAST when evaluating the expression by hand — that is the outer function.",
            "Differentiate the outer function, leaving the inner expression completely untouched.",
            "Multiply by the derivative of the inner expression.",
            "If the inner expression is itself composite, repeat the process on it.",
            "Simplify and factor at the end.",
            "Verify by counting layers: n nested functions should produce n multiplicative factors.",
        ],
        "solved": [
            {
                "title": "Power of a function",
                "problem": "Differentiate y = (3x² + 1)⁵.",
                "steps": [
                    "Outer: u⁵ with u = 3x² + 1 → derivative 5u⁴ = 5(3x² + 1)⁴.",
                    "Inner: derivative of 3x² + 1 is 6x.",
                    "Multiply: y′ = 5(3x² + 1)⁴ · 6x.",
                    "Simplify the constants: y′ = 30x(3x² + 1)⁴.",
                ],
                "answer": "y′ = 30x(3x² + 1)⁴",
                "pitfall": "Also differentiating the inside of the bracket when applying the power rule, "
                           "giving 5(6x)⁴. Leave the inside untouched until the multiplication step.",
            },
            {
                "title": "Exponential",
                "problem": "Differentiate y = e^(sin x).",
                "steps": [
                    "Outer: e^u stays e^u, with u = sin x → e^(sin x).",
                    "Inner: derivative of sin x is cos x.",
                    "Multiply: y′ = e^(sin x)·cos x.",
                ],
                "answer": "y′ = cos x · e^(sin x)",
                "pitfall": "Answering e^(cos x). The exponent never changes — only a factor of u′ is added.",
            },
            {
                "title": "Three layers deep",
                "problem": "Differentiate y = ln(cos(x²)).",
                "steps": [
                    "Layer 1 (outermost): ln u with u = cos(x²) → derivative 1/u = 1/cos(x²).",
                    "Layer 2: cos v with v = x² → derivative −sin(x²).",
                    "Layer 3: x² → derivative 2x.",
                    "Multiply all three: y′ = [1/cos(x²)]·(−sin(x²))·2x.",
                    "Simplify using sin/cos = tan: y′ = −2x·tan(x²).",
                    "Domain: cos(x²) > 0 and cos(x²) ≠ 0, so the answer is only valid where cos(x²) > 0.",
                ],
                "answer": "y′ = −2x·tan(x²)",
                "pitfall": "Stopping after two layers. Three nested functions must produce three factors — "
                           "count them before moving on.",
            },
        ],
    },
    "2.6 Implicit differentiation": {
        "method": [
            "Differentiate both sides of the equation with respect to x, term by term.",
            "Whenever you differentiate a term containing y, multiply by dy/dx — because y is a function of x.",
            "For mixed terms such as xy or x²y², use the product rule AND keep the dy/dx factor on the y part.",
            "Move every term containing dy/dx to one side and everything else to the other.",
            "Factor dy/dx out, then divide to isolate it.",
            "Substitute the given point (x, y) only at the very end.",
            "Check the point actually satisfies the original equation before calling the result a tangent.",
        ],
        "solved": [
            {
                "title": "Circle",
                "problem": "Find dy/dx for x² + y² = 25, and the tangent at (3, 4).",
                "steps": [
                    "Differentiate both sides: 2x + 2y·y′ = 0. The y² term gains y′ because y depends on x.",
                    "Move the x-term over: 2y·y′ = −2x.",
                    "Divide: y′ = −x/y.",
                    "Substitute (3, 4): y′ = −3/4.",
                    "Confirm the point is on the curve: 9 + 16 = 25 ✓.",
                    "Tangent line: y − 4 = −(3/4)(x − 3), which simplifies to y = −0.75x + 6.25.",
                ],
                "answer": "dy/dx = −x/y; slope at (3, 4) is −3/4; tangent y = −0.75x + 6.25",
                "pitfall": "Differentiating y² as 2y without the y′ factor. Every y-term carries dy/dx with it.",
            },
            {
                "title": "Product term in x and y",
                "problem": "Find dy/dx for x²y + y³ = 5, and the slope at (2, 1).",
                "steps": [
                    "The term x²y mixes x and y, so use the product rule: (2x)(y) + (x²)(y′) = 2xy + x²y′.",
                    "The y³ term gives 3y²·y′.",
                    "The right side is a constant, so its derivative is 0.",
                    "Assemble: 2xy + x²y′ + 3y²y′ = 0.",
                    "Isolate the y′ terms: y′(x² + 3y²) = −2xy.",
                    "Divide: y′ = −2xy/(x² + 3y²).",
                    "Substitute (2, 1): y′ = −2(2)(1)/(4 + 3) = −4/7.",
                    "Confirm the point: 4·1 + 1 = 5 ✓.",
                ],
                "answer": "dy/dx = −2xy/(x² + 3y²); slope at (2, 1) is −4/7",
                "pitfall": "Treating x²y as if only one factor varies. Both change with x, so the product rule is required.",
            },
            {
                "title": "Both sides contain products",
                "problem": "Find dy/dx for x³ + y³ = 6xy, and the slope at (3, 3).",
                "steps": [
                    "Left side: 3x² + 3y²·y′.",
                    "Right side needs the product rule on 6xy: 6y + 6x·y′.",
                    "Set them equal: 3x² + 3y²y′ = 6y + 6xy′.",
                    "Collect y′ on the left: 3y²y′ − 6xy′ = 6y − 3x².",
                    "Factor: y′(3y² − 6x) = 6y − 3x².",
                    "Divide: y′ = (6y − 3x²)/(3y² − 6x) = (2y − x²)/(y² − 2x).",
                    "Substitute (3, 3): (6 − 9)/(9 − 6) = −3/3 = −1.",
                    "Confirm the point: 27 + 27 = 54 and 6·9 = 54 ✓.",
                ],
                "answer": "dy/dx = (2y − x²)/(y² − 2x); slope at (3, 3) is −1",
                "pitfall": "Forgetting the product rule on the right-hand side because it 'looks simple'. "
                           "6xy has both variables, so it needs the full treatment.",
            },
        ],
    },
}

LAB_METHODS: dict[str, list[str]] = {
    "Simplify and graph": [
        "Expand or factor the expression before doing anything else — a simpler form makes every later step easier.",
        "Read the graph: intercepts, symmetry (even/odd), and end behaviour as x grows.",
        "Vertical gaps in the curve mark points that are not in the domain.",
    ],
    "Domain of definition": [
        "List the restricted operations: denominators, even roots, logarithms, inverse trig.",
        "Turn each one into an inequality or exclusion and solve it.",
        "Intersect all the results; report in interval notation with the right brackets at each endpoint.",
    ],
    "First derivative (with rule walkthrough)": [
        "Simplify algebraically first if that avoids the product or quotient rule.",
        "Name the outermost structure, then apply its rule and recurse into inner pieces.",
        "Multiply by the inner derivative whenever the argument is not simply x.",
        "Factor the final answer — you will need it factored to find critical points later.",
    ],
    "Second derivative": [
        "Differentiate the first derivative using the same rules.",
        "Set f″ = 0 and solve to find candidate inflection points.",
        "Check that the sign of f″ actually changes there; a zero without a sign change is not an inflection.",
    ],
    "Limit at a point": [
        "Substitute first. A finite number is the answer.",
        "0/0 means indeterminate: factor and cancel, use a conjugate, combine fractions, or use a trig identity.",
        "If the one-sided limits differ, the two-sided limit does not exist — say so explicitly.",
    ],
    "Limit at infinity": [
        "Divide every term by the highest power of x in the denominator.",
        "Every 1/xⁿ term goes to 0 as x grows.",
        "Equal leading powers → ratio of leading coefficients; higher power below → 0; higher power above → ±∞.",
    ],
    "Tangent and normal lines": [
        "Find the point: compute f(a).",
        "Find the slope: compute f′(a).",
        "Write y − f(a) = f′(a)(x − a) and solve for y.",
        "Normal line: same point, slope −1/f′(a).",
    ],
    "Evaluate f(a)": [
        "Substitute a into the expression and simplify exactly where possible.",
        "If substitution is undefined, the question is really a limit — switch modes.",
        "Compare values either side of a to see whether the function is behaving smoothly there.",
    ],
    "Solve f(x) = 0": [
        "Factor if the expression is polynomial; set each factor to zero.",
        "Otherwise isolate x algebraically, or bracket a root and refine it by bisection.",
        "Always check each candidate root by substituting it back.",
    ],
    "Implicit differentiation": [
        "Differentiate both sides with respect to x.",
        "Every y-term gains a factor of dy/dx; mixed terms need the product rule too.",
        "Collect the dy/dx terms, factor, divide, then substitute the point at the end.",
    ],
}

GENERAL_METHOD = [
    "Write down what you are given and what is being asked. Name the function and the point.",
    "Decide which tool the question needs: a domain, a limit, a derivative, a tangent line, or a classification.",
    "Simplify algebraically before differentiating — expand, factor, split fractions, rewrite roots as powers.",
    "Apply the rule and write every step. Never skip the inner derivative in a composition.",
    "Check the answer: substitute a number, compare with the numerical slope in the lab, or ask whether the sign "
    "matches the shape of the graph.",
    "State the final answer with interval notation or units where the question calls for them, and note any "
    "restrictions on the domain.",
]


# --------------------------------------------------------------------------
# Page: Start here
# --------------------------------------------------------------------------
def page_start() -> None:
    st.title("📘 Calculus Python Tutor")
    st.markdown("Read the idea · follow a worked example · change the numbers yourself · then check what Python says.")

    st.markdown(
        """
**How to work through a lesson**

1. Pick a lesson on the left.
2. Read the *big idea* and the *key rules* — no more than a minute each.
3. Follow the worked example on paper before scrolling to the Python output.
4. Change the function or the point and watch what happens.
5. Use **Python calculus lab** for your own homework problems.
6. Finish with the **Practice quiz**.
"""
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Chapter 1 — Functions and Limits")
        st.write("Domains, transformations, composition, limits, limit laws, epsilon-delta, continuity, IVT.")
        latex(r"\lim_{x\to a} f(x)=L")
    with c2:
        st.markdown("### Chapter 2 — Derivatives")
        st.write("Average vs instantaneous change, tangent lines, the rules, trig derivatives, chain rule, implicit differentiation.")
        latex(r"f'(x)=\lim_{h\to 0}\frac{f(x+h)-f(x)}{h}")

    st.subheader("One intuition to keep")
    st.write(
        "Drive 100 km in 2 hours and your *average* speed is 50 km/h. Your speedometer at one exact "
        "moment is the derivative. Calculus is the mathematics of going from the first number to the second."
    )
    st.code(
        "distance = 100      # km\ntime = 2            # h\nprint(distance / time, 'km/h  <- average rate')",
        language="python",
    )

    st.subheader("Syntax cheat sheet")
    st.markdown(
        "| You want | Type it as | Also accepted |\n|---|---|---|\n"
        "| x² | `x**2` | `x^2` |\n"
        "| √x | `sqrt(x)` | `√x` |\n"
        "| 3x, 2sin(x) | `3*x`, `2*sin(x)` | `3x`, `2sin(x)` |\n"
        "| eˣ² | `exp(x**2)` | `e^(x^2)` |\n"
        "| ln(x) | `ln(x)` | `log(x)` |\n"
        "| |x| | `abs(x)` | `\\|x\\|` |\n"
        "| π | `pi` | `π` |\n"
    )
    st.caption(
        "Everything is computed locally by SymPy, NumPy and Matplotlib — no network, no upload."
    )


# --------------------------------------------------------------------------
# Page: a written lesson
# --------------------------------------------------------------------------
def page_lesson(topic: str) -> None:
    lesson = LESSONS[topic]
    lesson_header(topic, lesson["bigidea"])
    if lesson.get("formula"):
        latex(lesson["formula"])
    rule_list(lesson["rules"])

    st.subheader("Worked example")
    st.info(lesson["worked"])

    how = HOW_TO_SOLVE.get(topic)
    if how:
        render_method(how["method"])
        render_solved(how["solved"])

    key = topic.split()[0]
    if key == "1.1":
        _lesson_domain_interactive()
    elif key == "1.2":
        _lesson_transformations()
    elif key == "1.3":
        _lesson_limits()
    elif key == "1.4":
        _lesson_epsilon_delta()
    elif key == "1.5":
        _lesson_continuity()
    elif key == "2.1":
        _lesson_tangent()
    elif key == "2.2":
        _lesson_derivative_function()
    elif key in {"2.3", "2.4", "2.5"}:
        _lesson_rules(key)
    elif key == "2.6":
        _lesson_implicit()

    st.subheader("Try it in Python")
    st.code(EXAMPLE_CODE.get(topic, ""), language="python")
    st.caption("Run this in a terminal, or paste the same function into the Python calculus lab.")


EXAMPLE_CODE = {
    "1.1 Functions and domains": (
        "import sympy as sp\n"
        "x = sp.symbols('x', real=True)\n"
        "f = sp.sqrt(x - 2) / (x - 5)\n"
        "print(sp.calculus.util.continuous_domain(f, x, sp.S.Reals))\n"
    ),
    "1.2 Transformations and composition": (
        "import sympy as sp\n"
        "x = sp.symbols('x')\n"
        "f, g = x**2, x + 1\n"
        "print('f(g(x)) =', sp.expand(f.subs(x, g)))\n"
        "print('g(f(x)) =', sp.expand(g.subs(x, f)))\n"
    ),
    "1.3 Limits and limit laws": (
        "import sympy as sp\n"
        "x = sp.symbols('x')\n"
        "print(sp.limit((x**2 - 1)/(x - 1), x, 1))\n"
        "print(sp.limit(sp.sin(x)/x, x, 0))\n"
        "print(sp.limit((3*x**2 + 1)/(x**2 - x), x, sp.oo))\n"
    ),
    "1.4 Epsilon-delta definition": (
        "import sympy as sp\n"
        "x, eps = sp.symbols('x epsilon', positive=True)\n"
        "f = 2*x + 1\n"
        "gap = sp.simplify(sp.Abs(f.subs(x, 3 + sp.Symbol('d')) - 7))\n"
        "print('|f(x)-7| =', gap)   # want this < eps, so delta = eps/2\n"
    ),
    "1.5 Continuity and IVT": (
        "import sympy as sp\n"
        "x = sp.symbols('x')\n"
        "f = x**3 - 3*x + 1\n"
        "print('f(0) =', f.subs(x, 0), ' f(1) =', f.subs(x, 1))\n"
        "print('root in (0,1):', sp.nsolve(f, 0.5))\n"
    ),
    "2.1 Rates of change and tangents": (
        "import sympy as sp\n"
        "x, h = sp.symbols('x h')\n"
        "f = x**2\n"
        "diff_quot = sp.simplify((f.subs(x, x + h) - f)/h)\n"
        "print('[f(x+h)-f(x)]/h =', diff_quot)\n"
        "print('f\\'(x) =', sp.limit(diff_quot, h, 0))\n"
    ),
    "2.2 The derivative as a function": (
        "import sympy as sp\n"
        "x = sp.symbols('x')\n"
        "f = x**4 - 3*x**2 + 2\n"
        "print(sp.diff(f, x))\n"
        "print(sp.diff(f, x, 2))\n"
    ),
    "2.3 Differentiation rules": (
        "import sympy as sp\n"
        "x = sp.symbols('x', positive=True)\n"
        "print(sp.diff(x**2*sp.log(x), x))\n"
        "print(sp.diff((x**2 + 1)/(x - 3), x))\n"
    ),
    "2.4 Trigonometric derivatives": (
        "import sympy as sp\n"
        "x = sp.symbols('x')\n"
        "for f in (sp.sin(x), sp.cos(x), sp.tan(x), x*sp.sin(x)):\n"
        "    print(f, '->', sp.simplify(sp.diff(f, x)))\n"
    ),
    "2.5 The chain rule": (
        "import sympy as sp\n"
        "x = sp.symbols('x')\n"
        "print(sp.diff(sp.sin(3*x**2), x))\n"
        "print(sp.diff(sp.exp(x**2), x))\n"
        "print(sp.diff(sp.log(x**2 + 1), x))\n"
    ),
    "2.6 Implicit differentiation": (
        "import sympy as sp\n"
        "x = sp.symbols('x')\n"
        "y = sp.Function('y')(x)\n"
        "eq = sp.Eq(x**2 + y**2, 25)\n"
        "derived = sp.diff(eq.lhs - eq.rhs, x)\n"
        "dy = sp.Derivative(y, x)\n"
        "print(sp.solve(sp.Eq(derived, 0), dy)[0].subs(dy, sp.Symbol('dy/dx')))\n"
    ),
}


# --------------------------------------------------------------------------
# Interactive blocks per lesson
# --------------------------------------------------------------------------
def _lesson_domain_interactive() -> None:
    st.subheader("Check a domain yourself")
    raw = st.text_input("f(x) =", "sqrt(x - 2)/(x - 5)", key="dom_in")
    try:
        expr = try_expr(raw)
    except Exception as err:  # noqa: BLE001
        error_box(err)
        return
    latex("f(x) = " + ce.to_latex(expr))
    report = ce.domain_of(expr)
    if report.restrictions:
        st.write("**Restrictions found:**")
        for item in report.restrictions:
            st.markdown(f"- {item}")
    else:
        st.success("No denominator, root or logarithm found — defined for all real x.")
    if report.text:
        st.markdown(f"**Domain:** $ {report.latex} $")
    fig = ce.plot_series([(expr, f"y = {ce.to_text(expr)}")], -1, 10, "f(x) — gaps mark undefined points")
    show(fig)


def _lesson_transformations() -> None:
    st.subheader("See a transformation happen")
    c1, c2 = st.columns([2, 1])
    with c2:
        base_name = st.selectbox("Base function", ["x**2", "sqrt(x)", "sin(x)", "abs(x)", "exp(x)"], key="tf_base")
        k = st.slider("Vertical shift k", -4.0, 4.0, 1.0, 0.5, key="tf_k")
        p = st.slider("Horizontal shift p", -4.0, 4.0, 1.0, 0.5, key="tf_p")
        reflect = st.checkbox("Reflect: −f(x)", key="tf_ref")
    with c1:
        try:
            base = try_expr(base_name)
            moved = sp.expand((-(base) if reflect else base).subs(X, X - p) + k)
            fig = ce.plot_series(
                [(base, f"y = f(x) = {base_name}"), (moved, f"y = f(x − {p:g}) {'− ' if reflect else '+ '}{k:g}")],
                -6, 6, "Black: original · blue: transformed",
            )
            show(fig)
            st.caption("Shifting by +p inside the brackets moves the graph to the **right**, and +k outside moves it up.")
        except Exception as err:  # noqa: BLE001
            error_box(err)

    st.subheader("Composition is not commutative")
    raw_f = st.text_input("f(x)", "x**2", key="comp_f")
    raw_g = st.text_input("g(x)", "x + 1", key="comp_g")
    try:
        f, g = try_expr(raw_f), try_expr(raw_g)
        fg, gf = sp.expand(f.subs(X, g)), sp.expand(g.subs(X, f))
        a, b = st.columns(2)
        a.markdown(f"**f(g(x)) =** $ {ce.to_latex(fg)} $")
        b.markdown(f"**g(f(x)) =** $ {ce.to_latex(gf)} $")
        st.write("Same pair of functions, different results." if sp.simplify(fg - gf) != 0
                 else "For these two functions the compositions happen to agree.")
    except Exception as err:  # noqa: BLE001
        error_box(err)


def _lesson_limits() -> None:
    st.subheader("Evaluate a limit — and see why the trick works")
    raw = st.text_input("Expression", "(x**2 - 1)/(x - 1)", key="lim_expr")
    a_raw = st.text_input("x approaches", "1", key="lim_a")
    try:
        expr = try_expr(raw, allow_y=False)
        a = ce.parse_number(a_raw, default=1, name="the point a")
        report = ce.limit_at(expr, a)
        direct = expr.subs(X, a)
        st.write(f"Direct substitution gives: **{ce.number_text(direct)}**")
        if report.exists:
            st.success(f"lim(x→{ce.number_text(a)}) f(x) = {ce.number_text(report.two_sided)}")
            latex(r"\lim_{x \to " + ce.to_latex(a) + r"}\; " + ce.to_latex(expr)
                  + r" = " + ce.to_latex(report.two_sided))
        else:
            st.warning("The two-sided limit **does not exist**.")
        st.caption(report.note)
        fig = ce.plot_series([(expr, "f(x)")], ce.to_float(a, 1) - 4, ce.to_float(a, 1) + 4,
                             "Zoom in on the point the limit asks about")
        show(fig)
    except Exception as err:  # noqa: BLE001
        error_box(err)

    st.markdown("**Classic indeterminate forms to try**")
    for probe in ["(x**2 - 1)/(x - 1)", "sin(x)/x", "(1 - cos(x))/x", "1/x", "abs(x)/x"]:
        st.code(f"import sympy as sp\nx = sp.symbols('x')\nprint(sp.limit({probe}, x, 0))", language="python")


def _lesson_epsilon_delta() -> None:
    st.subheader("Pick ε, watch the δ window shrink")
    raw = st.text_input("Linear f(x) = mx + b", "2*x + 1", key="ed_f")
    eps = st.slider("Tolerance ε (allowed output error)", 0.05, 2.0, 0.4, 0.05, key="ed_eps")
    try:
        expr = try_expr(raw, allow_y=False)
        a = sp.Rational(3)
        L = expr.subs(X, a)
        slope = sp.diff(expr, X)
        if not slope.is_number:
            st.info("For the visualisation keep f linear (for example `2*x + 1`).")
            return
        m = float(slope)
        delta = eps / abs(m) if m else float("inf")
        st.markdown(
            f"We need |f(x) − {ce.number_text(L)}| < {eps:g}. Since |f(x) − L| = {abs(m):g}·|x − 3|, "
            f"choosing **δ = ε / {abs(m):g} = {delta:.4f}** works."
        )
        latex(r"\left|f(x)-" + ce.to_latex(L) + r"\right| = " + ce.to_latex(sp.Abs(expr - L))
              + r" < " + str(eps))
        fig = ce.plot_series([(expr, "f(x)"), (sp.Symbol("L"), f"L = {ce.number_text(L)}")],
                             3 - delta * 4, 3 + delta * 4, f"ε = {eps:g} window around x = 3")
        show(fig)
        st.caption("Narrow the ε slider: the x-window must narrow with it. That is exactly what the definition says.")
    except Exception as err:  # noqa: BLE001
        error_box(err)


def _lesson_continuity() -> None:
    st.subheader("Where does this function break?")
    raw = st.text_input("f(x)", "(x**2 - 4)/(x - 2)", key="cont_f")
    try:
        expr = try_expr(raw, allow_y=False)
        report = ce.domain_of(expr)
        st.write("**Defined where:**")
        latex(report.latex or r"\mathbb{R}")
        left = sp.limit(expr, X, 2, "-")
        right = sp.limit(expr, X, 2, "+")
        value = expr.subs(X, 2)
        st.markdown(
            f"- lim(x→2⁻) = **{ce.number_text(left)}**\n"
            f"- lim(x→2⁺) = **{ce.number_text(right)}**\n"
            f"- f(2) = **{ce.number_text(value)}**"
        )
        if left == right and str(value) == str(left):
            st.success("All three agree, so f is continuous at x = 2.")
        elif left == right:
            st.warning("The limit exists but f(2) is undefined — a **removable** discontinuity (a hole).")
        else:
            st.error("The one-sided limits differ — a **jump** discontinuity.")
        fig = ce.plot_series([(expr, "f(x)")], -5, 5, "Look for the hole, jump or asymptote")
        show(fig)
    except Exception as err:  # noqa: BLE001
        error_box(err)

    st.subheader("IVT: proving a root exists")
    raw2 = st.text_input("Polynomial f(x)", "x**3 - 3*x + 1", key="ivt_f")
    try:
        expr = try_expr(raw2, allow_y=False)
        lo = st.number_input("left end a", value=0.0, step=0.5, key="ivt_a")
        hi = st.number_input("right end b", value=1.0, step=0.5, key="ivt_b")
        fa = expr.subs(X, lo)
        fb = expr.subs(X, hi)
        st.write(f"f({lo:g}) = {ce.number_text(fa)},  f({hi:g}) = {ce.number_text(fb)}")
        if fa.is_real and fb.is_real and float(fa) * float(fb) < 0:
            st.success(f"Sign changes between {lo:g} and {hi:g}, so by IVT some c in ({lo:g}, {hi:g}) has f(c) = 0. "
                       f"SymPy finds c ≈ {float(sp.nsolve(expr, (lo + hi) / 2)):.5f}.")
        else:
            st.info("No guaranteed root here — try an interval where the signs differ.")
    except Exception as err:  # noqa: BLE001
        error_box(err)


def _lesson_tangent() -> None:
    st.subheader("Secant becomes tangent")
    raw = st.text_input("f(x)", "x**2", key="tan_f")
    a = st.slider("Point a", -3.0, 5.0, 3.0, 0.25, key="tan_a")
    dx = st.slider("Secant gap h", 0.1, 3.0, 0.1, 0.1, key="tan_h")
    try:
        expr = try_expr(raw, allow_y=False)
        info = ce.tangent_line(expr, sp.nsimplify(a))
        secant_slope = (expr.subs(X, a + dx) - expr.subs(X, a)) / dx
        secant_line = sp.expand(expr.subs(X, a) + secant_slope * (X - a))
        st.markdown(
            f"- secant slope over [{a:g}, {a + dx:g}] = **{ce.number_text(secant_slope)}**\n"
            f"- true slope f′({a:g}) = **{ce.number_text(info['slope'])}**\n"
            f"- tangent line: y = **{ce.to_text(info['line'])}**"
        )
        latex(r"y - " + ce.to_latex(info["height"]) + r" = " + ce.to_latex(info["slope"])
              + r"\,(x - " + ce.to_latex(sp.nsimplify(a)) + r")")
        fig = ce.plot_series(
            [(expr, f"y = {ce.to_text(expr)}"),
             (info["line"], f"tangent: {ce.to_text(info['line'])}"),
             (secant_line, f"secant (h = {dx:g})")],
            a - 3.5, a + 3.5, "Shrink h and the secant rotates onto the tangent",
            markers=[(a, float(sp.N(info["height"])), f"({a:g}, {ce.number_text(info['height'])})")],
        )
        show(fig)
        st.caption("Set h to its smallest value: the two lines are visually indistinguishable. That limit is the derivative.")

        st.markdown("**Motion interpretation**")
        pos = st.text_input("position s(t)", "t**3 - 6*t**2 + 9*t", key="motion_s")
        try:
            s = try_expr(pos.replace("t", "x"), allow_y=False)
            v = sp.diff(s, X)
            acc = sp.diff(s, X, 2)
            tt = st.slider("time t", 0.0, 4.0, 1.0, 0.25, key="motion_t")
            st.markdown(
                f"- s({tt:g}) = {ce.number_text(s.subs(X, tt))}\n"
                f"- v(t) = s′(t) = {ce.to_text(v)} → v({tt:g}) = {ce.number_text(v.subs(X, tt))}\n"
                f"- a(t) = s″(t) = {ce.to_text(acc)} → a({tt:g}) = {ce.number_text(acc.subs(X, tt))}"
            )
        except Exception as err:  # noqa: BLE001
            error_box(err)
    except Exception as err:  # noqa: BLE001
        error_box(err)


def _lesson_derivative_function() -> None:
    st.subheader("f, f′ and f″ together")
    raw = st.text_input("f(x)", "x**4 - 3*x**2 + 2", key="df_f")
    try:
        expr = try_expr(raw, allow_y=False)
        d1 = sp.diff(expr, X)
        d2 = sp.diff(expr, X, 2)
        st.markdown(f"**f′(x) =** {ce.to_text(d1)}")
        st.markdown(f"**f″(x) =** {ce.to_text(d2)}")
        fig = ce.plot_series(
            [(expr, "f(x)"), (d1, "f′(x)"), (d2, "f″(x)")],
            -2.5, 2.5, "Where f′ = 0 the curve flattens; where f″ = 0 concavity flips",
        )
        show(fig)

        st.markdown("**Where does f′ fail to exist?**")
        probe = st.selectbox("Try a function", ["abs(x)", "x**(1/3)", "1/x", "sqrt(x)"], key="df_probe")
        p = try_expr(probe, allow_y=False)
        try:
            dp = sp.diff(p, X)
            st.write(f"Algebraic derivative: {ce.to_text(dp)}")
            for point in [sp.Rational(-1), sp.Rational(1)]:
                st.write(f"  at x = {ce.number_text(point)}: f′ = {ce.number_text(dp.subs(X, point))}")
            left = sp.limit(dp, X, 0, "-")
            right = sp.limit(dp, X, 0, "+")
            st.write(f"  one-sided derivative limits at 0: left {ce.number_text(left)}, right {ce.number_text(right)}")
            if left != right:
                st.error("Left and right differ, or blow up: **f′(0) does not exist**.")
            else:
                st.success("Both sides agree at x = 0.")
        except Exception as err:  # noqa: BLE001
            st.write(f"SymPy cannot express f′ here — itself evidence the derivative fails: {err}")
        fig2 = ce.plot_series([(p, f"y = {probe}")], -3, 3, "Look for corners, cusps and asymptotes")
        show(fig2)
    except Exception as err:  # noqa: BLE001
        error_box(err)


def _lesson_rules(key: str) -> None:
    default = {"2.3": "x**2*log(x)", "2.4": "x*sin(x)", "2.5": "sin(3*x**2)"}[key]
    st.subheader("Name the rule, then verify")
    raw = st.text_input("f(x)", default, key=f"rule_{key}")
    try:
        expr = try_expr(raw, allow_y=False)
        walk = ce.derivative_walkthrough(expr)
        st.write("**Which rules apply, in order:**")
        for i, step in enumerate(walk["steps"], 1):
            detail = step["detail"]
            if step["formula"]:
                st.markdown(f"{i}. **{step['rule']}** — $ {step['formula']} $\n   {detail}")
            else:
                st.markdown(f"{i}. {step['rule']}: {detail}")
        st.success(f"f′(x) = {ce.to_text(walk['result'])}")
        latex(r"f'(x) = " + ce.to_latex(walk["result"]))
        st.info(walk["check"]["note"])
        fig = ce.plot_series([(expr, "f(x)"), (walk["result"], "f′(x)")], -4, 4,
                             "f′ is negative exactly where f is falling")
        show(fig)
    except Exception as err:  # noqa: BLE001
        error_box(err)


def _lesson_implicit() -> None:
    st.subheader("Differentiate both sides — nothing is hard-coded")
    raw = st.text_input("Equation in x and y", "x**2 + y**2 = 25", key="imp_eq")
    pt_raw = st.text_input("Point (optional), as  x, y", "3, 4", key="imp_pt")
    try:
        point = None
        if pt_raw and "," in pt_raw:
            xs, ys = pt_raw.split(",", 1)
            point = (ce.parse_number(xs, name="x-coordinate"),
                     ce.parse_number(ys, name="y-coordinate"))
        result = ce.implicit_derivative(raw, at=point)
        st.markdown("**Step 1 — differentiate both sides with respect to x**")
        latex(ce.to_latex(result["differentiated"]) + r" = 0")
        st.markdown("**Step 2 — collect the dy/dx terms and solve**")
        latex(r"\frac{dy}{dx} = " + ce.to_latex(result["dydx"]))
        st.success(f"dy/dx = {ce.to_text(result['dydx'])}")

        if point is not None:
            xv, yv = point
            slope = result.get("slope")
            if slope is not None:
                st.markdown(f"**Step 3 — substitute the point ({ce.number_text(xv)}, {ce.number_text(yv)})**")
                st.markdown(f"slope m = {ce.number_text(slope)}")
                if result.get("on_curve"):
                    st.success("The point lies on the curve, so this is a genuine tangent there.")
                else:
                    st.warning("That point is **not** on this curve — the slope is still computable, "
                               "but it is not a tangent to this equation there.")
                if slope.is_finite:
                    line = sp.expand(yv + slope * (X - xv))
                    latex(r"y - " + ce.to_latex(yv) + r" = " + ce.to_latex(slope)
                          + r"(x - " + ce.to_latex(xv) + r")")
                    st.markdown(f"Tangent line: y = {ce.to_text(line)}")
                    _plot_implicit_curve(raw, xv, yv, slope, line)
                else:
                    st.info("The tangent here is vertical, so it has no y = mx + b form.")
            elif result.get("slope_error"):
                st.warning(f"Could not evaluate the slope at that point: {result['slope_error']}")
    except Exception as err:  # noqa: BLE001
        error_box(err)


def _plot_implicit_curve(equation_text: str, xv, yv, slope, line) -> None:
    """Draw the circle/curve branch near the point plus its tangent line."""
    try:
        lhs, rhs = ce.parse_equation(equation_text)
        diff_expr = sp.simplify(lhs - rhs)
        # Solve for y where possible to get a drawable branch.
        sols = sp.solve(diff_expr, Y)
        series = []
        for sol in sols:
            series.append((sol, f"y = {ce.to_text(sol)}"))
        if not series:
            return
        series.append((line, f"tangent: y = {ce.to_text(line)}"))
        cx = ce.to_float(xv, 0)
        cy = ce.to_float(yv, 0)
        fig = ce.plot_series(series, cx - 6, cx + 6, "Curve and tangent at the chosen point",
                             markers=[(cx, cy, f"({cx:g}, {cy:g})")])
        show(fig)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------
# Page: Python calculus lab
# --------------------------------------------------------------------------
LAB_MODES = [
    "Simplify and graph",
    "Domain of definition",
    "First derivative (with rule walkthrough)",
    "Second derivative",
    "Limit at a point",
    "Limit at infinity",
    "Tangent and normal lines",
    "Evaluate f(a)",
    "Solve f(x) = 0",
    "Implicit differentiation",
]


def page_lab() -> None:
    st.header("Python calculus lab")
    st.write(
        "Your working area. Type a function the way you would type it in Python — `**` for powers, "
        "`*` for multiplication — or in ordinary math notation, which the parser will clean up for you."
    )

    mode = st.selectbox("What should Python do?", LAB_MODES)
    with st.expander("How to solve this kind of problem by hand", expanded=True):
        render_method(LAB_METHODS.get(mode, GENERAL_METHOD), "Method")
        st.caption("Python below gives the answer; these steps are what you write on paper.")
    implicit_mode = mode == "Implicit differentiation"
    default = "x**2 + y**2 = 25" if implicit_mode else "x**2 - 4*x + 3"
    raw = st.text_input("Equation (needs =)" if implicit_mode else "f(x) =", default)

    try:
        if implicit_mode:
            _lab_implicit(raw)
            return
        expr = try_expr(raw, allow_y=False)
    except Exception as err:  # noqa: BLE001
        error_box(err)
        return

    st.markdown(f"**Understood as:** $ f(x) = {ce.to_latex(expr)} $")

    if mode == "Simplify and graph":
        simp = sp.simplify(expr)
        factored = sp.factor(expr) if expr.is_polynomial() else None
        st.write("Simplified:", ce.to_text(simp))
        if factored is not None and str(factored) != str(expr):
            st.write("Factored:", ce.to_text(factored))
        latex("f(x) = " + ce.to_latex(simp))
        show(ce.plot_series([(expr, f"y = {ce.to_text(expr)}")], title="Graph"))

    elif mode == "Domain of definition":
        report = ce.domain_of(expr)
        for item in report.restrictions:
            st.markdown(f"- {item}")
        if not report.restrictions:
            st.success("Defined for every real number.")
        if report.latex:
            st.markdown(f"**Domain:** $ {report.latex} $")
        show(ce.plot_series([(expr, "f(x)")], title="Undefined regions leave gaps"))

    elif mode == "First derivative (with rule walkthrough)":
        walk = ce.derivative_walkthrough(expr)
        for i, step in enumerate(walk["steps"], 1):
            if step["formula"]:
                st.markdown(f"{i}. **{step['rule']}** — $ {step['formula']} $")
                st.caption(step["detail"])
            else:
                st.markdown(f"{i}. {step['rule']} — {step['detail']}")
        st.success(f"f′(x) = {ce.to_text(walk['result'])}")
        latex("f'(x) = " + ce.to_latex(walk["result"]))
        st.info(walk["check"]["note"])
        a, b = st.columns(2)
        with a:
            show(ce.plot_series([(expr, "f(x)")], title="Original"))
        with b:
            show(ce.plot_series([(walk["result"], "f′(x)")], title="Derivative"))

    elif mode == "Second derivative":
        d1 = sp.diff(expr, X)
        d2 = sp.diff(expr, X, 2)
        st.markdown(f"f′(x) = {ce.to_text(d1)}")
        st.success(f"f″(x) = {ce.to_text(d2)}")
        latex("f''(x) = " + ce.to_latex(d2))
        try:
            infl = sp.solveset(sp.Eq(d2, 0), X, domain=sp.S.Reals)
            st.write("Possible inflection points (f″ = 0):", ce.to_text(infl))
        except Exception:  # noqa: BLE001
            pass
        show(ce.plot_series([(expr, "f"), (d2, "f″")], title="Concavity flips where f″ crosses zero"))

    elif mode == "Limit at a point":
        a_raw = st.text_input("x approaches", "1")
        direction = st.selectbox("Direction", ["Both sides", "From the left", "From the right"])
        try:
            a = ce.parse_number(a_raw, default=1, name="the point")
            if direction == "Both sides":
                report = ce.limit_at(expr, a)
                if report.exists:
                    st.success(f"Limit = {ce.number_text(report.two_sided)}")
                    latex(r"\lim_{x \to " + ce.to_latex(a) + r"} f(x) = " + ce.to_latex(report.two_sided))
                else:
                    st.error("Two-sided limit does not exist.")
                    st.markdown(
                        f"- left: {ce.number_text(report.left)}\n- right: {ce.number_text(report.right)}"
                    )
                st.caption(report.note)
            else:
                side = "-" if direction == "From the left" else "+"
                val = sp.limit(expr, X, a, side)
                st.success(f"{direction} limit = {ce.number_text(val)}")
                latex((r"\lim_{x \to " + ce.to_latex(a) + "^{" + side + r"}} f(x) = ") + ce.to_latex(val))
        except Exception as err:  # noqa: BLE001
            error_box(err)

    elif mode == "Limit at infinity":
        val = sp.limit(expr, X, sp.oo)
        st.success(f"lim(x→+∞) f(x) = {ce.number_text(val)}")
        latex(r"\lim_{x \to \infty} f(x) = " + ce.to_latex(val))
        st.write(
            "Rational-function shortcut: compare the top and bottom powers of x. Equal powers → ratio of "
            "leading coefficients; bottom bigger → 0; top bigger → ±∞."
        )
        show(ce.plot_series([(expr, "f(x)")], 0, 40, "Behaviour far to the right"))

    elif mode == "Tangent and normal lines":
        a_raw = st.text_input("Tangent point x = a", "2")
        try:
            a = ce.parse_number(a_raw, default=2, name="a")
            info = ce.tangent_line(expr, a)
            if not info["exists"]:
                st.error(f"No finite slope at x = {ce.number_text(a)} — the tangent is vertical here.")
            else:
                st.success(
                    f"f({ce.number_text(a)}) = {ce.number_text(info['height'])},  "
                    f"slope = {ce.number_text(info['slope'])}"
                )
                latex(r"y - " + ce.to_latex(info["height"]) + r" = " + ce.to_latex(info["slope"])
                      + r"(x - " + ce.to_latex(a) + r")")
                st.markdown(f"Tangent: y = {ce.to_text(info['line'])}")
                st.markdown(f"Normal slope: {ce.number_text(info['normal_slope'])}")
                av = ce.to_float(a, 0)
                show(ce.plot_series(
                    [(expr, "f(x)"), (info["line"], "tangent")],
                    av - 4, av + 4, "Tangent touches with the same slope as the curve",
                    markers=[(av, ce.to_float(info["height"], 0),
                              f"({av:g}, {ce.number_text(info['height'])})")],
                ))
        except Exception as err:  # noqa: BLE001
            error_box(err)

    elif mode == "Evaluate f(a)":
        a_raw = st.text_input("a =", "2")
        try:
            a = ce.parse_number(a_raw, default=2, name="a")
            value = expr.subs(X, a)
            st.success(f"f({ce.number_text(a)}) = {ce.number_text(value)}")
            latex(f"f({ce.to_latex(a)}) = " + ce.to_latex(value))
            st.markdown("**Values around the point**")
            # Rendered as a markdown table on purpose: st.dataframe pulls in
            # pandas, whose compiled extensions can be blocked by Windows
            # application-control policies on some machines.
            av = ce.to_float(a, 0)
            rows = []
            for k in range(-3, 4):
                probe = sp.nsimplify(av + 0.5 * k)
                try:
                    rows.append((ce.number_text(probe), ce.number_text(expr.subs(X, probe))))
                except Exception:  # noqa: BLE001
                    rows.append((ce.number_text(probe), "undefined"))
            table = ["| x | f(x) |", "|---|---|"]
            for label, value in rows:
                table.append(f"| {label} | {value} |")
            st.markdown("\n".join(table))
        except Exception as err:  # noqa: BLE001
            error_box(err)

    elif mode == "Solve f(x) = 0":
        try:
            sols = sp.solveset(sp.Eq(expr, 0), X, domain=sp.S.Reals)
            st.markdown(f"Real solutions: $ {ce.to_latex(sols)} $")
            numerics = sp.solve(expr, X)
            if numerics:
                st.write("Exact forms:", ", ".join(ce.to_text(s) for s in numerics))
            for guess in (-2.0, 0.0, 2.0):
                try:
                    st.write(f"root near {guess:g}:", ce.number_text(sp.nsolve(expr, guess)))
                except Exception:  # noqa: BLE001
                    pass
            show(ce.plot_series([(expr, "f(x)")], -6, 6, "Roots are where the curve crosses the x-axis"))
        except Exception as err:  # noqa: BLE001
            error_box(err)


def _lab_implicit(raw: str) -> None:
    try:
        pt = st.text_input("Slope at a point, as  x, y  (leave blank to skip)", "3, 4")
        point = None
        if pt and "," in pt:
            xs, ys = pt.split(",", 1)
            point = (ce.parse_number(xs, name="x-coordinate"),
                     ce.parse_number(ys, name="y-coordinate"))
        result = ce.implicit_derivative(raw, at=point)
        st.markdown("Equation: $ " + ce.to_latex(result["lhs"]) + r" = " + ce.to_latex(result["rhs"]) + " $")
        st.markdown("**After differentiating both sides with respect to x:**")
        latex(ce.to_latex(result["differentiated"]) + r" = 0")
        st.success(f"dy/dx = {ce.to_text(result['dydx'])}")
        latex(r"\frac{dy}{dx} = " + ce.to_latex(result["dydx"]))
        if point is not None:
            xv, yv = point
            slope = result.get("slope")
            if slope is None:
                st.warning(f"Could not evaluate the slope at that point: {result.get('slope_error', 'unknown')}")
            else:
                st.markdown(f"At ({ce.number_text(xv)}, {ce.number_text(yv)}): m = **{ce.number_text(slope)}**")
                if result.get("on_curve"):
                    st.success("That point satisfies the equation.")
                else:
                    st.warning("That point does not satisfy the equation — check the coordinates.")
                if slope.is_finite:
                    _plot_implicit_curve(raw, xv, yv, slope, sp.expand(yv + slope * (X - xv)))
                else:
                    st.info("Vertical tangent here, so no y = mx + b line to draw.")
    except Exception as err:  # noqa: BLE001
        error_box(err)


# --------------------------------------------------------------------------
# Page: Practice quiz
# --------------------------------------------------------------------------
QUIZ = [
    ("What restriction does 1/(x − 4) place on the domain?",
     ["x may be any real number", "x ≠ 4", "x > 4"], 1,
     "A denominator may not be zero, so x = 4 is excluded."),
    ("lim(x→1) (x² − 1)/(x − 1) = ?",
     ["0", "1", "2"], 2,
     "0/0 is indeterminate. Factor to (x − 1)(x + 1)/(x − 1), cancel, substitute x = 1 → 2."),
    ("The derivative of x⁵ is:",
     ["5x⁴", "x⁴", "5x"], 0,
     "Power rule: bring the exponent down, subtract one from it."),
    ("The derivative of sin(x) is:",
     ["−sin(x)", "cos(x)", "tan(x)"], 1,
     "(sin x)′ = cos x. The minus sign belongs to cosine's derivative."),
    ("The derivative of e^(x²) is:",
     ["e^(x²)", "x·e^(x²)", "2x·e^(x²)"], 2,
     "Chain rule: (e^u)′ = e^u·u′ with u = x², so u′ = 2x."),
    ("For x² + y² = 25, dy/dx = ?",
     ["x/y", "−x/y", "−y/x"], 1,
     "2x + 2y·y′ = 0 gives y′ = −x/y."),
    ("f is continuous at a exactly when:",
     ["f(a) exists", "lim(x→a) f(x) exists",
      "the limit exists, f(a) exists, and they are equal"], 2,
     "All three parts are required — a hole fails the third."),
    ("|x| at x = 0 is:",
     ["continuous but not differentiable", "neither continuous nor differentiable",
      "differentiable but not continuous"], 0,
     "The graph has no break, but it has a corner, so the slope is undefined at 0."),
    ("Which limit equals 1?",
     ["lim(x→0) sin(x)/x", "lim(x→0) x/sin(x) only", "neither"], 0,
     "This is the fundamental radian limit; it is why calculus insists on radians."),
    ("(ln x)′ equals:",
     ["x·ln x", "1/x", "ln x"], 1,
     "The logarithm rule is (ln x)′ = 1/x, defined for x > 0."),
]


QUIZ_METHODS = {
    "Domain questions": [
        "Spot the restricted operation: denominator (≠ 0), even root (≥ 0), logarithm (> 0).",
        "Turn each into an inequality or exclusion and solve.",
        "Combine all conditions; report with interval notation.",
    ],
    "Limit questions": [
        "Substitute first. A finite value is the answer.",
        "0/0 → factor and cancel, or use a conjugate.",
        "If the two sides disagree, the limit does not exist.",
    ],
    "Derivative questions": [
        "Name the structure: power, product, quotient, or composition.",
        "Apply that rule; for a composition, multiply by the inner derivative.",
        "Simplify and factor the result.",
    ],
    "Trigonometric derivatives": [
        "Work in radians.",
        "Memorise: (sin)′ = cos, (cos)′ = −sin, (tan)′ = sec².",
        "Cofunction derivatives (cos, csc, cot) all carry a minus sign.",
    ],
    "Implicit differentiation": [
        "Differentiate both sides; every y-term gains dy/dx.",
        "Collect the dy/dx terms, factor, divide.",
        "Substitute the point only at the end.",
    ],
    "Continuity questions": [
        "Check three things: is f(a) defined, does the limit exist, do they agree?",
        "Hole = removable, unequal sides = jump, infinite = asymptote.",
    ],
}


def page_quiz() -> None:
    st.header("Practice quiz")
    render_method(GENERAL_METHOD, "How to approach any problem here")
    st.subheader("Method by question type")
    for qtype, steps in QUIZ_METHODS.items():
        with st.expander(qtype):
            for i, step in enumerate(steps, 1):
                st.markdown(f"{i}. {step}")
    st.markdown("---")
    st.write("Answer all ten, then press **Check my answers**. Each result explains the reasoning.")
    answers = []
    for i, (question, options, _correct, _why) in enumerate(QUIZ):
        st.markdown(f"**{i + 1}. {question}**")
        answers.append(st.radio("Pick one", options, key=f"quiz_{i}", index=None,
                                horizontal=False))
    if st.button("Check my answers", type="primary"):
        unanswered = sum(1 for a in answers if a is None)
        if unanswered:
            st.warning(f"{unanswered} question(s) are still blank — they will count as incorrect.")
        score = 0
        detail = []
        for i, (question, options, correct, why) in enumerate(QUIZ):
            picked = answers[i]
            got = picked == options[correct]
            score += int(got)
            detail.append({"q": i + 1, "ok": got, "answer": options[correct], "why": why})
        for row in detail:
            if row["ok"]:
                st.success(f"Q{row['q']}: correct. {row['why']}")
            else:
                st.error(f"Q{row['q']}: the answer is **{row['answer']}**. {row['why']}")
        st.metric("Score", f"{score} / {len(QUIZ)}")
        if score == len(QUIZ):
            st.balloons()
        elif score >= 7:
            st.write("Solid. Re-read the explanations for the ones you missed and try the lab with those functions.")
        elif score >= 4:
            st.write("Halfway there. Go back to the lessons named in the questions you missed.")
        else:
            st.write("Start with 1.3 limits and 2.3 differentiation rules, then retry.")


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
st.sidebar.title("📘 Calculus Tutor")
topic = st.sidebar.radio("Lesson", TOPICS)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Type `x**2` or `x^2`, `sqrt(x)` or `√x`, `3x` or `3*x` — both work. "
    "All maths runs locally with SymPy."
)
if topic == "1.4 Epsilon-delta definition":
    st.sidebar.info("Symbols: ε is the tolerance you demand on the output; δ is the window around x that delivers it.")

if topic == "Start here":
    page_start()
elif topic == "Python calculus lab":
    page_lab()
elif topic == "Practice quiz":
    page_quiz()
else:
    page_lesson(topic)

st.markdown("---")
st.caption(
    "Teaching aid built from the uploaded Calculus Complete Study Notes, Chapters 1–2. "
    "Python checks your algebra — it does not replace practising the steps by hand."
)
