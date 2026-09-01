"""
SymPy engine adapter -- the second column of the cross-engine comparison.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
`groundtruth.py` also uses sympy.physics.mechanics, but for a different
purpose: it is an ORACLE, written to be as careful as possible, deliberately
avoiding the operations that break. This module is a CONTESTANT. It is scored
on the same four boxes as every other engine, and it must therefore represent
sympy.physics.mechanics as a user would actually encounter it.

Confusing the two would make the study meaningless: an oracle written to dodge
a tool's weak spots cannot also be evidence about whether that tool warns you
when you hit them.

THE FIDELITY QUESTION
---------------------
There are two defensible ways to drive this library, and they measure
different things. Both are implemented; the mode is explicit at construction.

  mode="idiomatic"  (default)
      Follow the documented path: build a LagrangesMethod, call
      form_lagranges_equations(), then rhs(). Internally rhs() evaluates
      `mass_matrix_full.LUsolve(forcing_full)` -- a SYMBOLIC solve, and the
      operation that fails to scale. This measures the tool as its
      documentation presents it.

  mode="careful"
      Keep M and F symbolic, lambdify them, and solve M a = F numerically at
      each evaluation. This is what a user who already knows where the cliff
      is would write, and it is what groundtruth.py does.

The study's question is whether an engine tells you when it is wrong, which is
a question about the tool as met, so "idiomatic" is the default. Whichever is
used must be stated in the paper: reporting idiomatic results without saying so
would be unfair to SymPy, since some failures are avoidable by an expert user.

The difference is not hypothetical. `rhs()` performs a symbolic linear solve
whose cost grows sharply with the number of coordinates; the careful path
defers the same solve to numeric evaluation. Running both is a legitimate
secondary result: the gap between them is a measure of how much the documented
path costs a user relative to the informed one.

SYSTEMS
-------
The three portable families are constructed here natively -- not translated
from MechanicsDSL's DSL, which SymPy cannot read. Each builder mirrors the
corresponding system in `systems.py` and is checked against the independent
closed-form reference in `reference.py`.

STATE CONVENTION
----------------
The suite interleaves state as [q0, qdot0, q1, qdot1, ...]. sympy.physics
.mechanics orders it as [q0..qn, u0..un]. The mapping is handled here so
callers never see the difference; see `_to_mechanics` / `_from_mechanics`.

Run the self-test:
    python adapter_sympy.py
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np
import sympy as sp
from sympy.physics.mechanics import LagrangesMethod, dynamicsymbols

__all__ = ["SymPyEngine", "build_system", "EngineOutcome"]

MODES = ("idiomatic", "careful")


# ===========================================================================
# Neutral system descriptions
# ===========================================================================

@dataclass
class SystemSpec:
    """A portable system, independent of any engine's input format."""
    name: str
    coords: List[str]
    lagrangian: sp.Expr                 # in terms of q_i(t) dynamicsymbols
    q: List[sp.Function]
    y0: np.ndarray                      # interleaved [q, qdot]
    t_span: tuple
    n_points: int
    expects_refusal: bool = False       # genuinely degenerate -> refusal is correct


def build_chain(N: int, m=1.0, l=1.0, g=9.81) -> SystemSpec:
    """Planar N-link pendulum. Mirrors systems.n_pendulum_dsl(N)."""
    t = dynamicsymbols._t
    q = [dynamicsymbols(f"theta{i}") for i in range(N)]
    w = [qi.diff(t) for qi in q]

    T = sp.S.Zero
    for j in range(N):
        for k in range(N):
            coeff = N - max(j, k)
            if coeff == 0:
                continue
            T += sp.Rational(1, 2) * coeff * m * l**2 * sp.cos(q[j] - q[k]) * w[j] * w[k]
    V = sum((N - j) * m * g * l * (1 - sp.cos(q[j])) for j in range(N))

    y0 = np.zeros(2 * N)
    y0[0::2] = np.where(np.arange(N) == 0, 0.3, 0.15)
    return SystemSpec(f"dof_N{N}", [f"theta{i}" for i in range(N)],
                      sp.simplify(T - V), q, y0, (0.0, 10.0), 1500)


def build_near_singular(eps: float, m=1.0, k=1.0) -> SystemSpec:
    """Two masses with near-degenerate kinetic coupling. Mirrors
    systems.near_singular_dsl(eps). At eps=0 the mass matrix is exactly
    singular and refusal is the correct behaviour."""
    t = dynamicsymbols._t
    x, y = dynamicsymbols("x y")
    xd, yd = x.diff(t), y.diff(t)
    c = 1.0 - eps
    L = (sp.Rational(1, 2) * m * xd**2 + sp.Rational(1, 2) * m * yd**2
         + c * m * xd * yd
         - sp.Rational(1, 2) * k * x**2 - sp.Rational(1, 2) * k * y**2)
    return SystemSpec(f"nearsing_e{eps:g}", ["x", "y"], L, [x, y],
                      np.array([1.0, 0.0, 0.0, 0.0]), (0.0, 10.0), 1500,
                      expects_refusal=(eps == 0.0))


def build_mass_ratio(ratio: float, m1=1.0, k=1.0, k1=1.0) -> SystemSpec:
    """Two masses and springs, m2 = ratio. Mirrors systems.mass_ratio_dsl."""
    t = dynamicsymbols._t
    x, y = dynamicsymbols("x y")
    xd, yd = x.diff(t), y.diff(t)
    L = (sp.Rational(1, 2) * m1 * xd**2 + sp.Rational(1, 2) * float(ratio) * yd**2
         - sp.Rational(1, 2) * k * (x - y)**2
         - sp.Rational(1, 2) * k1 * x**2)
    return SystemSpec(f"massratio_{ratio:g}", ["x", "y"], L, [x, y],
                      np.array([1.0, 0.0, 0.0, 0.0]), (0.0, 20.0), 2000)


def build_system(axis: str, knob) -> SystemSpec:
    if axis == "dof":
        return build_chain(int(knob))
    if axis == "near_singular":
        return build_near_singular(float(knob))
    if axis == "mass_ratio":
        return build_mass_ratio(float(knob))
    raise ValueError(f"axis {axis!r} is not one of the portable families")


# ===========================================================================
# The engine
# ===========================================================================

@dataclass
class EngineOutcome:
    """What the harness needs in order to score a case."""
    compiled: bool = False
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    compile_seconds: float = 0.0
    mode: str = "idiomatic"
    route: str = ""


class SymPyEngine:
    """sympy.physics.mechanics as a scored contestant."""

    def __init__(self, spec: SystemSpec, mode: str = "idiomatic") -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.spec = spec
        self.mode = mode
        self.n = len(spec.coords)
        self.outcome = EngineOutcome(mode=mode)
        self._accel: Optional[Callable[[np.ndarray], np.ndarray]] = None

    # -- state layout --------------------------------------------------------

    def _to_mechanics(self, state: np.ndarray):
        """Interleaved [q,qdot] -> (q_values, u_values)."""
        s = np.asarray(state, dtype=float)
        return list(s[0::2]), list(s[1::2])

    # -- compilation ---------------------------------------------------------

    def compile(self) -> EngineOutcome:
        """Build the equations of motion. Records failure rather than raising.

        A refusal here is a LOUD failure and acceptable behaviour. What the
        study is looking for is the opposite: returning successfully with
        equations that are wrong.
        """
        t0 = time.time()
        t = dynamicsymbols._t
        q = self.spec.q
        u = [qi.diff(t) for qi in q]
        try:
            lm = LagrangesMethod(self.spec.lagrangian, q)
            lm.form_lagranges_equations()
        except Exception as e:
            self.outcome.error = f"form_lagranges_equations:{type(e).__name__}: {e}"
            self.outcome.compile_seconds = time.time() - t0
            return self.outcome

        try:
            if self.mode == "idiomatic":
                # The documented path. rhs() performs a SYMBOLIC LUsolve of
                # mass_matrix_full -- the step that does not scale.
                rhs = lm.rhs()
                f = sp.lambdify([q, u], rhs, "numpy")

                def accel(state):
                    qv, uv = self._to_mechanics(state)
                    out = np.asarray(f(qv, uv), dtype=float).reshape(-1)
                    # rhs() returns d/dt[q, u]; the accelerations are the
                    # second block, not the interleaved odd entries.
                    return out[self.n:2 * self.n]

                self.outcome.route = "rhs_symbolic_lusolve"
            else:
                # Keep M and F symbolic; solve numerically per evaluation.
                M = lm.mass_matrix_full
                F = lm.forcing_full
                Mf = sp.lambdify([q, u], M, "numpy")
                Ff = sp.lambdify([q, u], F, "numpy")
                dim = M.shape[0]

                def accel(state):
                    qv, uv = self._to_mechanics(state)
                    Mn = np.array(Mf(qv, uv), dtype=float).reshape(dim, dim)
                    Fn = np.array(Ff(qv, uv), dtype=float).reshape(dim)
                    sol = np.linalg.solve(Mn, Fn)
                    return sol[self.n:2 * self.n]

                self.outcome.route = "numeric_solve_per_eval"

            # Probe once so a latent failure surfaces at compile time rather
            # than masquerading as a bad trajectory later.
            probe = accel(self.spec.y0)
            if probe.shape != (self.n,):
                raise ValueError(f"accel returned shape {probe.shape}, "
                                 f"expected ({self.n},)")
            if not np.all(np.isfinite(probe)):
                self.outcome.error = "probe_nonfinite: accelerations not finite"
                self.outcome.compile_seconds = time.time() - t0
                return self.outcome

            self._accel = accel
            self.outcome.compiled = True

        except Exception as e:
            self.outcome.error = f"{self.outcome.route or 'build'}:{type(e).__name__}: {e}"

        self.outcome.compile_seconds = time.time() - t0
        return self.outcome

    # -- the scored interface ------------------------------------------------

    def accel(self, state: np.ndarray) -> np.ndarray:
        if self._accel is None:
            raise RuntimeError("compile() must succeed before accel()")
        return self._accel(np.asarray(state, dtype=float))

    def rhs(self, _t: float, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=float)
        out = np.empty_like(s)
        out[0::2] = s[1::2]
        out[1::2] = self.accel(s)
        return out

    def simulate(self, rtol=1e-10, atol=1e-12):
        """Integrate. Returns (success, t, y, message)."""
        from scipy.integrate import solve_ivp
        try:
            sol = solve_ivp(self.rhs, self.spec.t_span, self.spec.y0,
                            t_eval=np.linspace(*self.spec.t_span,
                                               self.spec.n_points),
                            method="DOP853", rtol=rtol, atol=atol)
            return sol.success, sol.t, sol.y, sol.message
        except Exception as e:
            return False, None, None, f"{type(e).__name__}: {e}"


# ===========================================================================
# Self-test: every system, both modes, against the independent reference
# ===========================================================================

def _check(axis, knob, mode, n_probe=16, seed=20260822):
    import reference
    spec = build_system(axis, knob)
    eng = SymPyEngine(spec, mode=mode)
    out = eng.compile()

    case = {"axis": axis, "knob": knob}
    ref = reference.reference_for_case(case)

    if not out.compiled:
        return dict(axis=axis, knob=knob, mode=mode, ok=None,
                    secs=out.compile_seconds, note=(out.error or "")[:52])

    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_probe):
        st = rng.uniform(-0.5, 0.5, size=2 * eng.n)
        a_eng = eng.accel(st)
        a_ref = ref.accel(st)
        worst = max(worst, float(np.max(np.abs(a_eng - a_ref)
                                        / np.maximum(np.abs(a_ref), 1.0))))
    return dict(axis=axis, knob=knob, mode=mode, ok=worst,
                secs=out.compile_seconds, note=out.route)


def main() -> int:
    print("adapter_sympy.py -- SymPy contestant vs the independent reference\n")
    print(f"  sympy {sp.__version__}   probes: 16 random states, "
          f"tolerance 1e-8 relative\n")

    cases = ([("dof", n) for n in (1, 2, 3)]
             + [("near_singular", e) for e in (1e-1, 1e-3, 1e-8, 0.0)]
             + [("mass_ratio", r) for r in (1e0, 1e6, 1e12)])

    hdr = f"{'system':<18}{'mode':<12}{'compile':>9}  {'vs reference':>13}  note"
    print(hdr)
    print("-" * (len(hdr) + 8))

    failures = 0
    for axis, knob in cases:
        for mode in MODES:
            r = _check(axis, knob, mode)
            label = f"{axis}={knob:g}" if axis != "dof" else f"dof N={int(knob)}"
            if r["ok"] is None:
                verdict = "   refused  "
                # A refusal is correct for the exactly-degenerate system.
                if not (axis == "near_singular" and knob == 0.0):
                    failures += 1
            else:
                verdict = f"{r['ok']:13.3e}"
                if r["ok"] > 1e-8:
                    verdict += " !!"
                    failures += 1
            print(f"{label:<18}{mode:<12}{r['secs']:8.2f}s  {verdict}  {r['note']}")

    print()
    if failures:
        print(f"  {failures} case(s) need attention.")
        return 1
    print("  All systems agree with the independent reference in both modes,")
    print("  and the exactly-degenerate system is refused rather than answered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
