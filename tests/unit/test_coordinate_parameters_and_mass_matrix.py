"""Regression tests for two fixes:

1. A coordinate declared with a parameter-like type (Length, Real, ...) was also registered
   as a simulator parameter with default value 1.0, which replaced the state in the
   numerical equations: the equations were right but the trajectory was silently wrong.
2. The mass-matrix path now extracts M and F without expanding the equations of motion.
   The result must be identical to extraction from the expanded equations.
"""
import numpy as np
import pytest
import sympy as sp

from mechanics_dsl import PhysicsCompiler


SPRING = r"""
\system{spring}
\defvar{x}{%s}{m}
\parameter{m}{1.0}{kg}
\parameter{k}{4.0}{N/m}
\lagrangian{\frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2}
\initial{x=0.1, x_dot=0.0}
"""


@pytest.mark.parametrize("vartype", ["Length", "Position"])
def test_coordinate_is_not_pinned_as_a_parameter(vartype):
    c = PhysicsCompiler()
    assert c.compile_dsl(SPRING % vartype)["success"]
    assert "x" in c.get_coordinates()
    assert "x" not in c.simulator.parameters
    sol = c.simulate((0, 2), num_points=201, rtol=1e-10, atol=1e-12)
    x = np.asarray(sol["y"])[0]
    assert np.abs(x - 0.1 * np.cos(2.0 * np.asarray(sol["t"]))).max() < 1e-6


def chain_source(n):
    """Planar chain of n point masses on links (absolute angles): coupled, >= 4 coordinates."""
    th = [f"th{i}" for i in range(1, n + 1)]
    T, V = [], []
    for k in range(1, n + 1):
        xd = " + ".join(f"\\cos{{{th[j]}}} * \\dot{{{th[j]}}}" for j in range(k))
        yd = " + ".join(f"\\sin{{{th[j]}}} * \\dot{{{th[j]}}}" for j in range(k))
        T.append(f"\\frac{{1}}{{2}} * m * (({xd})^2 + ({yd})^2)")
        V.append("m * g * (" + " + ".join(f"\\cos{{{th[j]}}}" for j in range(k)) + ")")
    lines = [f"\\system{{chain{n}}}"] + [f"\\defvar{{{t}}}{{Angle}}{{rad}}" for t in th]
    lines += ["\\parameter{m}{0.5}{kg}", "\\parameter{g}{9.81}{m/s^2}"]
    lines.append("\\lagrangian{" + " + ".join(T) + " - (" + " + ".join(V) + ")}")
    lines.append("\\initial{" + ", ".join(f"{t}=0.3, {t}_dot=0.0" for t in th) + "}")
    return "\n".join(lines)


def test_mass_matrix_extraction_matches_expanded_form():
    n = 4
    c = PhysicsCompiler()
    assert c.compile_dsl(chain_source(n))["success"]
    eqs = c.symbolic.derive_equations_of_motion(
        c.symbolic.ast_to_sympy(c.lagrangian), c.get_coordinates()
    )
    coords = c.get_coordinates()
    M, F = c.symbolic.mass_matrix_form(eqs, coords)
    expanded, accel = c.symbolic._substitute_derivative_symbols(eqs, coords, expand=True)
    M_ref = sp.Matrix(n, n, lambda i, j: sp.diff(expanded[i], accel[j]))
    F_ref = sp.Matrix([-e.subs([(a, 0) for a in accel]) for e in expanded])
    syms = sorted(M.free_symbols | F.free_symbols | M_ref.free_symbols | F_ref.free_symbols,
                  key=str)
    rng = np.random.default_rng(0)
    for _ in range(3):
        point = {s: rng.uniform(-1, 1) for s in syms}
        assert np.allclose(np.array(M.subs(point), float), np.array(M_ref.subs(point), float),
                           rtol=1e-12, atol=1e-12)
        assert np.allclose(np.array(F.subs(point), float), np.array(F_ref.subs(point), float),
                           rtol=1e-12, atol=1e-12)
