"""
Bare-metal ARM code generation for mass-matrix systems, and integral exponents.

Systems with four or more coupled coordinates are kept by the compiler as
M(q, q_dot) q_ddot = F(q, q_dot). The embedded ARM target now evaluates M and F
on the device and solves for the accelerations, so such systems can be
generated. The reference used here is a planar chain written directly in NumPy;
it shares no code with MechanicsDSL.
"""

import shutil
import subprocess
import textwrap

import numpy as np
import pytest
import sympy as sp
from scipy.integrate import solve_ivp

from mechanics_dsl import PhysicsCompiler
from mechanics_dsl.codegen.arm import ARMGenerator
from mechanics_dsl.parser import MechanicsParser, tokenize
from mechanics_dsl.symbolic import SymbolicEngine
from mechanics_dsl.utils.config import config

N = 4
P = {"m": 0.05, "l": 0.1, "k": 1.0, "c": 2.0e-4, "g": 9.81}
THETA0 = [0.6 * (i + 0.5) / N for i in range(N)]

HARNESS = textwrap.dedent("""
    #include <stdio.h>
    #include <stdlib.h>
    void physics_step(float dt);
    float get_state(int idx);
    int physics_singular_events(void);
    int main(int argc, char** argv) {
        int dim = atoi(argv[1]); float dt = (float)atof(argv[2]);
        int steps = atoi(argv[3]); int every = atoi(argv[4]);
        for (int s = 0; s <= steps; s++) {
            if (s % every == 0) {
                for (int i = 0; i < dim; i++) printf(i ? ",%.9g" : "%.9g", get_state(i));
                printf("\\n");
            }
            if (s < steps) physics_step(dt);
        }
        fprintf(stderr, "%d\\n", physics_singular_events());
        return 0;
    }
    """)


def chain_source(n=N):
    """Planar chain of n links, absolute angles from the upward vertical, point
    mass at each link tip, torsional spring and damper at each joint."""
    th = [f"th{i}" for i in range(1, n + 1)]
    T, V, R = [], [], []
    for k in range(1, n + 1):
        xd = " + ".join(f"l * \\cos{{{th[j]}}} * \\dot{{{th[j]}}}" for j in range(k))
        yd = " + ".join(f"l * \\sin{{{th[j]}}} * \\dot{{{th[j]}}}" for j in range(k))
        T.append(f"\\frac{{1}}{{2}} * m * (({xd})^2 + ({yd})^2)")
        V.append("m * g * (" + " + ".join(f"l * \\cos{{{th[j]}}}" for j in range(k)) + ")")
    for i in range(n):
        rel = th[i] if i == 0 else f"({th[i]} - {th[i - 1]})"
        V.append(f"\\frac{{1}}{{2}} * k * {rel}^2")
        drel = f"\\dot{{{th[i]}}}" if i == 0 else f"(\\dot{{{th[i]}}} - \\dot{{{th[i - 1]}}})"
        R.append(f"\\frac{{1}}{{2}} * c * {drel}^2")
    lines = ["\\system{chain}"]
    lines += [f"\\defvar{{{t}}}{{Angle}}{{rad}}" for t in th]
    lines += [f"\\parameter{{{k}}}{{{v!r}}}{{1}}" for k, v in P.items()]
    lines.append("\\lagrangian{" + " + ".join(T) + " - (" + " + ".join(V) + ")}")
    lines.append("\\rayleigh{" + " + ".join(R) + "}")
    lines.append(
        "\\initial{" + ", ".join(f"{t}={v!r}, {t}_dot=0.0" for t, v in zip(th, THETA0)) + "}"
    )
    return "\n".join(lines)


def reference(t_eval, n=N):
    m, ell, k, c, g = (P[x] for x in ("m", "l", "k", "c", "g"))
    S = np.array([m * (n - i) for i in range(n)])
    Smax = S[np.maximum.outer(np.arange(n), np.arange(n))]
    D = np.eye(n) - np.eye(n, k=-1)

    def f(_, z):
        th, w = z[:n], z[n:]
        d = th[:, None] - th[None, :]
        M = ell * ell * np.cos(d) * Smax
        rhs = (-ell * ell * np.sin(d) * Smax) @ w**2 + g * ell * np.sin(th) * S
        rhs -= k * D.T @ D @ th + c * D.T @ D @ w
        return np.concatenate([w, np.linalg.solve(M, rhs)])

    z0 = np.concatenate([THETA0, np.zeros(n)])
    s = solve_ivp(
        f, (t_eval[0], t_eval[-1]), z0, t_eval=t_eval, method="DOP853", rtol=1e-12, atol=1e-13
    )
    return s.y


@pytest.fixture(scope="module")
def chain():
    old = config.enable_simplification
    config.enable_simplification = False  # keeps the test fast; equations are identical
    try:
        c = PhysicsCompiler()
        r = c.compile_dsl(chain_source())
        assert r["success"], r.get("error")
        yield c
    finally:
        config.enable_simplification = old


def _run(cfile, tmp_path, dim, dt, steps, every):
    exe = str(tmp_path / "sim")
    harness = tmp_path / "harness.c"
    harness.write_text(HARNESS)
    subprocess.run(["gcc", "-O2", "-o", exe, cfile, str(harness)], check=True)
    p = subprocess.run(
        [exe, str(dim), repr(dt), str(steps), str(every)],
        capture_output=True,
        text=True,
        check=True,
    )
    y = np.loadtxt(p.stdout.splitlines(), delimiter=",", ndmin=2).T
    return y, int(p.stderr.strip())


class TestIntegralExponents:
    def test_integer_power_is_integer(self):
        ast = MechanicsParser(tokenize(r"\lagrangian{(x * \dot{x})^2 + y^3 + z^0.5}")).parse()
        e = SymbolicEngine().ast_to_sympy(ast[0].expr)
        exps = {a.exp for a in e.atoms(sp.Pow)}
        assert sp.Integer(2) in exps and sp.Integer(3) in exps
        assert any(isinstance(x, sp.Float) for x in exps)  # 0.5 stays a float

    def test_mass_matrix_finite_at_rest(self, chain):
        # Before the fix, x^2 became x**2.0 and the unsimplified mass matrix of a
        # chain held terms like (v*s)**2.0/v**2: 0/0 for a system released from rest.
        M, F = chain.equations["__mass_matrix__"]
        at_rest = {sp.Symbol(f"th{i}_dot", real=True): 0 for i in range(1, N + 1)}
        at_rest.update({sp.Symbol(f"th{i}", real=True): v for i, v in enumerate(THETA0, 1)})
        at_rest.update({sp.Symbol(k, real=True): v for k, v in P.items()})
        Mn = np.array(M.subs(at_rest).evalf(), dtype=float)
        assert np.all(np.isfinite(Mn))

    def test_simulation_matches_reference(self, chain):
        t = np.linspace(0.0, 1.0, 101)
        sol = chain.simulate((0, 1.0), num_points=101, method="DOP853", rtol=1e-10, atol=1e-12)
        assert sol["success"]
        np.testing.assert_allclose(sol["y"][0::2], reference(t)[:N], atol=1e-8)


class TestEmbeddedMassMatrix:
    def test_generates_solver(self, chain, tmp_path):
        out = chain.export(
            "arm",
            str(tmp_path / "chain.c"),
            generator_options={"target": "cortex_m", "embedded": True},
        )
        code = open(out).read()
        assert "mdsl_solve(M, a);" in code
        assert "physics_singular_events" in code
        assert "MECHANICSDSL_" not in code
        for i in range(N):
            assert f"dydt[{2 * i + 1}] = a[{i}];" in code

    @pytest.mark.skipif(shutil.which("gcc") is None, reason="no host C compiler")
    def test_generated_c_matches_reference(self, chain, tmp_path):
        out = chain.export(
            "arm",
            str(tmp_path / "chain.c"),
            generator_options={"target": "cortex_m", "embedded": True},
        )
        dt, steps, every = 1e-4, 10000, 100
        y, singular = _run(out, tmp_path, 2 * N, dt, steps, every)
        assert singular == 0
        ref = reference(np.linspace(0.0, steps * dt, steps // every + 1))
        # float32 state with RK4: agreement to about 1e-6 rad
        assert np.abs(y[0::2] - ref[:N]).max() < 1e-4

    @pytest.mark.skipif(shutil.which("arm-none-eabi-gcc") is None, reason="no ARM cross-compiler")
    def test_cross_compiles_clean(self, chain, tmp_path):
        out = chain.export(
            "arm",
            str(tmp_path / "chain.c"),
            generator_options={"target": "cortex_m", "embedded": True},
        )
        p = subprocess.run(
            [
                "arm-none-eabi-gcc",
                "-mcpu=cortex-m4",
                "-mthumb",
                "-mfpu=fpv4-sp-d16",
                "-mfloat-abi=hard",
                "-Os",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-c",
                out,
                "-o",
                str(tmp_path / "chain.o"),
            ],
            capture_output=True,
            text=True,
        )
        assert p.returncode == 0, p.stderr

    def test_non_embedded_still_requires_closed_form(self, chain, tmp_path):
        with pytest.raises(ValueError, match="coupled"):
            chain.export("arm", str(tmp_path / "chain.c"))

    def test_mass_matrix_requires_embedded(self):
        q, qd = sp.symbols("q q_dot", real=True)
        gen = ARMGenerator(
            "one", ["q"], {}, {"q": 0.0}, {}, mass_matrix=(sp.Matrix([[1]]), sp.Matrix([[-q]]))
        )
        ok, errors = gen.validate()
        assert not ok and any("embedded=True" in e for e in errors)

    def test_shape_checked(self):
        with pytest.raises(ValueError, match="mass_matrix"):
            ARMGenerator(
                "bad", ["q"], {}, {}, {}, embedded=True, mass_matrix=(sp.eye(2), sp.zeros(2, 1))
            )

    @pytest.mark.skipif(shutil.which("gcc") is None, reason="no host C compiler")
    def test_singular_matrix_is_reported(self, tmp_path):
        q = sp.Symbol("q", real=True)
        gen = ARMGenerator(
            "singular",
            ["q"],
            {},
            {"q": 1.0, "q_dot": 0.0},
            {},
            target="cortex_m",
            embedded=True,
            mass_matrix=(sp.Matrix([[0]]), sp.Matrix([[-q]])),
        )
        out = gen.generate(str(tmp_path / "singular.c"))
        y, singular = _run(out, tmp_path, 2, 1e-3, 1, 1)
        assert singular > 0
        assert np.isnan(y[:, -1]).any()
