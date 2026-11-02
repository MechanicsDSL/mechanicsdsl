"""
Numerical tests for the code generators: generate, compile, run, compare.

The smoke tests check that every generator writes a plausible file. These tests
check that the generated programs compute the right motion. For each system the
program is built with the local toolchain, run, and its CSV compared with an
independent solution of the compiler's own equations (SciPy DOP853 at rtol 1e-12;
mass-matrix systems are solved with numpy.linalg.solve at every step).

The systems are chosen to catch the faults these generators have had:

* double pendulum: long rational expressions (SymPy's Rust printer dropped the
  brackets round a sum times ``.recip()``).
* forced oscillator: explicit time dependence (some targets did not pass the
  stage time to the right-hand side).
* name clashes: a coordinate called ``y`` (the state array) and parameters
  called ``k2`` (an RK4 stage), ``lambda`` (a Python keyword), ``gamma`` (a C
  library function), ``M`` next to ``m`` (Fortran is case-insensitive) and ``p``.
* six masses: six coordinates, so the compiler keeps M q'' = F unsolved, and
  ``\\defvar`` types that are descriptions rather than registry type names.

A target is skipped when its toolchain is not installed.
"""

import csv
import glob
import os
import re
import shutil
import subprocess  # nosec B404 - test harness, no untrusted input
import sys

import numpy as np
import pytest
import sympy as sp
from scipy.integrate import solve_ivp

from mechanics_dsl import PhysicsCompiler

pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    # the build and run commands below are POSIX shell; on Windows run these under WSL
    pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX shell"),
]

SYSTEMS = {
    "pendulum": (
        r"""\system{pendulum}
\defvar{theta}{Angle from vertical}{rad}
\parameter{m}{1.0}{kg}
\parameter{l}{1.0}{m}
\parameter{g}{9.81}{m/s^2}
\lagrangian{\frac{1}{2}*m*l^2*\dot{theta}^2 - m*g*l*(1-\cos{theta})}
\initial{theta=0.5, theta_dot=0}
""",
        10.0,
    ),
    "double_pendulum": (
        r"""\system{double_pendulum}
\defvar{theta1}{Angle}{rad}
\defvar{theta2}{Angle}{rad}
\parameter{m1}{1.0}{kg}
\parameter{m2}{1.0}{kg}
\parameter{l1}{1.0}{m}
\parameter{l2}{1.0}{m}
\parameter{g}{9.81}{m/s^2}
\lagrangian{\frac{1}{2}*(m1+m2)*l1^2*\dot{theta1}^2 + \frac{1}{2}*m2*l2^2*\dot{theta2}^2
 + m2*l1*l2*\dot{theta1}*\dot{theta2}*\cos{theta1-theta2}
 + (m1+m2)*g*l1*\cos{theta1} + m2*g*l2*\cos{theta2}}
\initial{theta1=2.5, theta2=2.0}
""",
        2.0,  # chaotic: compare over a short window
    ),
    "forced": (
        r"""\system{forced}
\defvar{x}{Position}{m}
\parameter{m}{1.0}{kg}
\parameter{k}{10.0}{N/m}
\parameter{F0}{1.0}{N}
\parameter{omega_d}{2.5}{rad/s}
\lagrangian{\frac{1}{2}*m*\dot{x}^2 - \frac{1}{2}*k*x^2}
\force{F0*\cos{omega_d*t}}
\initial{x=0.2, x_dot=0}
""",
        10.0,
    ),
    "name_clashes": (
        r"""\system{name_clashes}
\defvar{y}{Position}{m}
\defvar{y1}{Position}{m}
\parameter{m}{1.0}{kg}
\parameter{M}{2.0}{kg}
\parameter{k2}{4.0}{N/m}
\parameter{tmp}{1.5}{N/m}
\parameter{lambda}{0.7}{N/m}
\parameter{gamma}{0.3}{N/m}
\parameter{p}{0.2}{N/m}
\lagrangian{\frac{1}{2}*m*\dot{y}^2 + \frac{1}{2}*M*\dot{y1}^2 - \frac{1}{2}*k2*y^2
 - \frac{1}{2}*tmp*y1^2 - \frac{1}{2}*lambda*(y-y1)^2 - \frac{1}{4}*gamma*y^4 - p*y1^3}
\initial{y=0.5, y1=-0.2, y_dot=0.1}
""",
        10.0,
    ),
    "six_masses": (
        r"""\system{six_masses}
\defvar{x1}{Position of mass 1}{m}
\defvar{x2}{Position of mass 2}{m}
\defvar{x3}{Position of mass 3}{m}
\defvar{x4}{Position of mass 4}{m}
\defvar{x5}{Position of mass 5}{m}
\defvar{x6}{Position of mass 6}{m}
\parameter{m}{1.0}{kg}
\parameter{k}{3.0}{N/m}
\lagrangian{\frac{1}{2}*m*(\dot{x1}^2+\dot{x2}^2+\dot{x3}^2+\dot{x4}^2+\dot{x5}^2+\dot{x6}^2)
 - \frac{1}{2}*k*((x2-x1)^2+(x3-x2)^2+(x4-x3)^2+(x5-x4)^2+(x6-x5)^2+x1^2+x6^2)}
\initial{x1=0.3, x4=-0.2}
""",
        10.0,
    ),
}

ARDUINO_SHIM = r"""
#include <cstdio>
#include <cmath>
struct SerialShim {
  void begin(long) {}
  explicit operator bool() const { return true; }
  void print(const char* s) { std::printf("%s", s); }
  void print(double v, int d = 2) { std::printf("%.*f", d, v); }
  void println() { std::printf("\n"); }
  void println(const char* s) { std::printf("%s\n", s); }
} Serial;
static unsigned long fake_us = 0;  // the clock moves on 10 ms each time it is read
unsigned long micros() { fake_us += 10000UL; return fake_us; }
unsigned long millis() { return micros() / 1000UL; }
"""

WASM_MAIN = r"""
#include <stdio.h>
#include "%s"
int main(void) {
    FILE* f = fopen("out.csv", "w");
    fprintf(f, "t%s\n");
    init();
    set_dt(0.001);
    for (int n = 0; n <= 10000; n++) {
        if (n %% 10 == 0) {
            fprintf(f, "%%.10f", get_time());
            for (int i = 0; i < STATE_DIM; i++) fprintf(f, ",%%.15g", get_state(i));
            fprintf(f, "\n");
        }
        step(1);
    }
    fclose(f);
    return 0;
}
"""


def _openmp_works(tmp_path):
    """Apple clang (macOS's g++) has no -fopenmp; GCC does."""
    src = tmp_path / "omp_probe.cpp"
    src.write_text("#include <omp.h>\nint main() { return omp_get_max_threads() > 0 ? 0 : 1; }\n")
    p = subprocess.run(  # nosec B603 B607 - fixed probe command
        ["g++", "-fopenmp", str(src), "-o", str(tmp_path / "omp_probe")], capture_output=True
    )
    return p.returncode == 0


def _have(*tools):
    return all(shutil.which(t) for t in tools)


# target -> (needed tools, float32 program?)
TARGETS = {
    "cpp": (("g++",), False),
    "openmp": (("g++",), False),
    "cuda": (("g++",), False),  # the CPU reference file generated next to the .cu
    "python": ((), False),
    "rust": (("rustc",), False),
    "fortran": (("gfortran",), False),
    "javascript": (("node",), False),
    "wasm": (("gcc",), False),  # the C source, compiled natively
    "arm": (("gcc",), False),  # plain C; the NEON block is behind #ifdef
    "arduino": (("g++",), True),
    "matlab": (("octave",), False),
}
EXT = {
    "cpp": ".cpp",
    "openmp": ".cpp",
    "cuda": "",
    "python": ".py",
    "rust": ".rs",
    "fortran": ".f90",
    "javascript": ".js",
    "wasm": "",
    "arm": ".c",
    "arduino": ".ino",
    "matlab": ".m",
}


def _run(cmd, cwd):
    p = subprocess.run(  # nosec B602 - fixed commands built by this test
        cmd, cwd=cwd, shell=True, capture_output=True, text=True, timeout=180
    )
    assert p.returncode == 0, f"`{cmd}` failed:\n{(p.stdout + p.stderr)[-3000:]}"


def _build_and_run(target, work, name, coords):
    if target == "cpp":
        cmds = [f"g++ -std=c++17 -O2 -o prog {name}.cpp", "./prog"]
    elif target == "openmp":
        cmds = [f"g++ -std=c++17 -O2 -fopenmp -o prog {name}.cpp", "OMP_NUM_THREADS=2 ./prog"]
    elif target == "cuda":
        cpu = glob.glob(os.path.join(work, name, "*_cpu.cpp"))
        assert cpu, "no CPU reference file generated"
        cmds = [f"g++ -std=c++17 -O2 -I{name} -o prog '{os.path.relpath(cpu[0], work)}'", "./prog"]
    elif target == "python":
        script = f"import {name} as m; m.export_csv(m.simulate())"
        cmds = [f"MPLBACKEND=Agg '{sys.executable}' -c \"{script}\""]
    elif target == "rust":
        cmds = [f"rustc -O -o prog {name}.rs", "./prog"]
    elif target == "fortran":
        cmds = [f"gfortran -O2 -o prog {name}.f90", "./prog"]
    elif target == "javascript":
        cmds = [f"node {name}.js"]
    elif target == "wasm":
        os.makedirs(os.path.join(work, "stub"))
        with open(os.path.join(work, "stub", "emscripten.h"), "w") as fh:
            fh.write("#define EMSCRIPTEN_KEEPALIVE\n")
        with open(os.path.join(work, "main.c"), "w") as fh:
            cols = "".join(f",{c},{c}_dot" for c in coords)
            fh.write(WASM_MAIN % (f"{name}/{name}.c", cols))
        cmds = ["gcc -std=gnu11 -O2 -Istub -o prog main.c -lm", "./prog"]
    elif target == "arm":
        cmds = [f"gcc -std=gnu11 -O2 -o prog {name}.c -lm", "./prog"]
    elif target == "arduino":
        with open(os.path.join(work, "main.cpp"), "w") as fh:
            fh.write(ARDUINO_SHIM + f'#include "{name}.ino"\n')
            fh.write("int main() { setup(); for (int i = 0; i < 1001; ++i) loop(); }\n")
        cmds = ["g++ -std=c++17 -O2 -x c++ -o prog main.cpp", "./prog > serial.csv"]
    elif target == "matlab":
        path = os.path.join(work, f"{name}.m")
        with open(path) as fh:
            text = fh.read().replace("config.plot_results = true;", "config.plot_results = false;")
        with open(path, "w") as fh:
            fh.write(text)
        cmds = [f"octave --no-gui --no-window-system --quiet --eval '{name};'"]
    for cmd in cmds:
        _run(cmd, work)

    out = glob.glob(os.path.join(work, "*.csv"))
    assert out, "the program wrote no CSV"
    with open(max(out, key=os.path.getmtime)) as fh:
        rows = [r for r in csv.reader(fh) if r]
    start = next(i for i, r in enumerate(rows) if re.match(r"\s*t(ime)?\s*$", r[0], re.I))
    header = [h.strip().lower() for h in rows[start]]
    data = np.array([[float(x) for x in r[: len(header)]] for r in rows[start + 1 :]])
    return header, data


def _reference(comp, coords, t_eval):
    E = comp.equations
    params = {k: float(v["value"]) for k, v in comp.parameters_def.items()}
    mass = isinstance(E, dict) and "__mass_matrix__" in E
    exprs = list(E["__mass_matrix__"]) if mass else [E[f"{q}_ddot"] for q in coords]
    syms = {s.name: s for e in exprs for s in e.free_symbols}
    get = lambda n: syms.get(n, sp.Symbol(n))  # noqa: E731
    state = [get(n) for q in coords for n in (q, f"{q}_dot")] + [get("t")]
    subs = {get(k): v for k, v in params.items()}
    if mass:
        Mf = sp.lambdify(state, exprs[0].subs(subs), "numpy")
        Ff = sp.lambdify(state, exprs[1].subs(subs), "numpy")

        def acc(y, t):
            return np.linalg.solve(np.array(Mf(*y, t), float), np.array(Ff(*y, t), float).ravel())

    else:
        fs = [sp.lambdify(state, e.subs(subs), "numpy") for e in exprs]

        def acc(y, t):
            return np.array([f(*y, t) for f in fs], float)

    def rhs(t, y):
        out = np.empty_like(y)
        out[0::2] = y[1::2]
        out[1::2] = acc(y, t)
        return out

    ic = comp.initial_conditions
    y0 = [float(ic.get(n, 0.0)) for q in coords for n in (q, f"{q}_dot")]
    sol = solve_ivp(
        rhs, (0, t_eval[-1]), y0, method="DOP853", rtol=1e-12, atol=1e-12, t_eval=t_eval
    )
    return {q: sol.y[2 * i] for i, q in enumerate(coords)}


@pytest.mark.parametrize("system", sorted(SYSTEMS))
@pytest.mark.parametrize("target", sorted(TARGETS))
def test_generated_program_matches_reference(system, target, tmp_path):
    tools, float32 = TARGETS[target]
    if not _have(*tools):
        pytest.skip(f"{', '.join(tools)} not installed")
    if target == "openmp" and not _openmp_works(tmp_path):
        pytest.skip("this C++ compiler does not support OpenMP")
    source, window = SYSTEMS[system]
    comp = PhysicsCompiler()
    result = comp.compile_dsl(source)
    assert result["success"], result.get("error")
    coords = comp.get_coordinates()
    name = comp.system_name
    work = str(tmp_path)
    comp.export(target, os.path.join(work, name + EXT[target]))

    header, data = _build_and_run(target, work, name, coords)
    t = data[:, 0]
    keep = (t >= 0) & (t <= window + 1e-9)
    assert keep.sum() > 20, f"only {keep.sum()} output rows"
    ref = _reference(comp, coords, t[keep])

    tol = 5e-3 if float32 else 2e-4
    for q in coords:
        cols = [j for j, h in enumerate(header) if h.rstrip("_") == q.lower()]
        assert cols, f"no column for {q} in {header}"
        got = data[keep, cols[0]]
        scale = max(1.0, float(np.max(np.abs(ref[q]))))
        err = float(np.max(np.abs(got - ref[q]))) / scale
        assert err <= tol, f"{q}: max error {err:.2e} > {tol:.0e}"
