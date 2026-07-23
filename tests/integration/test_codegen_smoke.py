"""
Smoke tests for every code generator shipped under mechanics_dsl.codegen.

For each backend we:
  1. Generate code from a known-good pendulum system.
  2. Assert the output file is non-trivial.
  3. Assert it contains a sentinel referencing the system (proves the
     generator actually produced material output, not a stub).
  4. Assert it does NOT contain ``MECHANICSDSL_CODEGEN_FAILED`` — that's
     the marker introduced in v2.1.1 for symbolic-to-target conversion
     failures, and any occurrence here means the generator silently
     emitted broken output for the pendulum baseline.

Where a target compiler / interpreter is locally available (Python, Node,
g++) and we can run it cheaply, we also do a syntax check.
"""

import ast
import os
import shutil
import subprocess  # nosec B404 - test harness, no untrusted input
import tempfile

import pytest

from mechanics_dsl import PhysicsCompiler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pendulum_compiler():
    """A compiled simple pendulum system shared across all generator tests."""
    compiler = PhysicsCompiler()
    result = compiler.compile_dsl(
        r"\system{pendulum_smoke}"
        r"\defvar{theta}{Angle}{rad}"
        r"\parameter{m}{1.0}{kg}\parameter{l}{1.0}{m}\parameter{g}{9.81}{m/s^2}"
        r"\lagrangian{0.5*m*l^2*\dot{theta}^2 - m*g*l*(1 - \cos{theta})}"
    )
    assert result["success"], result
    return compiler


# ---------------------------------------------------------------------------
# Per-generator smoke
# ---------------------------------------------------------------------------


GENERATORS = [
    # (target key used by PhysicsCompiler.export, expected extension)
    ("arduino", ".ino"),
    ("arm", ".c"),
    ("cpp", ".cpp"),
    ("cuda", ".cu"),
    ("fortran", ".f90"),
    ("javascript", ".js"),
    ("julia", ".jl"),
    ("matlab", ".m"),
    ("openmp", ".cpp"),
    ("python", ".py"),
    ("rust", ".rs"),
    ("wasm", ".c"),
]


@pytest.mark.parametrize("target,extension", GENERATORS, ids=[g[0] for g in GENERATORS])
def test_generator_produces_nontrivial_output(pendulum_compiler, target, extension):
    """Every codegen target writes a non-empty file referencing the system."""
    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as f:
        path = f.name
    out_path = path  # default for cleanup if export() raises
    try:
        out_path = pendulum_compiler.export(target, path)
        assert os.path.exists(out_path), f"{target} produced no file"
        size = os.path.getsize(out_path)
        assert size > 200, f"{target} output suspiciously small ({size} bytes)"

        with open(out_path, "r", encoding="utf-8") as fh:
            source = fh.read()

        # Sentinel from v2.1.1 - any occurrence means symbolic-to-target
        # conversion silently fell back to a stub.
        assert "MECHANICSDSL_CODEGEN_FAILED" not in source, (
            f"{target} emitted codegen-failure marker; symbolic conversion "
            f"silently failed."
        )

        # The system name should appear somewhere in the generated code so
        # we know we got real output, not boilerplate.
        assert "pendulum_smoke" in source.lower() or "pendulum" in source.lower(), (
            f"{target} output doesn't reference the system name; likely a stub."
        )
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Where the host tooling is available, syntax-check the output
# ---------------------------------------------------------------------------


def test_python_output_parses_as_python(pendulum_compiler):
    """The Python generator's output must at least be syntactically valid."""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        path = f.name
    try:
        out = pendulum_compiler.export("python", path)
        with open(out, "r", encoding="utf-8") as fh:
            source = fh.read()
        ast.parse(source)  # raises SyntaxError on bad output
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_python_output_runs_and_simulates():
    """
    Behavioral check: the generated Python must actually *execute* and produce
    a physically sensible pendulum trajectory.

    A syntax-only check (ast.parse) passes even when transcendental calls are
    emitted unqualified (bare ``sin(theta)``), which raises NameError at
    runtime. Running the module is what catches that.
    """
    import runpy

    import numpy as np

    # Own compiler WITH initial conditions so the generated script actually
    # releases the bob from theta0 = 0.5 rad.
    compiler = PhysicsCompiler()
    result = compiler.compile_dsl(
        r"\system{pendulum_run}"
        r"\defvar{theta}{Angle}{rad}"
        r"\parameter{m}{1.0}{kg}\parameter{l}{1.0}{m}\parameter{g}{9.81}{m/s^2}"
        r"\lagrangian{0.5*m*l^2*\dot{theta}^2 - m*g*l*(1 - \cos{theta})}"
        r"\initial{theta=0.5, theta_dot=0.0}"
    )
    assert result["success"], result

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        path = f.name
    try:
        out = compiler.export("python", path)
        # Execute the generated module in isolation (it must not depend on
        # mechanics_dsl being importable).
        ns = runpy.run_path(out, run_name="_generated_pendulum")
        assert "simulate" in ns, "generated module exposes no simulate()"

        sol = ns["simulate"](t_span=(0, 5), num_points=200)
        theta = np.asarray(sol.y[0])

        assert np.all(np.isfinite(theta)), "generated simulation produced NaN/Inf"
        # Undamped pendulum released from rest at theta0=0.5 must oscillate:
        # it should swing to roughly -theta0 and stay bounded by it.
        assert theta.min() < -0.3, f"pendulum did not swing back (min {theta.min():.3f})"
        assert np.max(np.abs(theta)) < 0.6, f"energy not conserved (max |theta| {np.max(np.abs(theta)):.3f})"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_javascript_output_passes_node_syntax_check(pendulum_compiler):
    """If Node is installed, the JS output must pass `node --check`."""
    with tempfile.NamedTemporaryFile(suffix=".js", delete=False) as f:
        path = f.name
    try:
        out = pendulum_compiler.export("javascript", path)
        result = subprocess.run(  # nosec B603 - test harness
            ["node", "--check", out],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, (
            f"node --check failed for generated JS:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not on PATH")
def test_cpp_output_passes_compiler_syntax_check(pendulum_compiler):
    """If g++ is installed, the C++ output must compile with -fsyntax-only."""
    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as f:
        path = f.name
    try:
        out = pendulum_compiler.export("cpp", path)
        result = subprocess.run(  # nosec B603 - test harness
            ["g++", "-std=c++17", "-fsyntax-only", out],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"g++ -fsyntax-only failed for generated C++:\n"
            f"stderr: {result.stderr}"
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@pytest.fixture(scope="module")
def xy_compiler():
    """A 2D system whose coordinate 'y' collides with the conventional state
    vector name (the case that broke the C-family generators)."""
    compiler = PhysicsCompiler()
    result = compiler.compile_dsl(
        r"\system{proj}"
        r"\defvar{x}{Position}{m}\defvar{y}{Position}{m}"
        r"\parameter{m}{1.0}{kg}\parameter{g}{9.81}{m/s^2}"
        r"\lagrangian{0.5*m*(\dot{x}^2 + \dot{y}^2) - m*g*y}"
    )
    assert result["success"], result
    return compiler


@pytest.mark.parametrize("target,ext", [("cpp", ".cpp"), ("openmp", ".cpp")])
def test_state_array_not_shadowed_by_coordinate(xy_compiler, target, ext):
    """A coordinate named 'y' must not be unpacked from an array also named 'y'
    (``double y = y[...]`` shadows the array). The generator must alias the
    array to a free name first."""
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        path = f.name
    try:
        out = xy_compiler.export(target, path)
        with open(out, "r", encoding="utf-8") as fh:
            source = fh.read()
        assert "double y = y[" not in source, (
            f"{target}: coordinate 'y' unpacked from array 'y' (self-shadow)"
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _gxx_accepts(flags):
    """Whether the `g++` on PATH accepts these flags on a trivial program.

    On macOS `g++` is really Apple clang, which rejects `-fopenmp`
    ('unsupported option'); such environments should skip the OpenMP check
    rather than fail it.
    """
    try:
        probe = subprocess.run(  # nosec B603 - test harness
            ["g++", "-std=c++17", "-fsyntax-only", *flags, "-xc++", "-"],
            input="int main(){return 0;}",
            capture_output=True,
            text=True,
            timeout=30,
        )
        return probe.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not on PATH")
@pytest.mark.parametrize("target,flags", [("cpp", []), ("openmp", ["-fopenmp"])])
def test_y_coordinate_output_compiles(xy_compiler, target, flags):
    """If g++ is installed, a system with a coordinate named 'y' must still
    compile (regression for the state-array shadowing bug)."""
    if flags and not _gxx_accepts(flags):
        pytest.skip(f"g++ does not accept {flags} (e.g. Apple clang without OpenMP)")
    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as f:
        path = f.name
    try:
        out = xy_compiler.export(target, path)
        result = subprocess.run(  # nosec B603 - test harness
            ["g++", "-std=c++17", "-fsyntax-only", *flags, out],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"g++ -fsyntax-only failed for {target} with a 'y' coordinate:\n"
            f"stderr: {result.stderr}"
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
