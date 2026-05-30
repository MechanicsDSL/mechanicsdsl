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
