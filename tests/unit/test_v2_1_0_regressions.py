"""
Regression tests pinning the correctness/security fixes shipped in v2.1.0.

Each test maps to a specific finding from the v2.1.0 review pass; touching
that file should make you re-read the corresponding test before changing it.
"""

import logging
import os
import pickle
import tempfile

import pytest

from mechanics_dsl import PhysicsCompiler
from mechanics_dsl.io.serialization import SystemSerializer, deserialize_solution
from mechanics_dsl.parser import tokenize

# Silence the noisy compiler/solver logging during property-style tests.
logging.disable(logging.CRITICAL)


# ---------------------------------------------------------------------------
# Fix #1: compile_dsl resets per-system state between calls.
# ---------------------------------------------------------------------------


class TestCompilerStateReset:
    """A single PhysicsCompiler reused across compile_dsl() calls must not
    carry variables / constraints / forces / initial conditions over from a
    previously compiled system."""

    def test_no_state_leak_between_compiles(self):
        compiler = PhysicsCompiler()
        compiler.compile_dsl(
            r"\system{a}\defvar{x}{Position}{m}"
            r"\constraint{x}\force{1.0}\lagrangian{\dot{x}^2}"
        )
        assert "x" in compiler.variables
        assert len(compiler.constraints) == 1
        assert len(compiler.forces) == 1

        compiler.compile_dsl(
            r"\system{b}\defvar{y}{Position}{m}\lagrangian{\dot{y}^2}"
        )

        assert list(compiler.variables) == ["y"]
        assert compiler.constraints == []
        assert compiler.forces == []
        assert compiler.get_coordinates() == ["y"]
        assert compiler.system_name == "b"

    def test_simulator_state_cleared(self):
        compiler = PhysicsCompiler()
        compiler.compile_dsl(
            r"\system{a}\defvar{x}{Position}{m}\parameter{k}{7.0}{1}"
            r"\lagrangian{0.5*\dot{x}^2 - 0.5*k*x^2}"
        )
        assert compiler.simulator.parameters.get("k") == 7.0

        compiler.compile_dsl(
            r"\system{b}\defvar{y}{Position}{m}\lagrangian{\dot{y}^2}"
        )

        assert "k" not in compiler.simulator.parameters


# ---------------------------------------------------------------------------
# Fix #2: PhysicsCompiler.export() actually exists and produces files.
# ---------------------------------------------------------------------------


class TestPhysicsCompilerExport:
    """Previously the server /export route called compiler.export() which
    did not exist. The method now exists and dispatches to the codegen
    backends."""

    @pytest.fixture
    def compiled_pendulum(self):
        compiler = PhysicsCompiler()
        compiler.compile_dsl(
            r"\system{pendulum}"
            r"\defvar{theta}{Angle}{rad}"
            r"\parameter{m}{1.0}{kg}\parameter{l}{1.0}{m}\parameter{g}{9.81}{m/s^2}"
            r"\lagrangian{0.5*m*l^2*\dot{theta}^2 - m*g*l*(1 - \cos{theta})}"
        )
        return compiler

    @pytest.mark.parametrize(
        "target,extension",
        [("python", "py"), ("cpp", "cpp"), ("rust", "rs"), ("javascript", "js")],
    )
    def test_export_writes_generated_file(self, compiled_pendulum, target, extension):
        with tempfile.NamedTemporaryFile(suffix=f".{extension}", delete=False) as f:
            path = f.name
        try:
            out = compiled_pendulum.export(target, path)
            assert os.path.exists(out)
            assert os.path.getsize(out) > 0
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def test_export_rejects_unknown_target(self, compiled_pendulum):
        with pytest.raises(ValueError, match="Unsupported export target"):
            compiled_pendulum.export("notalanguage", "/tmp/x")

    def test_export_requires_compiled_equations(self):
        compiler = PhysicsCompiler()
        with pytest.raises(ValueError, match="No equations derived"):
            compiler.export("python", "/tmp/x.py")


# ---------------------------------------------------------------------------
# Fix #3: \force{coord}{expr} targets a named generalized coordinate;
# the legacy \force{expr} still applies positionally.
# ---------------------------------------------------------------------------


class TestForceCoordinateTargeting:
    def test_force_targets_named_coordinate(self):
        compiler = PhysicsCompiler()
        result = compiler.compile_dsl(
            r"\system{two_dof}"
            r"\defvar{x}{Position}{m}\defvar{y}{Position}{m}"
            r"\parameter{m}{1.0}{kg}"
            r"\lagrangian{0.5*m*\dot{x}^2 + 0.5*m*\dot{y}^2}"
            r"\force{y}{3.0}"
        )
        assert result["success"]
        # Force was meant for y, not x.
        assert str(compiler.equations["x_ddot"]) == "0.0"
        assert "3.0" in str(compiler.equations["y_ddot"])

    def test_legacy_positional_force_still_works(self):
        compiler = PhysicsCompiler()
        result = compiler.compile_dsl(
            r"\system{one_dof}"
            r"\defvar{x}{Position}{m}\parameter{m}{1.0}{kg}"
            r"\lagrangian{0.5*m*\dot{x}^2}"
            r"\force{5.0}"
        )
        assert result["success"]
        assert "5.0" in str(compiler.equations["x_ddot"])

    def test_unknown_target_coordinate_is_skipped_with_warning(self, caplog):
        compiler = PhysicsCompiler()
        # Hypothesis: a force targeting a coord that doesn't exist should
        # neither blow up the compile nor silently affect another coord.
        logging.disable(logging.NOTSET)  # re-enable for caplog
        try:
            with caplog.at_level(logging.WARNING):
                result = compiler.compile_dsl(
                    r"\system{x_only}"
                    r"\defvar{x}{Position}{m}\parameter{m}{1.0}{kg}"
                    r"\lagrangian{0.5*m*\dot{x}^2}"
                    r"\force{z}{42.0}"
                )
            assert result["success"]
            assert "42.0" not in str(compiler.equations["x_ddot"])
            assert any("unknown coordinate" in rec.message for rec in caplog.records)
        finally:
            logging.disable(logging.CRITICAL)


# ---------------------------------------------------------------------------
# Fix #4: compile_dsl exposes solver/simulator diagnostics via result["warnings"].
# ---------------------------------------------------------------------------


class TestCompilationWarnings:
    def test_warnings_empty_on_clean_compile(self):
        compiler = PhysicsCompiler()
        result = compiler.compile_dsl(
            r"\system{p}"
            r"\defvar{theta}{Angle}{rad}"
            r"\parameter{m}{1.0}{kg}\parameter{l}{1.0}{m}\parameter{g}{9.81}{m/s^2}"
            r"\lagrangian{0.5*m*l^2*\dot{theta}^2 - m*g*l*(1 - \cos{theta})}"
        )
        assert result["success"]
        assert result["warnings"] == []

    def test_warnings_surface_solve_failures(self):
        # A constrained 1-DOF Lagrangian where the multiplier can't be solved
        # for its acceleration produces a "No solution found for lambda_0_ddot"
        # diagnostic. Previously this was swallowed; now it must appear.
        compiler = PhysicsCompiler()
        result = compiler.compile_dsl(
            r"\system{a}\defvar{x}{Position}{m}"
            r"\constraint{x}\lagrangian{\dot{x}^2}"
        )
        assert result["success"]
        assert isinstance(result["warnings"], list)
        assert any("lambda" in w or "fallback" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# Fix #6: pickle deserialization is refused by default.
# ---------------------------------------------------------------------------


class TestPickleOptIn:
    @pytest.fixture
    def pickle_file(self):
        data = {"key": "value", "n": 42}
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            pickle.dump(data, f)
            path = f.name
        yield path, data
        try:
            os.unlink(path)
        except OSError:
            pass

    def test_load_pickle_refuses_by_default(self, pickle_file):
        path, _ = pickle_file
        assert SystemSerializer.load_pickle(path) is None
        assert SystemSerializer.load_pickle(path, allow_pickle=False) is None

    def test_load_pickle_allows_with_opt_in(self, pickle_file):
        path, data = pickle_file
        assert SystemSerializer.load_pickle(path, allow_pickle=True) == data

    def test_deserialize_solution_refuses_pickle_by_default(self, pickle_file):
        path, _ = pickle_file
        # Auto-detected as pickle from .pkl extension.
        assert deserialize_solution(path) is None

    def test_deserialize_solution_allows_pickle_when_opted_in(self, pickle_file):
        path, data = pickle_file
        assert deserialize_solution(path, allow_pickle=True) == data

    def test_import_system_refuses_pickle_by_default(self, pickle_file):
        # PhysicsCompiler.import_system is the public entry point used in the
        # README; it must also refuse pickle without the explicit opt-in.
        path, _ = pickle_file
        assert PhysicsCompiler.import_system(path) is None

    def test_import_system_allows_pickle_when_opted_in(self, pickle_file):
        path, _ = pickle_file
        # When opted in, the call succeeds and returns a PhysicsCompiler
        # constructed from the pickled state (not the raw dict).
        result = PhysicsCompiler.import_system(path, allow_pickle=True)
        assert result is not None


# ---------------------------------------------------------------------------
# Fix #7: tokenizer rejects unrecognized characters instead of silently
# dropping them.
# ---------------------------------------------------------------------------


class TestTokenizerStrictness:
    def test_clean_dsl_still_tokenizes(self):
        types = [t.type for t in tokenize(r"\lagrangian{T - V}")]
        assert types == ["LAGRANGIAN", "LBRACE", "IDENT", "MINUS", "IDENT", "RBRACE"]

    def test_unknown_character_raises(self):
        with pytest.raises(ValueError, match="Unrecognized character"):
            tokenize(r"\lagrangian{x & y}")

    def test_error_includes_position(self):
        with pytest.raises(ValueError) as exc:
            tokenize("foo\n  @bar")
        msg = str(exc.value)
        assert "@" in msg
        # The '@' is on line 2.
        assert "line 2" in msg

    def test_multiple_unknown_characters_summarised(self):
        with pytest.raises(ValueError) as exc:
            tokenize(r"\lagrangian{x & y @ z}")
        msg = str(exc.value)
        assert "&" in msg and "@" in msg

    def test_compile_dsl_reports_tokenization_failure(self):
        compiler = PhysicsCompiler()
        result = compiler.compile_dsl(r"\lagrangian{x & y}")
        assert result["success"] is False
        assert "Tokenization failed" in result["error"]
