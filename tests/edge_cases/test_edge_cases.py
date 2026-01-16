"""
Edge Case Tests
===============

Tests for boundary conditions, corner cases, and unusual inputs.
"""

import math
import os
import sys
import tempfile

import numpy as np
import pytest


class TestNumericalEdgeCases:
    """Tests for numerical edge cases."""

    def test_very_small_angle(self):
        """Test with very small angles (small angle approximation)."""
        theta = 1e-10

        # sin(θ) ≈ θ for small θ
        assert abs(np.sin(theta) - theta) < 1e-20

        # cos(θ) ≈ 1 for small θ
        assert abs(np.cos(theta) - 1) < 1e-20

    def test_angle_at_pi(self):
        """Test at θ = π (pendulum at top)."""
        theta = np.pi

        assert abs(np.sin(theta)) < 1e-15
        assert abs(np.cos(theta) + 1) < 1e-15

    def test_very_large_velocity(self):
        """Test with very large velocities."""
        v = 1e8  # 100 million m/s (close to speed of light)
        m = 1.0

        ke = 0.5 * m * v**2

        assert np.isfinite(ke)
        assert ke > 0

    def test_very_small_mass(self):
        """Test with very small mass."""
        m = 1e-30  # Electron mass scale
        v = 1e6

        ke = 0.5 * m * v**2

        assert np.isfinite(ke)
        assert ke > 0

    def test_zero_timestep(self):
        """Test with zero timestep."""
        dt = 0.0

        # No change with zero timestep
        x = 1.0
        v = 5.0

        x_new = x + v * dt
        assert x_new == x

    def test_negative_timestep(self):
        """Test with negative timestep (time reversal)."""
        dt = -0.001

        x = 1.0
        v = 5.0

        x_new = x + v * dt
        assert x_new < x  # Moving backward

    def test_infinity_handling(self):
        """Test that infinity is properly detected."""
        from mechanics_dsl.security import InputValidationError, validate_number

        with pytest.raises(InputValidationError, match="Inf"):
            validate_number(float("inf"))

        with pytest.raises(InputValidationError, match="Inf"):
            validate_number(float("-inf"))

    def test_nan_handling(self):
        """Test that NaN is properly detected."""
        from mechanics_dsl.security import InputValidationError, validate_number

        with pytest.raises(InputValidationError, match="NaN"):
            validate_number(float("nan"))


class TestStringEdgeCases:
    """Tests for string edge cases."""

    def test_empty_string(self):
        """Test empty string handling."""
        from mechanics_dsl.security import validate_string

        assert validate_string("") == ""

    def test_whitespace_only(self):
        """Test whitespace-only string."""
        from mechanics_dsl.security import validate_string

        assert validate_string("   ") == "   "
        assert validate_string("\t\n") == "\t\n"

    def test_unicode_string(self):
        """Test Unicode string handling."""
        from mechanics_dsl.security import validate_string

        unicode_str = "θ φ ψ → ∞ ∫"
        assert validate_string(unicode_str) == unicode_str

    def test_very_long_unicode(self):
        """Test very long Unicode string."""
        from mechanics_dsl.security import validate_string

        long_str = "θ" * 10000
        assert validate_string(long_str) == long_str

    def test_control_characters(self):
        """Test control characters."""
        from mechanics_dsl.security import InputValidationError, validate_string

        # Null byte should be rejected
        with pytest.raises(InputValidationError):
            validate_string("hello\x00world")

        # Other control chars should be allowed
        assert validate_string("hello\tworld") == "hello\tworld"

    def test_escape_sequences(self):
        """Test escape sequences in strings."""
        from mechanics_dsl.security import validate_string

        assert validate_string("\\n") == "\\n"
        assert validate_string("line1\nline2") == "line1\nline2"


class TestPathEdgeCases:
    """Tests for path edge cases."""

    def test_current_directory(self):
        """Test current directory reference."""
        from mechanics_dsl.security import validate_path

        # Single dot should be allowed
        result = validate_path(".")
        assert result.exists()

    def test_absolute_path(self):
        """Test absolute path handling."""
        from mechanics_dsl.security import validate_path

        with tempfile.TemporaryDirectory() as tmpdir:
            result = validate_path(tmpdir)
            assert result.is_absolute()

    def test_path_with_spaces(self):
        """Test path with spaces."""
        from mechanics_dsl.security import validate_path

        with tempfile.TemporaryDirectory() as tmpdir:
            space_dir = os.path.join(tmpdir, "path with spaces")
            os.makedirs(space_dir)

            result = validate_path(space_dir)
            assert result.exists()

    def test_path_with_unicode(self):
        """Test path with Unicode characters."""
        from mechanics_dsl.security import validate_path

        with tempfile.TemporaryDirectory() as tmpdir:
            unicode_dir = os.path.join(tmpdir, "θπ_folder")
            os.makedirs(unicode_dir)

            result = validate_path(unicode_dir)
            assert result.exists()

    def test_symlink_handling(self):
        """Test symbolic link handling."""
        from mechanics_dsl.security import validate_path

        with tempfile.TemporaryDirectory() as tmpdir:
            real_file = os.path.join(tmpdir, "real.txt")
            with open(real_file, "w") as f:
                f.write("test")

            link_path = os.path.join(tmpdir, "link.txt")
            try:
                os.symlink(real_file, link_path)
                result = validate_path(link_path)
                # Should resolve to real path
                assert result.exists()
            except OSError:
                pytest.skip("Symlinks not supported on this system")


class TestDSLEdgeCases:
    """Tests for DSL code edge cases."""

    def test_minimal_dsl(self):
        """Test minimal valid DSL."""
        from mechanics_dsl.security import validate_dsl_code

        minimal = "x"
        assert validate_dsl_code(minimal) == minimal

    def test_dsl_with_unicode_math(self):
        """Test DSL with Unicode math symbols."""
        from mechanics_dsl.security import validate_dsl_code

        code = r"\system{θ_oscillator}"
        assert validate_dsl_code(code) == code

    def test_dsl_with_comments(self):
        """Test DSL with various comment styles."""
        from mechanics_dsl.security import validate_dsl_code

        code = """
        % This is a comment
        \\system{test}
        # Another comment style
        """
        assert validate_dsl_code(code) == code

    def test_nested_braces(self):
        """Test DSL with nested braces."""
        from mechanics_dsl.security import validate_dsl_code

        code = r"\system{test{nested}{braces}}"
        assert validate_dsl_code(code) == code

    def test_empty_commands(self):
        """Test DSL with empty command arguments."""
        from mechanics_dsl.security import validate_dsl_code

        code = r"\system{}"
        assert validate_dsl_code(code) == code


class TestBoundaryConditions:
    """Tests for boundary condition handling."""

    def test_exactly_at_limit(self):
        """Test values exactly at limits."""
        from mechanics_dsl.security import validate_number

        assert validate_number(10, max_val=10) == 10
        assert validate_number(-5, min_val=-5) == -5

    def test_one_below_limit(self):
        """Test values one unit below limit."""
        from mechanics_dsl.security import InputValidationError, validate_number

        with pytest.raises(InputValidationError):
            validate_number(0, min_val=1)

    def test_epsilon_below_limit(self):
        """Test values epsilon below limit."""
        from mechanics_dsl.security import validate_number

        eps = sys.float_info.epsilon
        assert validate_number(10 - eps, max_val=10) == 10 - eps

    def test_max_float(self):
        """Test with maximum float value."""
        from mechanics_dsl.security import validate_number

        max_float = sys.float_info.max
        assert validate_number(max_float) == max_float

    def test_min_positive_float(self):
        """Test with minimum positive float."""
        from mechanics_dsl.security import validate_number

        min_float = sys.float_info.min
        assert validate_number(min_float) == min_float


class TestConcurrencyEdgeCases:
    """Tests for concurrency edge cases."""

    def test_rate_limiter_rapid_requests(self):
        """Test rate limiter with rapid successive requests."""
        from mechanics_dsl.security import RateLimitConfig, RateLimiter

        config = RateLimitConfig(max_requests=1000, window_seconds=1)
        limiter = RateLimiter(config)

        # Rapid requests
        for i in range(1000):
            assert limiter.check("test") is True

        # Next should fail
        assert limiter.check("test") is False

    def test_sandbox_nested_execution(self):
        """Test nested sandbox execution."""
        from mechanics_dsl.security import Sandbox, SandboxConfig

        config = SandboxConfig(max_time_seconds=5)

        def inner():
            return 42

        def outer():
            return inner() * 2

        with Sandbox(config) as sb:
            result = sb.execute(outer)
            assert result == 84
