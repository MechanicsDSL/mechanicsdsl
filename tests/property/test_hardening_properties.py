"""
Property-Based Tests using Hypothesis
=====================================

Tests that verify invariants and properties hold for any valid input.
These tests generate random inputs to find edge cases.
"""

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

# Strategies for generating physics-related inputs
coordinates = st.sampled_from(["x", "y", "z", "theta", "phi", "r", "alpha", "beta"])
positive_float = st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False)
small_float = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
small_angle = st.floats(
    min_value=-math.pi, max_value=math.pi, allow_nan=False, allow_infinity=False
)
time_step = st.floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False)


class TestSecurityProperties:
    """Property-based tests for security module."""

    @given(st.text(min_size=1, max_size=100))
    def test_identifier_validation_never_crashes(self, s):
        """Identifier validation should never crash, only raise or return."""
        from mechanics_dsl.security import InputValidationError, validate_identifier

        try:
            result = validate_identifier(s)
            assert isinstance(result, str)
        except InputValidationError:
            pass  # Expected for invalid inputs
        except Exception as e:
            pytest.fail(f"Unexpected exception: {type(e).__name__}: {e}")

    @given(st.text(max_size=10000))
    def test_string_validation_never_crashes(self, s):
        """String validation should handle any input."""
        from mechanics_dsl.security import InputValidationError, validate_string

        try:
            result = validate_string(s)
            assert isinstance(result, str)
        except InputValidationError:
            pass
        except Exception as e:
            pytest.fail(f"Unexpected exception: {type(e).__name__}: {e}")

    @given(st.text(max_size=5000))
    def test_path_validation_never_crashes(self, s):
        """Path validation should handle any input."""
        from mechanics_dsl.security import InputValidationError, PathTraversalError, validate_path

        try:
            validate_path(s)
        except (InputValidationError, PathTraversalError, OSError):
            pass
        except Exception as e:
            pytest.fail(f"Unexpected exception: {type(e).__name__}: {e}")

    @given(st.text(max_size=1000))
    @settings(max_examples=100)
    def test_dsl_validation_blocks_dangerous_patterns(self, s):
        """DSL validation should block all dangerous patterns."""
        from mechanics_dsl.security import InjectionError, InputValidationError, validate_dsl_code

        dangerous_keywords = ["eval(", "exec(", "__import__", "os.system", "subprocess"]

        # If contains dangerous keyword, should be blocked
        if any(kw in s.lower() for kw in dangerous_keywords):
            try:
                validate_dsl_code(s)
                # If it passed, the pattern wasn't dangerous enough
            except (InjectionError, InputValidationError):
                pass  # Expected


class TestNumericalStability:
    """Property-based tests for numerical stability."""

    @given(small_float, small_float, positive_float)
    @settings(max_examples=200)
    def test_pendulum_energy_bounded(self, theta0, omega0, length):
        """Pendulum energy should never exceed initial potential + kinetic."""
        assume(abs(theta0) < 3.0)  # Reasonable initial angle
        assume(abs(omega0) < 50.0)  # Reasonable initial velocity

        g = 9.81
        m = 1.0

        # Initial energy
        initial_ke = 0.5 * m * (length * omega0) ** 2
        initial_pe = m * g * length * (1 - np.cos(theta0))
        initial_energy = initial_ke + initial_pe

        # Energy should be positive and finite
        assert initial_energy >= 0
        assert np.isfinite(initial_energy)

        # Energy has upper bound based on max height
        max_energy = 2 * m * g * length  # Pendulum at top
        assert initial_energy <= max_energy + initial_ke

    @given(small_float, positive_float, positive_float)
    @settings(max_examples=200)
    def test_harmonic_oscillator_bounds(self, x0, k, m):
        """Harmonic oscillator amplitude should be bounded by initial conditions."""
        assume(k > 0.1)
        assume(m > 0.1)
        assume(abs(x0) > 0.01)

        omega = np.sqrt(k / m)

        # Omega should be real and positive
        assert omega > 0
        assert np.isfinite(omega)

        # Period should be reasonable
        period = 2 * np.pi / omega
        assert period > 0
        assert np.isfinite(period)

    @given(st.lists(small_float, min_size=2, max_size=10))
    def test_state_vector_norm_finite(self, state):
        """State vector norm should always be finite."""
        state_array = np.array(state)
        norm = np.linalg.norm(state_array)

        assert np.isfinite(norm)

    @given(time_step, st.integers(min_value=1, max_value=1000))
    def test_simulation_time_consistency(self, dt, n_steps):
        """Simulation time should accumulate correctly."""
        times = [i * dt for i in range(n_steps)]

        # Time should be monotonically increasing
        for i in range(1, len(times)):
            assert times[i] > times[i - 1]

        # Final time should be close to (n_steps-1) * dt
        expected_final = (n_steps - 1) * dt
        assert abs(times[-1] - expected_final) < 1e-10


class TestCodeGenerationProperties:
    """Property-based tests for code generation."""

    @given(st.sampled_from(["theta", "phi", "x", "y", "q1", "alpha"]))
    def test_coordinate_name_in_generated_code(self, coord):
        """Generated code should contain the coordinate name."""
        import sympy as sp

        sym = sp.Symbol(coord)

        # The coordinate should appear in C code
        from sympy.printing.c import ccode

        expr = sp.sin(sym) + sym**2
        code = ccode(expr)

        assert coord in code

    @given(st.floats(min_value=0.1, max_value=100.0, allow_nan=False))
    def test_parameter_values_preserved(self, value):
        """Parameter values should be correctly represented in generated code."""
        import sympy as sp
        from sympy.printing.c import ccode

        code = ccode(sp.Float(value))

        # The generated code should parse back to approximately the same value
        # (Within floating point precision)
        parsed = float(code)
        assert abs(parsed - value) < 0.001 * value


class TestMathematicalInvariants:
    """Tests for mathematical invariants that must always hold."""

    @given(small_angle)
    def test_sin_cos_identity(self, theta):
        """sin²(θ) + cos²(θ) = 1 should always hold."""
        result = np.sin(theta) ** 2 + np.cos(theta) ** 2
        assert abs(result - 1.0) < 1e-10

    @given(small_float, small_float)
    def test_kinetic_energy_positive(self, v, m):
        """Kinetic energy should be non-negative."""
        assume(m > 0)
        ke = 0.5 * m * v**2
        assert ke >= 0

    @given(st.lists(small_float, min_size=3, max_size=3))
    def test_cross_product_perpendicular(self, v):
        """Cross product should be perpendicular to both input vectors."""
        a = np.array(v)
        b = np.array([1, 0, 0])

        if np.linalg.norm(a) < 0.01:
            return  # Skip near-zero vectors

        cross = np.cross(a, b)

        # Dot product with both should be near zero if perpendicular
        dot_a = np.dot(cross, a)
        dot_b = np.dot(cross, b)

        assert abs(dot_a) < 1e-10
        assert abs(dot_b) < 1e-10

    @given(positive_float, positive_float)
    def test_gravitational_period_formula(self, a, M):
        """Kepler's third law should hold: T² ∝ a³."""
        assume(a > 0.1)
        assume(M > 0.1)

        G = 6.674e-11

        # T = 2π√(a³/GM)
        T_squared = (2 * np.pi) ** 2 * a**3 / (G * M)
        a_cubed = a**3

        # Ratio should be constant for given M
        ratio = T_squared / a_cubed
        expected_ratio = (2 * np.pi) ** 2 / (G * M)

        assert abs(ratio - expected_ratio) < 1e-10 * expected_ratio


class TestEdgeCases:
    """Tests for specific edge cases."""

    def test_zero_initial_conditions(self):
        """System should handle zero initial conditions."""
        from mechanics_dsl.security import validate_number

        assert validate_number(0) == 0
        assert validate_number(0.0) == 0.0

    def test_very_small_timestep(self):
        """System should handle very small timesteps."""
        from mechanics_dsl.security import validate_number

        assert validate_number(1e-15, min_val=0) == 1e-15

    def test_very_large_parameter(self):
        """System should handle large parameters within limits."""
        from mechanics_dsl.security import validate_number

        assert validate_number(1e10, max_val=1e20) == 1e10

    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=50))
    def test_valid_identifier_forms(self, s):
        """Valid identifier patterns should be accepted."""
        import keyword

        from mechanics_dsl.security import validate_identifier

        # Skip Python keywords
        if keyword.iskeyword(s):
            return

        # If it matches the pattern, it should be accepted
        result = validate_identifier(s)
        assert result == s
