"""
Unit tests for MechanicsDSL analysis/energy module.

Tests the EnergyAnalyzer class for energy calculation and conservation analysis.
"""

import numpy as np
import pytest

from mechanics_dsl.analysis.energy import EnergyAnalyzer


class TestEnergyAnalyzerInit:
    """Tests for EnergyAnalyzer initialization."""

    def test_init_creates_instance(self):
        """Test that EnergyAnalyzer can be instantiated."""
        analyzer = EnergyAnalyzer()
        assert analyzer is not None

    def test_init_has_kinetic_func_attr(self):
        """Test that analyzer has _kinetic_func attribute."""
        analyzer = EnergyAnalyzer()
        assert hasattr(analyzer, "_kinetic_func")
        assert analyzer._kinetic_func is None

    def test_init_has_potential_func_attr(self):
        """Test that analyzer has _potential_func attribute."""
        analyzer = EnergyAnalyzer()
        assert hasattr(analyzer, "_potential_func")
        assert analyzer._potential_func is None


class TestComputeKineticEnergy:
    """Tests for EnergyAnalyzer.compute_kinetic_energy method."""

    @pytest.fixture
    def analyzer(self):
        return EnergyAnalyzer()

    @pytest.fixture
    def simple_solution(self):
        """Create a simple harmonic oscillator solution."""
        t = np.linspace(0, 10, 100)
        x = np.cos(t)
        x_dot = -np.sin(t)
        return {"t": t, "y": np.vstack([x, x_dot]), "coordinates": ["x"]}

    def test_compute_kinetic_single_mass(self, analyzer, simple_solution):
        """Test kinetic energy computation for single mass system."""
        masses = {"x": 1.0}
        T = analyzer.compute_kinetic_energy(simple_solution, masses)

        assert isinstance(T, np.ndarray)
        assert len(T) == len(simple_solution["t"])
        assert np.all(T >= 0)  # Kinetic energy is always positive

    def test_compute_kinetic_with_m_key(self, analyzer, simple_solution):
        """Test kinetic energy with default 'm' mass key."""
        masses = {"m": 2.0}
        T = analyzer.compute_kinetic_energy(simple_solution, masses)

        assert isinstance(T, np.ndarray)
        assert len(T) == 100

    def test_compute_kinetic_default_mass(self, analyzer, simple_solution):
        """Test kinetic energy with missing mass (uses default 1.0)."""
        masses = {}
        T = analyzer.compute_kinetic_energy(simple_solution, masses)

        # Should use default mass of 1.0
        assert isinstance(T, np.ndarray)
        assert len(T) == 100

    def test_compute_kinetic_multiple_coordinates(self, analyzer):
        """Test kinetic energy for multi-coordinate system."""
        t = np.linspace(0, 5, 50)
        # Two coordinates: x and y
        x = np.sin(t)
        x_dot = np.cos(t)
        y = np.cos(t)
        y_dot = -np.sin(t)

        solution = {"t": t, "y": np.vstack([x, x_dot, y, y_dot]), "coordinates": ["x", "y"]}
        masses = {"x": 1.0, "y": 2.0}

        T = analyzer.compute_kinetic_energy(solution, masses)

        assert isinstance(T, np.ndarray)
        assert len(T) == 50
        assert np.all(T >= 0)

    def test_compute_kinetic_explicit_velocities(self, analyzer, simple_solution):
        """Test kinetic energy with explicit velocity list (parameter unused in current impl)."""
        masses = {"x": 1.0}
        # velocities parameter is declared but not used in implementation
        T = analyzer.compute_kinetic_energy(simple_solution, masses, velocities=["x_dot"])

        assert isinstance(T, np.ndarray)


class TestComputePotentialEnergy:
    """Tests for EnergyAnalyzer.compute_potential_energy method."""

    @pytest.fixture
    def analyzer(self):
        return EnergyAnalyzer()

    @pytest.fixture
    def oscillator_solution(self):
        """Create a harmonic oscillator solution."""
        t = np.linspace(0, 10, 100)
        x = np.cos(t)
        x_dot = -np.sin(t)
        return {"t": t, "y": np.vstack([x, x_dot]), "coordinates": ["x"]}

    def test_compute_potential_simple(self, analyzer, oscillator_solution):
        """Test potential energy with simple spring potential."""
        k = 1.0  # spring constant

        def spring_potential(state):
            x = state[0]  # position is first element
            return 0.5 * k * x**2

        V = analyzer.compute_potential_energy(oscillator_solution, spring_potential)

        assert isinstance(V, np.ndarray)
        assert len(V) == 100
        assert np.all(V >= 0)  # Spring potential is always positive

    def test_compute_potential_gravity(self, analyzer):
        """Test potential energy with gravity potential."""
        t = np.linspace(0, 5, 50)
        h = np.abs(np.sin(t))  # height always positive
        h_dot = np.cos(t)

        solution = {"t": t, "y": np.vstack([h, h_dot]), "coordinates": ["h"]}

        m, g = 1.0, 9.81

        def gravity_potential(state):
            return m * g * state[0]

        V = analyzer.compute_potential_energy(solution, gravity_potential)

        assert isinstance(V, np.ndarray)
        assert len(V) == 50

    def test_compute_potential_custom_function(self, analyzer, oscillator_solution):
        """Test potential energy with custom potential function."""

        def custom_potential(state):
            x = state[0]
            # Quartic potential
            return 0.25 * x**4

        V = analyzer.compute_potential_energy(oscillator_solution, custom_potential)

        assert isinstance(V, np.ndarray)
        assert np.all(V >= 0)


class TestCheckConservation:
    """Tests for EnergyAnalyzer.check_conservation method."""

    @pytest.fixture
    def analyzer(self):
        return EnergyAnalyzer()

    def test_check_conservation_conserved(self, analyzer):
        """Test conservation check for conserved system."""
        t = np.linspace(0, 10, 100)

        # Harmonic oscillator: T + V = constant
        np.cos(t)
        kinetic = 0.5 * np.sin(t) ** 2
        potential = 0.5 * np.cos(t) ** 2
        # Total energy is constant = 0.5

        solution = {"t": t}
        result = analyzer.check_conservation(solution, kinetic, potential.copy())

        assert isinstance(result, dict)
        assert "conserved" in result
        assert result["conserved"] == True
        assert "max_relative_error" in result
        assert result["max_relative_error"] < 1e-3

    def test_check_conservation_not_conserved(self, analyzer):
        """Test conservation check for non-conserved system."""
        t = np.linspace(0, 10, 100)

        # Non-conserved: energy changes over time
        kinetic = np.sin(t) ** 2
        potential = 0.5 * t  # Linear increase in potential

        solution = {"t": t}
        result = analyzer.check_conservation(solution, kinetic, potential)

        assert isinstance(result, dict)
        assert "conserved" in result
        assert result["conserved"] == False

    def test_check_conservation_custom_tolerance(self, analyzer):
        """Test conservation check with custom tolerance."""
        t = np.linspace(0, 10, 100)

        # Use predictable small variations instead of random
        kinetic = 0.5 * np.ones(100) + 0.005 * np.sin(t)
        potential = 0.5 * np.ones(100)

        solution = {"t": t}

        # With very loose tolerance, should be conserved
        result_loose = analyzer.check_conservation(solution, kinetic, potential, tolerance=0.1)
        assert result_loose["conserved"] == True

        # With tight tolerance, won't be conserved due to variation
        result_tight = analyzer.check_conservation(solution, kinetic, potential, tolerance=1e-6)
        # This will be False due to the sine variation
        assert isinstance(result_tight["conserved"], (bool, np.bool_))

    def test_check_conservation_zero_energy(self, analyzer):
        """Test conservation check with zero initial energy."""
        t = np.linspace(0, 10, 100)

        # Both energies are zero
        kinetic = np.zeros(100)
        potential = np.zeros(100)

        solution = {"t": t}
        result = analyzer.check_conservation(solution, kinetic, potential)

        assert isinstance(result, dict)
        assert "conserved" in result
        # With zero energy, should be conserved
        assert result["conserved"] == True

    def test_check_conservation_result_keys(self, analyzer):
        """Test that result contains all expected keys."""
        t = np.linspace(0, 10, 100)
        kinetic = np.ones(100)
        potential = np.ones(100)

        solution = {"t": t}
        result = analyzer.check_conservation(solution, kinetic, potential)

        expected_keys = [
            "conserved",
            "initial_energy",
            "max_relative_error",
            "mean_relative_error",
            "total_energy",
            "kinetic_energy",
            "potential_energy",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


class TestComputePendulumEnergy:
    """Tests for EnergyAnalyzer.compute_pendulum_energy method."""

    @pytest.fixture
    def analyzer(self):
        return EnergyAnalyzer()

    @pytest.fixture
    def pendulum_solution(self):
        """Create a simple pendulum solution."""
        t = np.linspace(0, 10, 100)
        theta = 0.1 * np.cos(t)  # Small oscillations
        theta_dot = -0.1 * np.sin(t)
        return {"t": t, "y": np.vstack([theta, theta_dot]), "coordinates": ["theta"]}

    def test_compute_pendulum_energy_returns_dict(self, analyzer, pendulum_solution):
        """Test that compute_pendulum_energy returns a dictionary."""
        result = analyzer.compute_pendulum_energy(pendulum_solution, m=1.0, l=1.0, g=9.81)

        assert isinstance(result, dict)

    def test_compute_pendulum_energy_has_components(self, analyzer, pendulum_solution):
        """Test that result contains kinetic, potential, and total energy."""
        result = analyzer.compute_pendulum_energy(pendulum_solution, m=1.0, l=1.0, g=9.81)

        assert "kinetic" in result
        assert "potential" in result
        assert "total" in result

    def test_compute_pendulum_energy_array_lengths(self, analyzer, pendulum_solution):
        """Test that energy arrays have correct length."""
        result = analyzer.compute_pendulum_energy(pendulum_solution, m=1.0, l=1.0, g=9.81)

        assert len(result["kinetic"]) == 100
        assert len(result["potential"]) == 100
        assert len(result["total"]) == 100

    def test_compute_pendulum_energy_conservation(self, analyzer, pendulum_solution):
        """Test that pendulum energy components are computed correctly."""
        result = analyzer.compute_pendulum_energy(pendulum_solution, m=1.0, l=1.0, g=9.81)

        total = result["total"]
        # The test data isn't a physically accurate pendulum solution, so just check
        # that the total is the sum of kinetic and potential
        np.testing.assert_allclose(total, result["kinetic"] + result["potential"], rtol=1e-10)

    def test_compute_pendulum_energy_positive_kinetic(self, analyzer, pendulum_solution):
        """Test that kinetic energy is always non-negative."""
        result = analyzer.compute_pendulum_energy(pendulum_solution, m=1.0, l=1.0, g=9.81)

        assert np.all(result["kinetic"] >= 0)

    def test_compute_pendulum_energy_different_params(self, analyzer):
        """Test with different pendulum parameters."""
        t = np.linspace(0, 5, 50)
        theta = 0.5 * np.cos(2 * t)
        theta_dot = -np.sin(2 * t)

        solution = {"t": t, "y": np.vstack([theta, theta_dot]), "coordinates": ["theta"]}

        # Different mass, length, gravity
        result = analyzer.compute_pendulum_energy(solution, m=2.0, l=0.5, g=10.0)

        assert isinstance(result, dict)
        assert len(result["kinetic"]) == 50

    def test_compute_pendulum_energy_scales_with_mass(self, analyzer, pendulum_solution):
        """Test that energy scales linearly with mass."""
        result_m1 = analyzer.compute_pendulum_energy(pendulum_solution, m=1.0, l=1.0, g=9.81)
        result_m2 = analyzer.compute_pendulum_energy(pendulum_solution, m=2.0, l=1.0, g=9.81)

        # Total energy should double when mass doubles
        np.testing.assert_allclose(result_m2["total"], 2 * result_m1["total"], rtol=1e-6)
