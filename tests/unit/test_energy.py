"""
Unit tests for MechanicsDSL energy module.

Tests the PotentialEnergyCalculator class and related energy calculations.
"""

import numpy as np
import pytest

from mechanics_dsl.energy import PotentialEnergyCalculator


class TestComputePEOffset:
    """Tests for PotentialEnergyCalculator.compute_pe_offset method."""

    def test_simple_pendulum_offset(self):
        """Test PE offset for simple pendulum."""
        params = {"m": 1.0, "l": 1.0, "g": 9.81}
        offset = PotentialEnergyCalculator.compute_pe_offset("simple_pendulum", params)
        expected = -1.0 * 9.81 * 1.0
        assert offset == pytest.approx(expected)

    def test_double_pendulum_offset(self):
        """Test PE offset for double pendulum."""
        params = {"m1": 1.0, "m2": 1.0, "l1": 1.0, "l2": 1.0, "g": 9.81}
        offset = PotentialEnergyCalculator.compute_pe_offset("double_pendulum", params)
        expected = -1.0 * 9.81 * 1.0 - 1.0 * 9.81 * (1.0 + 1.0)
        assert offset == pytest.approx(expected)

    def test_simple_pendulum_case_insensitive(self):
        """Test that system type matching is case insensitive."""
        params = {"m": 2.0, "l": 0.5, "g": 10.0}
        offset_lower = PotentialEnergyCalculator.compute_pe_offset("pendulum", params)
        offset_upper = PotentialEnergyCalculator.compute_pe_offset("PENDULUM", params)
        assert offset_lower == pytest.approx(offset_upper)

    def test_oscillator_zero_offset(self):
        """Test that oscillator has zero PE offset."""
        params = {"k": 10.0, "m": 1.0}
        offset = PotentialEnergyCalculator.compute_pe_offset("oscillator", params)
        assert offset == 0.0

    def test_spring_zero_offset(self):
        """Test that spring system has zero PE offset."""
        params = {"k": 5.0}
        offset = PotentialEnergyCalculator.compute_pe_offset("spring_mass", params)
        assert offset == 0.0

    def test_unknown_system_zero_offset(self):
        """Test that unknown system types return zero offset."""
        params = {"mass": 1.0}
        offset = PotentialEnergyCalculator.compute_pe_offset("unknown_system", params)
        assert offset == 0.0

    def test_empty_system_type(self):
        """Test with empty system type string."""
        params = {"m": 1.0}
        offset = PotentialEnergyCalculator.compute_pe_offset("", params)
        assert offset == 0.0

    def test_default_parameters_pendulum(self):
        """Test that default parameters are used when not provided."""
        params = {}
        offset = PotentialEnergyCalculator.compute_pe_offset("pendulum", params)
        # Default: m=1.0, l=1.0, g=9.81
        expected = -1.0 * 9.81 * 1.0
        assert offset == pytest.approx(expected)

    def test_custom_mass_pendulum(self):
        """Test pendulum with custom mass."""
        params = {"m": 5.0, "l": 1.0, "g": 9.81}
        offset = PotentialEnergyCalculator.compute_pe_offset("pendulum", params)
        expected = -5.0 * 9.81 * 1.0
        assert offset == pytest.approx(expected)

    def test_custom_length_pendulum(self):
        """Test pendulum with custom length."""
        params = {"m": 1.0, "l": 2.5, "g": 9.81}
        offset = PotentialEnergyCalculator.compute_pe_offset("pendulum", params)
        expected = -1.0 * 9.81 * 2.5
        assert offset == pytest.approx(expected)


class TestComputeKineticEnergy:
    """Tests for PotentialEnergyCalculator.compute_kinetic_energy method."""

    def test_simple_pendulum_kinetic_energy(self):
        """Test KE calculation for simple pendulum."""
        solution = {
            "success": True,
            "t": np.array([0.0, 0.1, 0.2]),
            "y": np.array([[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]]),
            "coordinates": ["theta"],
        }
        params = {"m": 1.0, "l": 1.0}
        ke = PotentialEnergyCalculator.compute_kinetic_energy(solution, params)
        # KE = 0.5 * m * l^2 * theta_dot^2
        expected = 0.5 * 1.0 * 1.0**2 * np.array([0.5, 0.6, 0.7]) ** 2
        np.testing.assert_array_almost_equal(ke, expected)

    def test_double_pendulum_kinetic_energy(self):
        """Test KE calculation for double pendulum."""
        solution = {
            "success": True,
            "t": np.array([0.0, 0.1]),
            "y": np.array(
                [
                    [0.1, 0.15],  # theta1
                    [0.5, 0.6],  # theta1_dot
                    [0.2, 0.25],  # theta2
                    [0.3, 0.4],  # theta2_dot
                ]
            ),
            "coordinates": ["theta1", "theta2"],
        }
        params = {"m1": 1.0, "m2": 1.0, "l1": 1.0, "l2": 1.0}
        ke = PotentialEnergyCalculator.compute_kinetic_energy(solution, params)
        assert len(ke) == 2
        assert np.all(ke >= 0)

    def test_cartesian_kinetic_energy(self):
        """Test KE calculation for Cartesian coordinate system."""
        solution = {
            "success": True,
            "t": np.array([0.0, 0.5, 1.0]),
            "y": np.array([[0.0, 0.5, 1.0], [1.0, 2.0, 3.0]]),
            "coordinates": ["x"],
        }
        params = {"m": 2.0}
        ke = PotentialEnergyCalculator.compute_kinetic_energy(solution, params)
        expected = 0.5 * 2.0 * np.array([1.0, 2.0, 3.0]) ** 2
        np.testing.assert_array_almost_equal(ke, expected)

    def test_empty_coordinates(self):
        """Test KE with empty y raises ValueError."""
        solution = {
            "success": True,
            "t": np.array([0.0, 1.0]),
            "y": np.array([]),
            "coordinates": [],
        }
        params = {"m": 1.0}
        with pytest.raises(ValueError):
            PotentialEnergyCalculator.compute_kinetic_energy(solution, params)

    def test_invalid_parameters_type(self):
        """Test that non-dict parameters raise TypeError."""
        solution = {
            "success": True,
            "t": np.array([0.0, 1.0]),
            "y": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "coordinates": ["theta"],
        }
        with pytest.raises(TypeError):
            PotentialEnergyCalculator.compute_kinetic_energy(solution, "invalid")

    def test_default_mass(self):
        """Test that default mass is used when not provided."""
        solution = {
            "success": True,
            "t": np.array([0.0]),
            "y": np.array([[0.1], [0.5]]),
            "coordinates": ["theta"],
        }
        params = {"l": 1.0}  # No 'm' provided
        ke = PotentialEnergyCalculator.compute_kinetic_energy(solution, params)
        # Default m=1.0
        expected = 0.5 * 1.0 * 1.0**2 * 0.5**2
        assert ke[0] == pytest.approx(expected)


class TestComputePotentialEnergy:
    """Tests for PotentialEnergyCalculator.compute_potential_energy method."""

    def test_simple_pendulum_potential_energy(self):
        """Test PE calculation for simple pendulum."""
        solution = {
            "success": True,
            "t": np.array([0.0, 0.5, 1.0]),
            "y": np.array([[0.0, np.pi / 4, np.pi / 2], [0.1, 0.2, 0.3]]),  # theta  # theta_dot
            "coordinates": ["theta"],
        }
        params = {"m": 1.0, "l": 1.0, "g": 9.81}
        pe = PotentialEnergyCalculator.compute_potential_energy(solution, params, "simple_pendulum")
        # At theta=0, PE should be 0 (after offset correction)
        assert pe[0] == pytest.approx(0.0)

    def test_cartesian_potential_energy(self):
        """Test PE calculation for Cartesian (spring) system."""
        solution = {
            "success": True,
            "t": np.array([0.0, 0.5, 1.0]),
            "y": np.array([[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]]),  # x  # v
            "coordinates": ["x"],
        }
        params = {"k": 10.0}
        pe = PotentialEnergyCalculator.compute_potential_energy(solution, params)
        # PE = 0.5 * k * x^2
        expected = 0.5 * 10.0 * np.array([0.0, 0.5, 1.0]) ** 2
        np.testing.assert_array_almost_equal(pe, expected)

    def test_empty_coordinates_pe(self):
        """Test PE with empty coordinates returns zeros (logs warning)."""
        solution = {
            "success": True,
            "t": np.array([0.0, 1.0]),
            "y": np.array([]),
            "coordinates": [],
        }
        params = {"k": 1.0}
        pe = PotentialEnergyCalculator.compute_potential_energy(solution, params)
        np.testing.assert_array_almost_equal(pe, np.zeros(2))

    def test_double_pendulum_potential_energy(self):
        """Test PE calculation for double pendulum."""
        solution = {
            "success": True,
            "t": np.array([0.0]),
            "y": np.array(
                [
                    [0.0],  # theta1 = 0 (straight down)
                    [0.0],  # theta1_dot
                    [0.0],  # theta2 = 0 (straight down)
                    [0.0],  # theta2_dot
                ]
            ),
            "coordinates": ["theta1", "theta2"],
        }
        params = {"m1": 1.0, "m2": 1.0, "l1": 1.0, "l2": 1.0, "g": 9.81}
        pe = PotentialEnergyCalculator.compute_potential_energy(solution, params, "double_pendulum")
        # At both angles = 0, PE should be 0 (minimum position, with offset)
        assert pe[0] == pytest.approx(0.0)

    def test_potential_energy_always_non_negative(self):
        """Test that PE with offset is always non-negative for pendulum at minimum."""
        solution = {
            "success": True,
            "t": np.array([0.0]),
            "y": np.array([[0.0], [0.0]]),  # theta=0 (at minimum)
            "coordinates": ["theta"],
        }
        params = {"m": 1.0, "l": 1.0, "g": 9.81}
        pe = PotentialEnergyCalculator.compute_potential_energy(solution, params, "simple_pendulum")
        assert pe[0] >= 0.0

    def test_stiff_spring_high_pe(self):
        """Test high PE with stiff spring."""
        solution = {
            "success": True,
            "t": np.array([0.0]),
            "y": np.array([[1.0], [0.0]]),  # x=1
            "coordinates": ["x"],
        }
        params = {"k": 1000.0}  # Very stiff spring
        pe = PotentialEnergyCalculator.compute_potential_energy(solution, params)
        expected = 0.5 * 1000.0 * 1.0**2
        assert pe[0] == pytest.approx(expected)
