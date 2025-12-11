"""
Tests for energy.py edge cases to achieve 90%+ coverage

Covers warning paths for:
- Empty coordinates
- Insufficient state vectors
"""

import pytest
import numpy as np
from unittest.mock import patch

from mechanics_dsl.energy import PotentialEnergyCalculator


class TestComputePeOffset:
    """Tests for compute_pe_offset method."""
    
    def test_double_pendulum(self):
        result = PotentialEnergyCalculator.compute_pe_offset(
            'double_pendulum', {'m1': 1.0, 'm2': 1.0, 'l1': 1.0, 'l2': 1.0, 'g': 9.81}
        )
        assert result < 0  # Negative offset for minimum at hanging position
    
    def test_simple_pendulum(self):
        result = PotentialEnergyCalculator.compute_pe_offset(
            'pendulum', {'m': 1.0, 'l': 1.0, 'g': 9.81}
        )
        assert result == -1.0 * 9.81 * 1.0
    
    def test_oscillator(self):
        result = PotentialEnergyCalculator.compute_pe_offset('oscillator', {})
        assert result == 0.0
    
    def test_spring(self):
        result = PotentialEnergyCalculator.compute_pe_offset('spring_mass', {})
        assert result == 0.0
    
    def test_unknown(self):
        result = PotentialEnergyCalculator.compute_pe_offset('unknown_system', {})
        assert result == 0.0


class TestComputeKineticEnergyEdgeCases:
    """Tests for compute_kinetic_energy edge cases using mocking."""
    
    def test_empty_coordinates(self):
        """Test with empty coordinates list - bypass validation."""
        solution = {
            'success': True,
            't': np.array([0.0, 1.0]),
            'y': np.zeros((2, 2)),
            'coordinates': []
        }
        parameters = {'m': 1.0}
        with patch('mechanics_dsl.energy.validate_solution_dict'):
            result = PotentialEnergyCalculator.compute_kinetic_energy(solution, parameters)
        assert np.all(result == 0)
    
    def test_simple_pendulum_insufficient_state(self):
        """Test simple pendulum with insufficient state vector - bypass validation."""
        solution = {
            'success': True,
            't': np.array([0.0, 1.0]),
            'y': np.array([[0.1, 0.2]]),  # Only 1 row, need 2 for theta_dot
            'coordinates': ['theta']
        }
        parameters = {'m': 1.0, 'l': 1.0}
        with patch('mechanics_dsl.energy.validate_solution_dict'):
            result = PotentialEnergyCalculator.compute_kinetic_energy(solution, parameters)
        assert np.all(result == 0)
    
    def test_double_pendulum_insufficient_state(self):
        """Test double pendulum with insufficient state vector - bypass validation."""
        solution = {
            'success': True,
            't': np.array([0.0, 1.0]),
            'y': np.array([[0.1, 0.2], [0.1, 0.2]]),  # Only 2 rows, need 4
            'coordinates': ['theta1', 'theta2']
        }
        parameters = {'m1': 1.0, 'm2': 1.0, 'l1': 1.0, 'l2': 1.0}
        with patch('mechanics_dsl.energy.validate_solution_dict'):
            result = PotentialEnergyCalculator.compute_kinetic_energy(solution, parameters)
        assert np.all(result == 0)
    
    def test_invalid_parameters_type(self):
        """Test with invalid parameters type."""
        solution = {
            'success': True,
            't': np.array([0.0, 1.0]),
            'y': np.array([[0.1, 0.2], [0.1, 0.2]]),
            'coordinates': ['x']
        }
        with pytest.raises(TypeError):
            PotentialEnergyCalculator.compute_kinetic_energy(solution, "not a dict")


class TestComputePotentialEnergyEdgeCases:
    """Tests for compute_potential_energy edge cases (no validation call)."""
    
    def test_empty_coordinates(self):
        """Test with empty coordinates list."""
        solution = {
            'success': True,
            't': np.array([0.0, 1.0]),
            'y': np.zeros((2, 2)),
            'coordinates': []
        }
        parameters = {'m': 1.0, 'l': 1.0, 'g': 9.81}
        result = PotentialEnergyCalculator.compute_potential_energy(solution, parameters)
        assert np.all(result == 0)
    
    def test_simple_pendulum_insufficient_state(self):
        """Test simple pendulum with insufficient state vector."""
        solution = {
            'success': True,
            't': np.array([0.0, 1.0]),
            'y': np.array([]).reshape(0, 2),  # 0 rows
            'coordinates': ['theta']
        }
        parameters = {'m': 1.0, 'l': 1.0, 'g': 9.81}
        result = PotentialEnergyCalculator.compute_potential_energy(solution, parameters)
        assert np.all(result == 0)
    
    def test_double_pendulum_insufficient_state(self):
        """Test double pendulum with insufficient state vector."""
        solution = {
            'success': True,
            't': np.array([0.0, 1.0]),
            'y': np.array([[0.1, 0.2], [0.1, 0.2]]),  # Only 2 rows, need at least 3
            'coordinates': ['theta1', 'theta2']
        }
        parameters = {'m1': 1.0, 'm2': 1.0, 'l1': 1.0, 'l2': 1.0, 'g': 9.81}
        result = PotentialEnergyCalculator.compute_potential_energy(solution, parameters)
        assert np.all(result == 0)
    
    def test_cartesian_insufficient_state(self):
        """Test Cartesian system with insufficient state vector."""
        solution = {
            'success': True,
            't': np.array([0.0, 1.0]),
            'y': np.array([]).reshape(0, 2),  # 0 rows
            'coordinates': ['x']
        }
        parameters = {'k': 1.0}
        result = PotentialEnergyCalculator.compute_potential_energy(solution, parameters)
        assert np.all(result == 0)
    
    def test_double_pendulum_success(self):
        """Test double pendulum PE calculation."""
        solution = {
            'success': True,
            't': np.array([0.0, 1.0]),
            'y': np.array([[0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2]]),  # 4 rows
            'coordinates': ['theta1', 'theta2']
        }
        parameters = {'m1': 1.0, 'm2': 1.0, 'l1': 1.0, 'l2': 1.0, 'g': 9.81}
        result = PotentialEnergyCalculator.compute_potential_energy(solution, parameters)
        assert len(result) == 2


class TestComputeKineticEnergySuccess:
    """Tests for successful kinetic energy calculations."""
    
    def test_simple_pendulum(self):
        """Test simple pendulum KE calculation."""
        solution = {
            'success': True,
            't': np.array([0.0, 1.0, 2.0]),
            'y': np.array([[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]]),
            'coordinates': ['theta']
        }
        parameters = {'m': 1.0, 'l': 1.0}
        result = PotentialEnergyCalculator.compute_kinetic_energy(solution, parameters)
        assert len(result) == 3
        assert np.all(result >= 0)
    
    def test_double_pendulum(self):
        """Test double pendulum KE calculation."""
        solution = {
            'success': True,
            't': np.array([0.0, 1.0]),
            'y': np.array([[0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2]]),
            'coordinates': ['theta1', 'theta2']
        }
        parameters = {'m1': 1.0, 'm2': 1.0, 'l1': 1.0, 'l2': 1.0}
        result = PotentialEnergyCalculator.compute_kinetic_energy(solution, parameters)
        assert len(result) == 2
    
    def test_cartesian(self):
        """Test Cartesian system KE calculation."""
        solution = {
            'success': True,
            't': np.array([0.0, 1.0]),
            'y': np.array([[0.1, 0.2], [0.5, 0.6]]),
            'coordinates': ['x']
        }
        parameters = {'m': 1.0}
        result = PotentialEnergyCalculator.compute_kinetic_energy(solution, parameters)
        assert len(result) == 2

