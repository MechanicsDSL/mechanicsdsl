"""
Unit tests for MechanicsDSL solver module.

Tests the NumericalSimulator class and related simulation functionality.
"""

import pytest
import numpy as np
import sympy as sp

from mechanics_dsl.solver import NumericalSimulator
from mechanics_dsl.symbolic import SymbolicEngine


class TestNumericalSimulatorWithSymbolicEngine:
    """Tests for NumericalSimulator using SymbolicEngine."""

    def test_init_creates_instance(self):
        """Test that NumericalSimulator can be instantiated."""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        assert sim is not None
        assert sim.symbolic is engine

    def test_simulator_has_required_attributes(self):
        """Test simulator has required attributes."""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        assert hasattr(sim, 'equations')
        assert hasattr(sim, 'parameters')
        assert hasattr(sim, 'initial_conditions')
        assert hasattr(sim, 'constraints')

    def test_set_parameters(self):
        """Test setting parameters."""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        sim.set_parameters({'m': 1.0, 'k': 10.0})
        assert sim.parameters['m'] == 1.0
        assert sim.parameters['k'] == 10.0

    def test_set_initial_conditions(self):
        """Test setting initial conditions."""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        sim.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        assert sim.initial_conditions['x'] == 1.0
        assert sim.initial_conditions['x_dot'] == 0.0

    def test_add_constraint(self):
        """Test adding constraints."""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        x = sp.Symbol('x')
        sim.add_constraint(x - 1)
        assert len(sim.constraints) == 1

    def test_compile_equations_sets_coordinates(self):
        """Test that compiling equations sets coordinates."""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        sim.set_parameters({'k': 1.0})
        x = sp.Symbol('x')
        # Note: compiled equations keys should be 'x_ddot', not 'x'
        accelerations = {'x_ddot': -x}
        sim.compile_equations(accelerations, ['x'])
        assert 'x' in sim.coordinates

    def test_compile_equations_sets_state_vars(self):
        """Test that compiling equations sets state variables."""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        sim.set_parameters({'k': 1.0})
        x = sp.Symbol('x')
        accelerations = {'x_ddot': -x}
        sim.compile_equations(accelerations, ['x'])
        assert 'x' in sim.state_vars
        assert 'x_dot' in sim.state_vars


class TestNumericalSimulatorSimulation:
    """Integration tests for simulation functionality."""

    def test_simulate_harmonic_oscillator(self):
        """Test simulating a harmonic oscillator."""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        
        sim.set_parameters({'k': 1.0})
        sim.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        
        x = sp.Symbol('x')
        accelerations = {'x_ddot': -x}
        sim.compile_equations(accelerations, ['x'])
        
        result = sim.simulate((0, 1), num_points=10)
        assert isinstance(result, dict)
        assert 'success' in result
        assert 't' in result
        assert 'y' in result

    def test_simulation_time_span(self):
        """Test that simulation respects time span."""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        
        sim.set_parameters({'k': 1.0})
        sim.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        
        x = sp.Symbol('x')
        sim.compile_equations({'x_ddot': -x}, ['x'])
        
        result = sim.simulate((0, 5), num_points=50)
        assert result['t'][0] == pytest.approx(0.0)
        assert result['t'][-1] == pytest.approx(5.0)

    def test_simulation_num_points(self):
        """Test that simulation returns correct number of points."""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        
        sim.set_parameters({'k': 1.0})
        sim.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        
        x = sp.Symbol('x')
        sim.compile_equations({'x_ddot': -x}, ['x'])
        
        result = sim.simulate((0, 1), num_points=100)
        assert len(result['t']) == 100


class TestSimulatorValidation:
    """Tests for input validation."""

    def test_invalid_time_span_order(self):
        """Test that reversed time span raises error."""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        
        sim.set_parameters({'k': 1.0})
        sim.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        
        x = sp.Symbol('x')
        sim.compile_equations({'x_ddot': -x}, ['x'])
        
        with pytest.raises(ValueError):
            sim.simulate((5, 0), num_points=10)

    def test_negative_time_raises_error(self):
        """Test that negative start time raises error."""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        
        sim.set_parameters({'k': 1.0})
        sim.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        
        x = sp.Symbol('x')
        sim.compile_equations({'x_ddot': -x}, ['x'])
        
        with pytest.raises(ValueError):
            sim.simulate((-1, 5), num_points=10)


class TestEquationsOfMotion:
    """Tests for equations of motion computation."""

    def test_eom_returns_array(self):
        """Test that equations_of_motion returns numpy array."""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        
        sim.set_parameters({'k': 1.0})
        sim.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        
        x = sp.Symbol('x')
        sim.compile_equations({'x_ddot': -x}, ['x'])
        
        y = np.array([1.0, 0.0])
        result = sim.equations_of_motion(0.0, y)
        assert isinstance(result, np.ndarray)

    def test_eom_correct_length(self):
        """Test that equations_of_motion returns correct length."""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        
        sim.set_parameters({'k': 1.0})
        sim.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        
        x = sp.Symbol('x')
        sim.compile_equations({'x_ddot': -x}, ['x'])
        
        y = np.array([1.0, 0.0])
        result = sim.equations_of_motion(0.0, y)
        assert len(result) == 2


class TestSimulatorAttributeDefaults:
    """Tests for simulator attribute defaults."""

    def test_default_attributes(self):
        """Test that simulator has correct default values."""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        assert sim.parameters == {}
        assert sim.initial_conditions == {}
        assert sim.constraints == []
        assert sim.coordinates == []
        assert sim.use_hamiltonian is False

    def test_use_hamiltonian_flag(self):
        """Test that use_hamiltonian flag can be set."""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        sim.use_hamiltonian = True
        assert sim.use_hamiltonian is True
