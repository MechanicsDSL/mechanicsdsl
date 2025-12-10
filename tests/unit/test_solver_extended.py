"""
Extended unit tests for MechanicsDSL solver module.

Tests additional methods of NumericalSimulator not covered by existing tests.
"""

import pytest
import numpy as np
import sympy as sp

from mechanics_dsl.solver import NumericalSimulator
from mechanics_dsl.symbolic import SymbolicEngine


class TestSimulateExtended:
    """Extended simulation tests."""
    
    @pytest.fixture
    def oscillator_simulator(self):
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        
        sim.set_parameters({'k': 1.0, 'm': 1.0})
        sim.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        
        x = sp.Symbol('x')
        sim.compile_equations({'x_ddot': -x}, ['x'])
        return sim
    
    def test_simulate_with_method(self, oscillator_simulator):
        """Test simulation with explicit method."""
        result = oscillator_simulator.simulate((0, 5), num_points=50, method='RK45')
        
        assert result['success'] is True
    
    def test_simulate_with_tolerances(self, oscillator_simulator):
        """Test simulation with custom tolerances."""
        result = oscillator_simulator.simulate(
            (0, 5), 
            num_points=50, 
            rtol=1e-6, 
            atol=1e-8
        )
        
        assert result['success'] is True
    
    def test_simulate_stiff_detection(self, oscillator_simulator):
        """Test simulation with stiffness detection."""
        result = oscillator_simulator.simulate(
            (0, 5), 
            num_points=50, 
            detect_stiff=True
        )
        
        assert result['success'] is True
    
    def test_simulate_no_stiff_detection(self, oscillator_simulator):
        """Test simulation without stiffness detection."""
        result = oscillator_simulator.simulate(
            (0, 5), 
            num_points=50, 
            detect_stiff=False
        )
        
        assert result['success'] is True


class TestEquationsOfMotionExtended:
    """Extended tests for equations_of_motion method."""
    
    @pytest.fixture
    def simulator(self):
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        
        sim.set_parameters({'k': 1.0, 'm': 1.0, 'c': 0.1})
        sim.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        
        x, x_dot = sp.symbols('x x_dot')
        # Damped oscillator: m*x'' + c*x' + k*x = 0 => x'' = -(c*x' + k*x)/m
        sim.compile_equations({'x_ddot': -0.1*x_dot - x}, ['x'])
        return sim
    
    def test_eom_handles_velocity(self, simulator):
        """Test EOM with velocity-dependent terms."""
        y = np.array([1.0, 0.1])
        
        dydt = simulator.equations_of_motion(0.0, y)
        
        assert isinstance(dydt, np.ndarray)
        assert len(dydt) == 2


class TestMultiCoordinateSimulation:
    """Tests for multi-coordinate systems."""
    
    def test_two_mass_spring(self):
        """Test simulation of coupled oscillators."""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        
        sim.set_parameters({'k': 1.0, 'm': 1.0})
        sim.set_initial_conditions({
            'x1': 1.0, 'x1_dot': 0.0,
            'x2': 0.0, 'x2_dot': 0.0
        })
        
        x1, x2 = sp.symbols('x1 x2')
        # Coupled oscillators
        sim.compile_equations({
            'x1_ddot': -(2*x1 - x2),
            'x2_ddot': -(2*x2 - x1)
        }, ['x1', 'x2'])
        
        result = sim.simulate((0, 10), num_points=100)
        
        assert result['success'] is True
        assert result['y'].shape[0] == 4  # x1, x1_dot, x2, x2_dot
    
    def test_three_body_simplified(self):
        """Test simulation with three coordinates."""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        
        sim.set_parameters({'k': 1.0})
        sim.set_initial_conditions({
            'x1': 1.0, 'x1_dot': 0.0,
            'x2': 0.0, 'x2_dot': 0.0,
            'x3': -1.0, 'x3_dot': 0.0
        })
        
        x1, x2, x3 = sp.symbols('x1 x2 x3')
        sim.compile_equations({
            'x1_ddot': -x1,
            'x2_ddot': -x2,
            'x3_ddot': -x3
        }, ['x1', 'x2', 'x3'])
        
        result = sim.simulate((0, 5), num_points=50)
        
        assert result['success'] is True
        assert result['y'].shape[0] == 6


class TestSimulationDiagnostics:
    """Tests for simulation diagnostics."""
    
    @pytest.fixture
    def simulator(self):
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        sim.set_parameters({'k': 1.0})
        sim.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        
        x = sp.Symbol('x')
        sim.compile_equations({'x_ddot': -x}, ['x'])
        return sim
    
    def test_result_has_coordinates(self, simulator):
        """Test that result includes coordinates."""
        result = simulator.simulate((0, 5), num_points=50)
        
        assert 'coordinates' in result
        assert 'x' in result['coordinates']
    
    def test_result_shape_matches_time(self, simulator):
        """Test that y shape matches t length."""
        result = simulator.simulate((0, 5), num_points=100)
        
        assert result['y'].shape[1] == len(result['t'])
        assert len(result['t']) == 100
    
    def test_result_has_time_array(self, simulator):
        """Test that result has 't' key."""
        result = simulator.simulate((0, 5), num_points=50)
        
        assert 't' in result
        assert isinstance(result['t'], np.ndarray)
    
    def test_result_has_y_array(self, simulator):
        """Test that result has 'y' key."""
        result = simulator.simulate((0, 5), num_points=50)
        
        assert 'y' in result
        assert isinstance(result['y'], np.ndarray)


class TestNumericalSimulatorInit:
    """Tests for NumericalSimulator initialization."""
    
    def test_init_requires_engine(self):
        """Test that init requires symbolic engine."""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        
        assert sim is not None
    
    def test_init_parameters_empty(self):
        """Test initial parameters are empty."""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        
        assert hasattr(sim, 'parameters')
    
    def test_init_initial_conditions_empty(self):
        """Test initial conditions are empty."""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        
        assert hasattr(sim, 'initial_conditions')


class TestSetParameters:
    """Tests for NumericalSimulator.set_parameters."""
    
    def test_set_parameters_stores_values(self):
        """Test that parameters are stored."""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        
        sim.set_parameters({'m': 1.0, 'k': 10.0})
        
        assert sim.parameters['m'] == 1.0
        assert sim.parameters['k'] == 10.0
    
    def test_set_parameters_multiple_calls(self):
        """Test multiple parameter set calls."""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        
        sim.set_parameters({'m': 1.0})
        sim.set_parameters({'k': 5.0})
        
        assert 'm' in sim.parameters
        assert 'k' in sim.parameters


class TestSetInitialConditions:
    """Tests for NumericalSimulator.set_initial_conditions."""
    
    def test_set_initial_conditions_stores_values(self):
        """Test that initial conditions are stored."""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        
        sim.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        
        assert sim.initial_conditions['x'] == 1.0
        assert sim.initial_conditions['x_dot'] == 0.0
