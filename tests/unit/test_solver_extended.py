"""
Extended unit tests for the solver module.

Tests the NumericalSimulator class with more coverage for edge cases
and different simulation scenarios.
"""

import pytest
import numpy as np
import sympy as sp

from mechanics_dsl.solver import NumericalSimulator
from mechanics_dsl.symbolic import SymbolicEngine


@pytest.fixture
def engine():
    """Create a symbolic engine."""
    return SymbolicEngine()


@pytest.fixture
def simulator(engine):
    """Create a numerical simulator."""
    return NumericalSimulator(engine)


class TestSimulatorInit:
    """Tests for NumericalSimulator initialization."""
    
    def test_init(self, engine):
        sim = NumericalSimulator(engine)
        assert sim is not None
    
    def test_init_has_parameters(self, simulator):
        assert hasattr(simulator, 'parameters')
    
    def test_init_has_initial_conditions(self, simulator):
        assert hasattr(simulator, 'initial_conditions')


class TestSetParameters:
    """Tests for set_parameters method."""
    
    def test_set_single_parameter(self, simulator):
        simulator.set_parameters({'m': 1.0})
        assert simulator.parameters['m'] == 1.0
    
    def test_set_multiple_parameters(self, simulator):
        simulator.set_parameters({'m': 1.0, 'k': 10.0, 'c': 0.1})
        assert simulator.parameters['m'] == 1.0
        assert simulator.parameters['k'] == 10.0
        assert simulator.parameters['c'] == 0.1
    
    def test_update_parameters(self, simulator):
        simulator.set_parameters({'m': 1.0})
        simulator.set_parameters({'k': 2.0})
        assert 'm' in simulator.parameters
        assert 'k' in simulator.parameters
    
    def test_overwrite_parameter(self, simulator):
        simulator.set_parameters({'m': 1.0})
        simulator.set_parameters({'m': 2.0})
        assert simulator.parameters['m'] == 2.0


class TestSetInitialConditions:
    """Tests for set_initial_conditions method."""
    
    def test_set_initial_conditions(self, simulator):
        simulator.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        assert simulator.initial_conditions['x'] == 1.0
        assert simulator.initial_conditions['x_dot'] == 0.0
    
    def test_update_initial_conditions(self, simulator):
        simulator.set_initial_conditions({'x': 1.0})
        simulator.set_initial_conditions({'y': 2.0})
        assert 'x' in simulator.initial_conditions
        assert 'y' in simulator.initial_conditions


class TestSimulate:
    """Tests for simulate method."""
    
    def test_simple_oscillator(self, simulator):
        x = sp.Symbol('x')
        simulator.set_parameters({'k': 1.0, 'm': 1.0})
        simulator.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        simulator.compile_equations({'x_ddot': -x}, ['x'])
        
        result = simulator.simulate((0, 5), num_points=50)
        
        assert result['success'] is True
        assert 't' in result
        assert 'y' in result
        assert len(result['t']) == 50
    
    def test_with_rk45(self, simulator):
        x = sp.Symbol('x')
        simulator.set_parameters({'k': 1.0, 'm': 1.0})
        simulator.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        simulator.compile_equations({'x_ddot': -x}, ['x'])
        
        result = simulator.simulate((0, 5), num_points=50, method='RK45')
        assert result['success'] is True
    
    def test_with_tolerances(self, simulator):
        x = sp.Symbol('x')
        simulator.set_parameters({'k': 1.0, 'm': 1.0})
        simulator.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        simulator.compile_equations({'x_ddot': -x}, ['x'])
        
        result = simulator.simulate((0, 5), num_points=50, rtol=1e-8, atol=1e-10)
        assert result['success'] is True
    
    def test_damped_oscillator(self, simulator):
        x, x_dot = sp.symbols('x x_dot')
        simulator.set_parameters({'k': 1.0, 'c': 0.1, 'm': 1.0})
        simulator.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        simulator.compile_equations({'x_ddot': -x - 0.1*x_dot}, ['x'])
        
        result = simulator.simulate((0, 10), num_points=100)
        assert result['success'] is True
    
    def test_strongly_damped(self, simulator):
        x, x_dot = sp.symbols('x x_dot')
        simulator.set_parameters({'k': 1.0, 'c': 2.0, 'm': 1.0})
        simulator.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        simulator.compile_equations({'x_ddot': -x - 2.0*x_dot}, ['x'])
        
        result = simulator.simulate((0, 10), num_points=100)
        assert result['success'] is True
    
    def test_large_number_of_points(self, simulator):
        x = sp.Symbol('x')
        simulator.set_parameters({'k': 1.0, 'm': 1.0})
        simulator.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        simulator.compile_equations({'x_ddot': -x}, ['x'])
        
        result = simulator.simulate((0, 10), num_points=1000)
        assert result['success'] is True
        assert len(result['t']) == 1000
    
    def test_short_time_span(self, simulator):
        x = sp.Symbol('x')
        simulator.set_parameters({'k': 1.0, 'm': 1.0})
        simulator.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        simulator.compile_equations({'x_ddot': -x}, ['x'])
        
        result = simulator.simulate((0, 0.1), num_points=20)
        assert result['success'] is True


class TestMultiCoordinateSimulation:
    """Tests for multi-coordinate systems."""
    
    def test_coupled_oscillators(self, engine):
        sim = NumericalSimulator(engine)
        
        x1, x2 = sp.symbols('x1 x2')
        sim.set_parameters({'k': 1.0, 'm': 1.0})
        sim.set_initial_conditions({
            'x1': 1.0, 'x1_dot': 0.0,
            'x2': 0.0, 'x2_dot': 0.0
        })
        sim.compile_equations({
            'x1_ddot': -(2*x1 - x2),
            'x2_ddot': -(2*x2 - x1)
        }, ['x1', 'x2'])
        
        result = sim.simulate((0, 10), num_points=100)
        assert result['success'] is True
        assert result['y'].shape[0] == 4


class TestPendulumSimulation:
    """Tests for pendulum-like systems."""
    
    def test_simple_pendulum_small_angle(self, engine):
        sim = NumericalSimulator(engine)
        
        theta = sp.Symbol('theta')
        sim.set_parameters({'g': 9.81, 'l': 1.0})
        sim.set_initial_conditions({'theta': 0.1, 'theta_dot': 0.0})
        
        sim.compile_equations({'theta_ddot': -9.81 * theta}, ['theta'])
        
        result = sim.simulate((0, 5), num_points=100)
        assert result['success'] is True
        assert 'theta' in result['coordinates']


class TestEquationsOfMotion:
    """Tests for equations_of_motion method."""
    
    def test_eom_returns_array(self, simulator):
        x = sp.Symbol('x')
        simulator.set_parameters({'k': 1.0})
        simulator.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        simulator.compile_equations({'x_ddot': -x}, ['x'])
        
        y = np.array([1.0, 0.5])
        dydt = simulator.equations_of_motion(0.0, y)
        
        assert isinstance(dydt, np.ndarray)
        assert len(dydt) == 2


class TestSimulationDiagnostics:
    """Tests for simulation diagnostics."""
    
    def test_result_has_coordinates(self, simulator):
        x = sp.Symbol('x')
        simulator.set_parameters({'k': 1.0})
        simulator.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        simulator.compile_equations({'x_ddot': -x}, ['x'])
        
        result = simulator.simulate((0, 5), num_points=50)
        
        assert 'coordinates' in result
        assert 'x' in result['coordinates']
    
    def test_result_shape(self, simulator):
        x = sp.Symbol('x')
        simulator.set_parameters({'k': 1.0})
        simulator.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        simulator.compile_equations({'x_ddot': -x}, ['x'])
        
        result = simulator.simulate((0, 5), num_points=100)
        
        assert result['y'].shape[1] == 100
        assert len(result['t']) == 100


class TestNonlinearSystems:
    """Tests for nonlinear systems."""
    
    def test_nonlinear_spring(self, simulator):
        x = sp.Symbol('x')
        simulator.set_parameters({'k': 1.0, 'alpha': 0.1})
        simulator.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        simulator.compile_equations({'x_ddot': -x - 0.1*x**3}, ['x'])
        
        result = simulator.simulate((0, 10), num_points=100)
        assert result['success'] is True
    
    def test_vanderpol(self, simulator):
        x, x_dot = sp.symbols('x x_dot')
        simulator.set_parameters({'mu': 1.0})
        simulator.set_initial_conditions({'x': 2.0, 'x_dot': 0.0})
        # Van der Pol: x'' - mu*(1-x^2)*x' + x = 0
        simulator.compile_equations({'x_ddot': 1.0*(1 - x**2)*x_dot - x}, ['x'])
        
        result = simulator.simulate((0, 20), num_points=500)
        assert result['success'] is True


class TestDifferentMethods:
    """Tests for different integration methods."""
    
    def test_rk45(self, simulator):
        x = sp.Symbol('x')
        simulator.set_parameters({'k': 1.0})
        simulator.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        simulator.compile_equations({'x_ddot': -x}, ['x'])
        
        result = simulator.simulate((0, 5), num_points=50, method='RK45')
        assert result['success'] is True
    
    def test_rk23(self, simulator):
        x = sp.Symbol('x')
        simulator.set_parameters({'k': 1.0})
        simulator.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        simulator.compile_equations({'x_ddot': -x}, ['x'])
        
        result = simulator.simulate((0, 5), num_points=50, method='RK23')
        assert result['success'] is True
    
    def test_dop853(self, simulator):
        x = sp.Symbol('x')
        simulator.set_parameters({'k': 1.0})
        simulator.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        simulator.compile_equations({'x_ddot': -x}, ['x'])
        
        result = simulator.simulate((0, 5), num_points=50, method='DOP853')
        assert result['success'] is True
