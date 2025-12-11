"""
Comprehensive tests for solver.py to achieve 90%+ coverage

Tests NumericalSimulator methods including:
- compile_equations and compile_hamiltonian_equations
- _replace_derivatives
- equations_of_motion and _hamiltonian_ode
- _select_optimal_solver
- simulate with various inputs and error conditions
"""

import pytest
import numpy as np
import sympy as sp
from unittest.mock import patch, MagicMock

from mechanics_dsl.solver import NumericalSimulator
from mechanics_dsl.symbolic import SymbolicEngine
from mechanics_dsl.utils import config


@pytest.fixture
def symbolic_engine():
    """Create a SymbolicEngine instance."""
    return SymbolicEngine()


@pytest.fixture
def simulator(symbolic_engine):
    """Create a NumericalSimulator instance."""
    return NumericalSimulator(symbolic_engine)


@pytest.fixture
def oscillator_setup(simulator):
    """Setup a simple harmonic oscillator."""
    simulator.set_parameters({'m': 1.0, 'k': 4.0})
    simulator.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
    
    m = simulator.symbolic.get_symbol('m')
    k = simulator.symbolic.get_symbol('k')
    x = simulator.symbolic.get_symbol('x')
    
    accelerations = {'x_ddot': -k * x / m}
    simulator.compile_equations(accelerations, ['x'])
    return simulator


# ============================================================================
# INITIALIZATION AND BASIC METHODS
# ============================================================================

class TestNumericalSimulatorInit:
    """Tests for NumericalSimulator initialization."""
    
    def test_init(self, symbolic_engine):
        sim = NumericalSimulator(symbolic_engine)
        assert sim is not None
        assert sim.symbolic is symbolic_engine
    
    def test_init_attributes(self, simulator):
        assert simulator.equations == {}
        assert simulator.parameters == {}
        assert simulator.initial_conditions == {}
        assert simulator.constraints == []
        assert simulator.state_vars == []
        assert simulator.coordinates == []
        assert simulator.use_hamiltonian is False


class TestSetParameters:
    """Tests for set_parameters method."""
    
    def test_set_single_parameter(self, simulator):
        simulator.set_parameters({'m': 1.0})
        assert simulator.parameters['m'] == 1.0
    
    def test_set_multiple_parameters(self, simulator):
        simulator.set_parameters({'m': 1.0, 'k': 2.0, 'g': 9.81})
        assert simulator.parameters['m'] == 1.0
        assert simulator.parameters['k'] == 2.0
        assert simulator.parameters['g'] == 9.81
    
    def test_update_parameters(self, simulator):
        simulator.set_parameters({'m': 1.0})
        simulator.set_parameters({'m': 2.0})
        assert simulator.parameters['m'] == 2.0


class TestSetInitialConditions:
    """Tests for set_initial_conditions method."""
    
    def test_set_conditions(self, simulator):
        simulator.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        assert simulator.initial_conditions['x'] == 1.0
        assert simulator.initial_conditions['x_dot'] == 0.0


class TestAddConstraint:
    """Tests for add_constraint method."""
    
    def test_add_constraint(self, simulator):
        constraint = sp.Symbol('x') - 1
        simulator.add_constraint(constraint)
        assert len(simulator.constraints) == 1
        assert simulator.constraints[0] == constraint


# ============================================================================
# COMPILE EQUATIONS
# ============================================================================

class TestCompileEquations:
    """Tests for compile_equations method."""
    
    def test_simple_oscillator(self, simulator):
        simulator.set_parameters({'k': 4.0, 'm': 1.0})
        
        m = simulator.symbolic.get_symbol('m')
        k = simulator.symbolic.get_symbol('k')
        x = simulator.symbolic.get_symbol('x')
        
        accelerations = {'x_ddot': -k * x / m}
        simulator.compile_equations(accelerations, ['x'])
        
        assert 'x_ddot' in simulator.equations
        assert simulator.coordinates == ['x']
    
    def test_multiple_coordinates(self, simulator):
        simulator.set_parameters({'m': 1.0})
        
        m = simulator.symbolic.get_symbol('m')
        x = simulator.symbolic.get_symbol('x')
        y = simulator.symbolic.get_symbol('y')
        
        accelerations = {
            'x_ddot': -x / m,
            'y_ddot': -y / m
        }
        simulator.compile_equations(accelerations, ['x', 'y'])
        
        assert 'x_ddot' in simulator.equations
        assert 'y_ddot' in simulator.equations
    
    def test_constant_acceleration(self, simulator):
        simulator.set_parameters({'g': 9.81})
        
        g = simulator.symbolic.get_symbol('g')
        accelerations = {'y_ddot': -g}
        simulator.compile_equations(accelerations, ['y'])
        
        assert 'y_ddot' in simulator.equations
    
    def test_time_dependent(self, simulator):
        t = simulator.symbolic.time_symbol
        accelerations = {'x_ddot': sp.cos(t)}
        simulator.compile_equations(accelerations, ['x'])
        
        assert 'x_ddot' in simulator.equations
    
    def test_empty_coordinates(self, simulator):
        simulator.compile_equations({}, [])
        assert simulator.coordinates == []
    
    def test_missing_acceleration(self, simulator):
        simulator.set_parameters({'k': 1.0})
        accelerations = {}  # No acceleration for x
        simulator.compile_equations(accelerations, ['x'])
        # Should not fail, just logs warning


# ============================================================================
# COMPILE HAMILTONIAN EQUATIONS
# ============================================================================

class TestCompileHamiltonianEquations:
    """Tests for compile_hamiltonian_equations method."""
    
    def test_simple_hamiltonian(self, simulator):
        simulator.set_parameters({'m': 1.0, 'k': 4.0})
        
        m = simulator.symbolic.get_symbol('m')
        k = simulator.symbolic.get_symbol('k')
        q = simulator.symbolic.get_symbol('q')
        p_q = simulator.symbolic.get_symbol('p_q')
        
        q_dots = [p_q / m]
        p_dots = [-k * q]
        
        simulator.compile_hamiltonian_equations(q_dots, p_dots, ['q'])
        
        assert simulator.use_hamiltonian is True
        assert 'q_dots' in simulator.hamiltonian_equations
        assert 'p_dots' in simulator.hamiltonian_equations
    
    def test_constant_equations(self, simulator):
        q_dots = [sp.Float(1.0)]
        p_dots = [sp.Float(0.0)]
        
        simulator.compile_hamiltonian_equations(q_dots, p_dots, ['x'])
        
        assert simulator.use_hamiltonian is True


# ============================================================================
# REPLACE DERIVATIVES
# ============================================================================

class TestReplaceDerivatives:
    """Tests for _replace_derivatives method."""
    
    def test_no_derivatives(self, simulator):
        expr = sp.Symbol('x') + 1
        result = simulator._replace_derivatives(expr, ['x'])
        assert result == expr
    
    def test_first_derivative(self, simulator):
        t = simulator.symbolic.time_symbol
        x = sp.Function('x')(t)
        expr = sp.diff(x, t)
        
        result = simulator._replace_derivatives(expr, ['x'])
        # Should attempt replacement
        assert result is not None
    
    def test_second_derivative(self, simulator):
        t = simulator.symbolic.time_symbol
        x = sp.Function('x')(t)
        expr = sp.diff(x, t, 2)
        
        result = simulator._replace_derivatives(expr, ['x'])
        assert result is not None


# ============================================================================
# EQUATIONS OF MOTION
# ============================================================================

class TestEquationsOfMotion:
    """Tests for equations_of_motion method."""
    
    def test_valid_input(self, oscillator_setup):
        y = np.array([1.0, 0.0])
        dydt = oscillator_setup.equations_of_motion(0.0, y)
        
        assert isinstance(dydt, np.ndarray)
        assert dydt.shape == y.shape
    
    def test_invalid_time(self, oscillator_setup):
        y = np.array([1.0, 0.0])
        dydt = oscillator_setup.equations_of_motion(np.nan, y)
        
        assert isinstance(dydt, np.ndarray)
    
    def test_none_state(self, oscillator_setup):
        dydt = oscillator_setup.equations_of_motion(0.0, None)
        
        assert isinstance(dydt, np.ndarray)
    
    def test_non_array_state(self, oscillator_setup):
        dydt = oscillator_setup.equations_of_motion(0.0, [1.0, 0.0])
        
        assert isinstance(dydt, np.ndarray)
    
    def test_non_finite_state(self, oscillator_setup):
        y = np.array([np.nan, np.inf])
        dydt = oscillator_setup.equations_of_motion(0.0, y)
        
        assert isinstance(dydt, np.ndarray)
        assert np.all(np.isfinite(dydt))
    
    def test_wrong_size_smaller(self, oscillator_setup):
        y = np.array([1.0])  # Too small
        dydt = oscillator_setup.equations_of_motion(0.0, y)
        
        assert isinstance(dydt, np.ndarray)
    
    def test_wrong_size_larger(self, oscillator_setup):
        y = np.array([1.0, 0.0, 0.5, 0.5])  # Too large
        dydt = oscillator_setup.equations_of_motion(0.0, y)
        
        assert isinstance(dydt, np.ndarray)
    
    def test_empty_coordinates(self, simulator):
        simulator.coordinates = []
        dydt = simulator.equations_of_motion(0.0, np.array([]))
        
        assert isinstance(dydt, np.ndarray)


# ============================================================================
# HAMILTONIAN ODE
# ============================================================================

class TestHamiltonianODE:
    """Tests for _hamiltonian_ode method."""
    
    def test_valid_hamiltonian(self, simulator):
        simulator.set_parameters({'m': 1.0, 'k': 4.0})
        simulator.set_initial_conditions({'q': 1.0, 'p_q': 0.0})
        
        m = simulator.symbolic.get_symbol('m')
        k = simulator.symbolic.get_symbol('k')
        q = simulator.symbolic.get_symbol('q')
        p_q = simulator.symbolic.get_symbol('p_q')
        
        simulator.compile_hamiltonian_equations([p_q / m], [-k * q], ['q'])
        
        y = np.array([1.0, 0.0])
        dydt = simulator._hamiltonian_ode(0.0, y)
        
        assert isinstance(dydt, np.ndarray)
    
    def test_invalid_time(self, simulator):
        simulator.use_hamiltonian = True
        simulator.coordinates = ['x']
        simulator.hamiltonian_equations = None
        
        dydt = simulator._hamiltonian_ode(np.nan, np.array([1.0, 0.0]))
        assert isinstance(dydt, np.ndarray)
    
    def test_none_state(self, simulator):
        simulator.use_hamiltonian = True
        simulator.coordinates = ['x']
        
        dydt = simulator._hamiltonian_ode(0.0, None)
        assert isinstance(dydt, np.ndarray)
    
    def test_missing_equations(self, simulator):
        simulator.use_hamiltonian = True
        simulator.coordinates = ['x']
        simulator.hamiltonian_equations = None
        
        dydt = simulator._hamiltonian_ode(0.0, np.array([1.0, 0.0]))
        assert isinstance(dydt, np.ndarray)
    
    def test_invalid_equations_type(self, simulator):
        simulator.use_hamiltonian = True
        simulator.coordinates = ['x']
        simulator.hamiltonian_equations = "not a dict"
        
        dydt = simulator._hamiltonian_ode(0.0, np.array([1.0, 0.0]))
        assert isinstance(dydt, np.ndarray)
    
    def test_missing_keys(self, simulator):
        simulator.use_hamiltonian = True
        simulator.coordinates = ['x']
        simulator.hamiltonian_equations = {}
        
        dydt = simulator._hamiltonian_ode(0.0, np.array([1.0, 0.0]))
        assert isinstance(dydt, np.ndarray)


# ============================================================================
# SELECT OPTIMAL SOLVER
# ============================================================================

class TestSelectOptimalSolver:
    """Tests for _select_optimal_solver method."""
    
    def test_adaptive_disabled(self, simulator):
        original = config.enable_adaptive_solver
        config.enable_adaptive_solver = False
        
        simulator.coordinates = ['x']
        method = simulator._select_optimal_solver((0, 10), np.array([1.0, 0.0]))
        assert method == 'LSODA'
        
        config.enable_adaptive_solver = original
    
    def test_large_system(self, simulator):
        original = config.enable_adaptive_solver
        config.enable_adaptive_solver = True
        
        simulator.coordinates = ['x' + str(i) for i in range(15)]
        method = simulator._select_optimal_solver((0, 10), np.zeros(30))
        assert method == 'LSODA'
        
        config.enable_adaptive_solver = original
    
    def test_long_time_span(self, simulator):
        original = config.enable_adaptive_solver
        config.enable_adaptive_solver = True
        
        simulator.coordinates = ['x']
        method = simulator._select_optimal_solver((0, 200), np.array([1.0, 0.0]))
        assert method == 'LSODA'
        
        config.enable_adaptive_solver = original
    
    def test_small_simple_system(self, simulator):
        original = config.enable_adaptive_solver
        config.enable_adaptive_solver = True
        
        simulator.coordinates = ['x']
        method = simulator._select_optimal_solver((0, 5), np.array([1.0, 0.0]))
        assert method == 'RK45'
        
        config.enable_adaptive_solver = original


# ============================================================================
# SIMULATE
# ============================================================================

class TestSimulate:
    """Tests for simulate method."""
    
    def test_successful_simulation(self, oscillator_setup):
        result = oscillator_setup.simulate((0, 1), num_points=100)
        
        assert result['success'] is True
        assert 't' in result
        assert 'y' in result
        assert len(result['t']) == 100
    
    def test_invalid_t_span_type(self, oscillator_setup):
        with pytest.raises(TypeError):
            oscillator_setup.simulate([0, 1])
    
    def test_invalid_t_span_order(self, oscillator_setup):
        with pytest.raises(ValueError):
            oscillator_setup.simulate((1, 0))
    
    def test_invalid_num_points_type(self, oscillator_setup):
        with pytest.raises(TypeError):
            oscillator_setup.simulate((0, 1), num_points=100.5)
    
    def test_invalid_num_points_small(self, oscillator_setup):
        with pytest.raises(ValueError):
            oscillator_setup.simulate((0, 1), num_points=1)
    
    def test_invalid_num_points_large(self, oscillator_setup):
        with pytest.raises(ValueError):
            oscillator_setup.simulate((0, 1), num_points=100_000_000)
    
    def test_invalid_method_type(self, oscillator_setup):
        with pytest.raises(TypeError):
            oscillator_setup.simulate((0, 1), method=123)
    
    def test_invalid_method_value(self, oscillator_setup):
        with pytest.raises(ValueError):
            oscillator_setup.simulate((0, 1), method='InvalidMethod')
    
    def test_valid_method(self, oscillator_setup):
        result = oscillator_setup.simulate((0, 1), method='LSODA', num_points=50)
        assert result['success'] is True
    
    def test_invalid_rtol_type(self, oscillator_setup):
        with pytest.raises(TypeError):
            oscillator_setup.simulate((0, 1), rtol="1e-6")
    
    def test_invalid_rtol_value(self, oscillator_setup):
        with pytest.raises(ValueError):
            oscillator_setup.simulate((0, 1), rtol=0.0)
        with pytest.raises(ValueError):
            oscillator_setup.simulate((0, 1), rtol=1.0)
    
    def test_invalid_atol_type(self, oscillator_setup):
        with pytest.raises(TypeError):
            oscillator_setup.simulate((0, 1), atol="1e-9")
    
    def test_invalid_atol_value(self, oscillator_setup):
        with pytest.raises(ValueError):
            oscillator_setup.simulate((0, 1), atol=0.0)
    
    def test_invalid_detect_stiff_type(self, oscillator_setup):
        with pytest.raises(TypeError):
            oscillator_setup.simulate((0, 1), detect_stiff="yes")
    
    def test_with_performance_monitoring(self, oscillator_setup):
        original = config.enable_performance_monitoring
        config.enable_performance_monitoring = True
        
        result = oscillator_setup.simulate((0, 1), num_points=50)
        assert result['success'] is True
        
        config.enable_performance_monitoring = original
    
    def test_hamiltonian_simulation(self, simulator):
        simulator.set_parameters({'m': 1.0, 'k': 4.0})
        simulator.set_initial_conditions({'q': 1.0, 'p_q': 0.0})
        
        m = simulator.symbolic.get_symbol('m')
        k = simulator.symbolic.get_symbol('k')
        q = simulator.symbolic.get_symbol('q')
        p_q = simulator.symbolic.get_symbol('p_q')
        
        simulator.compile_hamiltonian_equations([p_q / m], [-k * q], ['q'])
        
        result = simulator.simulate((0, 1), num_points=50)
        assert result['success'] is True
        assert result['use_hamiltonian'] is True


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Edge case tests."""
    
    def test_simplification_timeout_zero(self, simulator):
        original = config.simplification_timeout
        config.simplification_timeout = 0
        
        simulator.set_parameters({'k': 1.0})
        accelerations = {'x_ddot': sp.Symbol('k') * sp.Symbol('x')}
        simulator.compile_equations(accelerations, ['x'])
        
        config.simplification_timeout = original
    
    def test_compilation_with_complex_expression(self, simulator):
        x = simulator.symbolic.get_symbol('x')
        x_dot = simulator.symbolic.get_symbol('x_dot')
        t = simulator.symbolic.time_symbol
        
        # Complex expression with trig functions
        accel = sp.sin(x) * sp.cos(t) + x_dot**2
        accelerations = {'x_ddot': accel}
        simulator.compile_equations(accelerations, ['x'])
        
        assert 'x_ddot' in simulator.equations
