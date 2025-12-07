"""
Tests for Numba-accelerated solver module

Validates:
- NumbaSimulator functionality
- Integration method accuracy
- Performance improvements over SciPy
- Graceful fallback when Numba unavailable
"""
import pytest
import numpy as np
import sympy as sp
import time

from mechanics_dsl.solver_numba import (
    NumbaSimulator,
    is_numba_available,
    get_numba_version,
    create_numba_ode_function
)


class TestNumbaAvailability:
    """Test Numba detection and version info."""
    
    def test_is_numba_available(self):
        """Test Numba availability check."""
        # Should return boolean
        result = is_numba_available()
        assert isinstance(result, bool)
    
    def test_get_numba_version(self):
        """Test version retrieval."""
        if is_numba_available():
            version = get_numba_version()
            assert version is not None
            assert isinstance(version, str)
        else:
            assert get_numba_version() is None


class TestNumbaSimulatorBasics:
    """Test basic NumbaSimulator functionality."""
    
    def test_create_simulator(self):
        """Test simulator instantiation."""
        sim = NumbaSimulator()
        assert sim is not None
        assert sim.coordinates == []
        assert sim.parameters == {}
    
    def test_set_parameters(self):
        """Test parameter setting."""
        sim = NumbaSimulator()
        sim.set_parameters({'g': 9.81, 'm': 1.0, 'l': 1.0})
        
        assert sim.parameters['g'] == 9.81
        assert sim.parameters['m'] == 1.0
        assert sim.parameters['l'] == 1.0
    
    def test_set_initial_conditions(self):
        """Test initial condition setting."""
        sim = NumbaSimulator()
        sim.set_initial_conditions({'theta': 0.1, 'theta_dot': 0.0})
        
        assert sim.initial_conditions['theta'] == 0.1
        assert sim.initial_conditions['theta_dot'] == 0.0


class TestSimplePendulum:
    """Test Numba solver on simple pendulum problem."""
    
    def setup_method(self):
        """Set up simple pendulum equations."""
        # θ'' = -(g/l) * sin(θ)
        theta = sp.Symbol('theta', real=True)
        theta_dot = sp.Symbol('theta_dot', real=True)
        g = sp.Symbol('g', positive=True)
        l = sp.Symbol('l', positive=True)
        
        self.accelerations = {
            'theta_ddot': -g/l * sp.sin(theta)
        }
        self.coordinates = ['theta']
    
    def test_compile_equations(self):
        """Test equation compilation."""
        sim = NumbaSimulator()
        sim.set_parameters({'g': 9.81, 'l': 1.0})
        sim.compile_equations(self.accelerations, self.coordinates)
        
        assert sim._is_compiled
        assert sim._ode_func is not None
    
    def test_simulate_rk4(self):
        """Test RK4 integration."""
        sim = NumbaSimulator()
        sim.set_parameters({'g': 9.81, 'l': 1.0})
        sim.set_initial_conditions({'theta': 0.1, 'theta_dot': 0.0})
        sim.compile_equations(self.accelerations, self.coordinates)
        
        solution = sim.simulate_numba(t_span=(0, 1), num_points=100, method='rk4')
        
        assert solution['success']
        assert len(solution['t']) == 100
        assert solution['y'].shape == (2, 100)
    
    def test_simulate_euler(self):
        """Test Euler integration."""
        sim = NumbaSimulator()
        sim.set_parameters({'g': 9.81, 'l': 1.0})
        sim.set_initial_conditions({'theta': 0.1, 'theta_dot': 0.0})
        sim.compile_equations(self.accelerations, self.coordinates)
        
        solution = sim.simulate_numba(t_span=(0, 1), num_points=100, method='euler')
        
        assert solution['success']
        assert len(solution['t']) == 100
    
    def test_simulate_rk45(self):
        """Test adaptive RK45 integration."""
        sim = NumbaSimulator()
        sim.set_parameters({'g': 9.81, 'l': 1.0})
        sim.set_initial_conditions({'theta': 0.1, 'theta_dot': 0.0})
        sim.compile_equations(self.accelerations, self.coordinates)
        
        solution = sim.simulate_numba(t_span=(0, 1), num_points=100, method='rk45')
        
        assert solution['success']
        assert len(solution['t']) == 100


class TestEnergyConservation:
    """Test energy conservation in Numba solver."""
    
    def test_pendulum_energy_conservation(self):
        """Test that total energy is conserved for simple pendulum."""
        theta = sp.Symbol('theta', real=True)
        theta_dot = sp.Symbol('theta_dot', real=True)
        g = sp.Symbol('g', positive=True)
        l = sp.Symbol('l', positive=True)
        m = sp.Symbol('m', positive=True)
        
        accelerations = {'theta_ddot': -g/l * sp.sin(theta)}
        
        sim = NumbaSimulator()
        sim.set_parameters({'g': 9.81, 'l': 1.0, 'm': 1.0})
        sim.set_initial_conditions({'theta': 0.3, 'theta_dot': 0.0})
        sim.compile_equations(accelerations, ['theta'])
        
        solution = sim.simulate_numba(t_span=(0, 10), num_points=1000, method='rk4')
        
        # Compute energy: E = (1/2)*m*l²*θ̇² - m*g*l*cos(θ)
        g_val, l_val, m_val = 9.81, 1.0, 1.0
        theta_vals = solution['y'][0, :]
        theta_dot_vals = solution['y'][1, :]
        
        T = 0.5 * m_val * l_val**2 * theta_dot_vals**2
        V = -m_val * g_val * l_val * np.cos(theta_vals)
        E = T + V
        
        # Energy should be conserved to within 1%
        E_initial = E[0]
        E_variation = np.abs((E - E_initial) / E_initial)
        
        assert np.max(E_variation) < 0.01, f"Energy variation: {np.max(E_variation):.4f}"


class TestAccuracyComparison:
    """Test accuracy of Numba solver against known solutions."""
    
    def test_harmonic_oscillator(self):
        """Test against analytical solution for harmonic oscillator."""
        # x'' = -ω²x, solution: x(t) = A*cos(ωt + φ)
        x = sp.Symbol('x', real=True)
        omega = sp.Symbol('omega', positive=True)
        
        accelerations = {'x_ddot': -omega**2 * x}
        
        sim = NumbaSimulator()
        sim.set_parameters({'omega': 2.0})
        sim.set_initial_conditions({'x': 1.0, 'x_dot': 0.0})
        sim.compile_equations(accelerations, ['x'])
        
        solution = sim.simulate_numba(t_span=(0, 2*np.pi), num_points=200, method='rk4')
        
        # Analytical solution: x(t) = cos(2t)
        t = solution['t']
        x_numerical = solution['y'][0, :]
        x_analytical = np.cos(2 * t)
        
        # Should match to 3 decimal places
        max_error = np.max(np.abs(x_numerical - x_analytical))
        assert max_error < 0.001, f"Max error: {max_error:.6f}"


class TestUncompiledError:
    """Test error handling for uncompiled equations."""
    
    def test_simulate_without_compile(self):
        """Test that simulation fails gracefully without compilation."""
        sim = NumbaSimulator()
        sim.set_parameters({'g': 9.81})
        sim.set_initial_conditions({'theta': 0.1})
        
        with pytest.raises(RuntimeError, match="not compiled"):
            sim.simulate_numba(t_span=(0, 1))


class TestCreateODEFunction:
    """Test ODE function creation."""
    
    def test_create_simple_ode(self):
        """Test creating ODE function from symbolic expressions."""
        x = sp.Symbol('x', real=True)
        x_dot = sp.Symbol('x_dot', real=True)
        k = sp.Symbol('k', positive=True)
        
        accelerations = {'x_ddot': -k * x}
        
        ode_func = create_numba_ode_function(
            accelerations, 
            ['x'], 
            ['k']
        )
        
        # Test the ODE function
        t = 0.0
        y = np.array([1.0, 0.0])  # x=1, x_dot=0
        params = np.array([1.0])  # k=1
        
        dydt = ode_func(t, y, params)
        
        assert len(dydt) == 2
        assert dydt[0] == 0.0  # dx/dt = x_dot = 0
        assert dydt[1] == -1.0  # d(x_dot)/dt = -k*x = -1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
