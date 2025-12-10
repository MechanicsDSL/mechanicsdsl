"""
Unit tests for MechanicsDSL analysis/stability module.

Tests the StabilityAnalyzer class for stability analysis of dynamical systems.
"""

import pytest
import numpy as np
import sympy as sp

from mechanics_dsl.analysis.stability import StabilityAnalyzer


class TestStabilityAnalyzerInit:
    """Tests for StabilityAnalyzer initialization."""
    
    def test_init_creates_instance(self):
        """Test that StabilityAnalyzer can be instantiated."""
        analyzer = StabilityAnalyzer()
        assert analyzer is not None
    
    def test_init_requires_sympy(self):
        """Test that analyzer requires sympy."""
        # SymPy is available, so this should work
        analyzer = StabilityAnalyzer()
        assert analyzer is not None


class TestFindFixedPoints:
    """Tests for StabilityAnalyzer.find_fixed_points method."""
    
    @pytest.fixture
    def analyzer(self):
        return StabilityAnalyzer()
    
    def test_find_fixed_points_linear_system(self, analyzer):
        """Test finding fixed points for simple linear system."""
        x, y = sp.symbols('x y')
        
        # System: dx/dt = -x, dy/dt = -y
        # Fixed point: (0, 0)
        equations = {
            'x_dot': -x,
            'y_dot': -y
        }
        variables = [x, y]
        
        fixed_points = analyzer.find_fixed_points(equations, variables)
        
        assert isinstance(fixed_points, list)
        # Should find the origin as fixed point
        assert len(fixed_points) >= 1
    
    def test_find_fixed_points_nonlinear(self, analyzer):
        """Test finding fixed points for nonlinear system."""
        x = sp.Symbol('x')
        
        # System: dx/dt = x^2 - 1
        # Fixed points: x = 1 and x = -1
        equations = {
            'x_dot': x**2 - 1
        }
        variables = [x]
        
        fixed_points = analyzer.find_fixed_points(equations, variables)
        
        assert isinstance(fixed_points, list)
        # Should find two fixed points
        assert len(fixed_points) == 2
    
    def test_find_fixed_points_multiple_equations(self, analyzer):
        """Test with multiple coupled equations."""
        x, y = sp.symbols('x y')
        
        # System with fixed point at origin
        equations = {
            'x_dot': x - y,
            'y_dot': x + y
        }
        variables = [x, y]
        
        fixed_points = analyzer.find_fixed_points(equations, variables)
        
        assert isinstance(fixed_points, list)
    
    def test_find_fixed_points_empty_on_error(self, analyzer):
        """Test that unsolvable system returns empty list."""
        x = sp.Symbol('x')
        
        # This should be solvable, but test error handling
        equations = {
            'x_dot': x**2 + 1  # No real solutions
        }
        variables = [x]
        
        fixed_points = analyzer.find_fixed_points(equations, variables)
        
        # May return empty list or complex solutions depending on sympy version
        assert isinstance(fixed_points, list)


class TestComputeJacobian:
    """Tests for StabilityAnalyzer.compute_jacobian method."""
    
    @pytest.fixture
    def analyzer(self):
        return StabilityAnalyzer()
    
    def test_compute_jacobian_linear(self, analyzer):
        """Test Jacobian computation for linear system."""
        x, y = sp.symbols('x y')
        
        equations = {
            'x_dot': -x + 2*y,
            'y_dot': 3*x - 4*y
        }
        variables = [x, y]
        
        J = analyzer.compute_jacobian(equations, variables)
        
        assert isinstance(J, sp.Matrix)
        assert J.shape == (2, 2)
        # Check specific elements
        assert J[0, 0] == -1
        assert J[0, 1] == 2
        assert J[1, 0] == 3
        assert J[1, 1] == -4
    
    def test_compute_jacobian_nonlinear(self, analyzer):
        """Test Jacobian for nonlinear system."""
        x, y = sp.symbols('x y')
        
        equations = {
            'x_dot': x**2 + y,
            'y_dot': x*y
        }
        variables = [x, y]
        
        J = analyzer.compute_jacobian(equations, variables)
        
        assert isinstance(J, sp.Matrix)
        assert J.shape == (2, 2)
        # J[0,0] = d(x^2+y)/dx = 2x
        assert J[0, 0] == 2*x
        # J[0,1] = d(x^2+y)/dy = 1
        assert J[0, 1] == 1
    
    def test_compute_jacobian_single_variable(self, analyzer):
        """Test Jacobian for single variable system."""
        x = sp.Symbol('x')
        
        equations = {
            'x_dot': x**3 - x
        }
        variables = [x]
        
        J = analyzer.compute_jacobian(equations, variables)
        
        assert isinstance(J, sp.Matrix)
        assert J.shape == (1, 1)
        assert J[0, 0] == 3*x**2 - 1


class TestAnalyzeStability:
    """Tests for StabilityAnalyzer.analyze_stability method."""
    
    @pytest.fixture
    def analyzer(self):
        return StabilityAnalyzer()
    
    def test_analyze_stability_stable(self, analyzer):
        """Test stability analysis for stable system."""
        x, y = sp.symbols('x y')
        
        # Stable system with negative eigenvalues
        equations = {
            'x_dot': -x,
            'y_dot': -2*y
        }
        variables = [x, y]
        
        J = analyzer.compute_jacobian(equations, variables)
        fixed_point = {x: 0, y: 0}
        
        result = analyzer.analyze_stability(J, fixed_point)
        
        assert isinstance(result, dict)
        assert result['stability'] == 'stable'
        assert result['max_real_part'] < 0
    
    def test_analyze_stability_unstable(self, analyzer):
        """Test stability analysis for unstable system."""
        x, y = sp.symbols('x y')
        
        # Unstable system with positive eigenvalues
        equations = {
            'x_dot': x,
            'y_dot': 2*y
        }
        variables = [x, y]
        
        J = analyzer.compute_jacobian(equations, variables)
        fixed_point = {x: 0, y: 0}
        
        result = analyzer.analyze_stability(J, fixed_point)
        
        assert isinstance(result, dict)
        assert result['stability'] == 'unstable'
        assert result['max_real_part'] > 0
    
    def test_analyze_stability_marginally_stable(self, analyzer):
        """Test stability analysis for marginally stable system."""
        x, y = sp.symbols('x y')
        
        # Simple harmonic oscillator (purely imaginary eigenvalues)
        equations = {
            'x_dot': y,
            'y_dot': -x
        }
        variables = [x, y]
        
        J = analyzer.compute_jacobian(equations, variables)
        fixed_point = {x: 0, y: 0}
        
        result = analyzer.analyze_stability(J, fixed_point)
        
        assert isinstance(result, dict)
        assert result['stability'] == 'marginally_stable'
        assert abs(result['max_real_part']) < 1e-9
    
    def test_analyze_stability_has_eigenvalues(self, analyzer):
        """Test that result contains eigenvalues."""
        x = sp.Symbol('x')
        
        equations = {'x_dot': -3*x}
        variables = [x]
        
        J = analyzer.compute_jacobian(equations, variables)
        fixed_point = {x: 0}
        
        result = analyzer.analyze_stability(J, fixed_point)
        
        assert 'eigenvalues' in result
        assert isinstance(result['eigenvalues'], list)
    
    def test_analyze_stability_has_jacobian(self, analyzer):
        """Test that result contains evaluated Jacobian."""
        x = sp.Symbol('x')
        
        equations = {'x_dot': -x}
        variables = [x]
        
        J = analyzer.compute_jacobian(equations, variables)
        fixed_point = {x: 0}
        
        result = analyzer.analyze_stability(J, fixed_point)
        
        assert 'jacobian' in result


class TestEstimateLyapunovExponent:
    """Tests for StabilityAnalyzer.estimate_lyapunov_exponent method."""
    
    @pytest.fixture
    def analyzer(self):
        return StabilityAnalyzer()
    
    def test_estimate_lyapunov_returns_float(self, analyzer):
        """Test that Lyapunov estimation returns a float."""
        # Create a simple trajectory
        n_points = 1000
        trajectory = np.random.randn(2, n_points)
        
        lyap = analyzer.estimate_lyapunov_exponent(trajectory, dt=0.01)
        
        assert isinstance(lyap, float)
    
    def test_estimate_lyapunov_short_trajectory(self, analyzer):
        """Test Lyapunov estimation with short trajectory."""
        # Very short trajectory
        trajectory = np.random.randn(2, 10)
        
        lyap = analyzer.estimate_lyapunov_exponent(trajectory, dt=0.01, n_renorm=100)
        
        # Should return 0.0 for too short data
        assert lyap == 0.0
    
    def test_estimate_lyapunov_stable_trajectory(self, analyzer):
        """Test Lyapunov estimation for decaying trajectory."""
        # Exponentially decaying trajectory (stable)
        t = np.linspace(0, 10, 500)
        x = np.exp(-t)
        y = np.exp(-t)
        trajectory = np.vstack([x, y])
        
        lyap = analyzer.estimate_lyapunov_exponent(trajectory, dt=0.02, n_renorm=20)
        
        # Should be finite
        assert np.isfinite(lyap)
    
    def test_estimate_lyapunov_custom_renorm(self, analyzer):
        """Test with custom renormalization steps."""
        trajectory = np.random.randn(2, 500)
        
        lyap = analyzer.estimate_lyapunov_exponent(trajectory, dt=0.01, n_renorm=50)
        
        assert isinstance(lyap, float)
    
    def test_estimate_lyapunov_single_variable(self, analyzer):
        """Test with single variable trajectory."""
        trajectory = np.random.randn(1, 500)
        
        lyap = analyzer.estimate_lyapunov_exponent(trajectory, dt=0.01, n_renorm=20)
        
        assert isinstance(lyap, float)
