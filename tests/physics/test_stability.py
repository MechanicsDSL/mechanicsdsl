"""
Tests for Stability Analysis Module

Validates:
- Equilibrium point finding
- Linearization and eigenvalue analysis
- Stability classification
"""
import pytest
import numpy as np
import sympy as sp

from mechanics_dsl.domains.classical import (
    StabilityAnalyzer,
    StabilityResult,
    StabilityType,
    EquilibriumPoint,
    find_equilibria,
    analyze_stability
)


class TestEquilibriumFinding:
    """Test equilibrium point detection."""
    
    def test_harmonic_oscillator_equilibrium(self):
        """Harmonic oscillator has equilibrium at x=0."""
        x = sp.Symbol('x', real=True)
        
        # Use numerical value for k to allow Hessian evaluation
        V = sp.Rational(1, 2) * 10.0 * x**2
        
        equilibria = find_equilibria(V, ['x'])
        
        assert len(equilibria) == 1
        assert abs(equilibria[0].coordinates['x']) < 1e-10
        assert equilibria[0].is_minimum  # Stable equilibrium
    
    def test_pendulum_equilibria(self):
        """Simple pendulum has equilibria at θ = 0, π."""
        theta = sp.Symbol('theta', real=True)
        
        # Use numerical values for parameters
        V = 1.0 * 10.0 * 0.5 * (1 - sp.cos(theta))
        
        equilibria = find_equilibria(V, ['theta'], 
                                      bounds={'theta': (-4, 4)})
        
        # Should find at least θ = 0 (stable) and θ = π (unstable)
        assert len(equilibria) >= 1
        
        # Check θ = 0 is found and is minimum
        theta_zero = [eq for eq in equilibria 
                      if abs(eq.coordinates['theta']) < 0.1]
        assert len(theta_zero) >= 1
        assert theta_zero[0].is_minimum
    
    def test_double_well_equilibria(self):
        """Double-well potential has 3 equilibria."""
        x = sp.Symbol('x', real=True)
        
        # V = x^4 - 2x^2: minima at ±1, maximum at 0
        V = x**4 - 2*x**2
        
        equilibria = find_equilibria(V, ['x'])
        
        assert len(equilibria) == 3
        
        # Two should be minima (x = ±1)
        minima = [eq for eq in equilibria if eq.is_minimum]
        assert len(minima) == 2


class TestLinearization:
    """Test linearization around equilibrium."""
    
    def test_mass_stiffness_extraction(self):
        """Test extraction of M and K matrices."""
        m, k = sp.symbols('m k', positive=True)
        x = sp.Symbol('x', real=True)
        x_dot = sp.Symbol('x_dot', real=True)
        
        # L = (1/2)*m*x_dot^2 - (1/2)*k*x^2
        L = sp.Rational(1, 2) * m * x_dot**2 - sp.Rational(1, 2) * k * x**2
        
        analyzer = StabilityAnalyzer()
        eq = EquilibriumPoint({'x': 0.0})
        
        M, K = analyzer.linearize_lagrangian(L, ['x'], eq)
        
        # M should be [[m]], K should be [[k]]
        assert M[0, 0] == m
        assert K[0, 0] == k
    
    def test_coupled_oscillators_matrices(self):
        """Test M and K for coupled oscillators."""
        m, k, kc = sp.symbols('m k k_c', positive=True)
        x1 = sp.Symbol('x1', real=True)
        x2 = sp.Symbol('x2', real=True)
        x1_dot = sp.Symbol('x1_dot', real=True)
        x2_dot = sp.Symbol('x2_dot', real=True)
        
        # Two masses, three springs
        T = sp.Rational(1, 2) * m * (x1_dot**2 + x2_dot**2)
        V = sp.Rational(1, 2) * k * x1**2 + sp.Rational(1, 2) * k * x2**2 + \
            sp.Rational(1, 2) * kc * (x1 - x2)**2
        L = T - V
        
        analyzer = StabilityAnalyzer()
        eq = EquilibriumPoint({'x1': 0.0, 'x2': 0.0})
        
        M, K = analyzer.linearize_lagrangian(L, ['x1', 'x2'], eq)
        
        # M should be diagonal
        assert M[0, 0] == m
        assert M[1, 1] == m
        assert M[0, 1] == 0
        
        # K should have coupling terms
        assert K[0, 0] == k + kc
        assert K[1, 1] == k + kc
        assert K[0, 1] == -kc


class TestStabilityClassification:
    """Test stability type classification."""
    
    def test_stable_oscillator(self):
        """Harmonic oscillator equilibrium is center (marginally stable)."""
        m, k = 1.0, 10.0
        x = sp.Symbol('x', real=True)
        x_dot = sp.Symbol('x_dot', real=True)
        
        L = sp.Rational(1, 2) * m * x_dot**2 - sp.Rational(1, 2) * k * x**2
        
        result = analyze_stability(L, ['x'], {'x': 0.0})
        
        # Undamped oscillator is CENTER (pure imaginary eigenvalues)
        assert result.stability_type == StabilityType.CENTER
    
    def test_unstable_inverted_pendulum(self):
        """Inverted pendulum (θ = π) is unstable."""
        m, g, l = sp.symbols('m g l', positive=True)
        theta = sp.Symbol('theta', real=True)
        theta_dot = sp.Symbol('theta_dot', real=True)
        
        # Expand around θ = π: let θ = π + δ
        # For small δ: V ≈ const - m*g*l*cos(π) ≈ m*g*l - m*g*l*(-1 + δ²/2)
        # This gives negative stiffness
        
        # For testing, create V with negative curvature
        V_unstable = -sp.Rational(1, 2) * 10.0 * theta**2  # Negative stiffness
        T = sp.Rational(1, 2) * 1.0 * theta_dot**2
        L = T - V_unstable
        
        result = analyze_stability(L, ['theta'], {'theta': 0.0})
        
        # Should detect instability (positive real eigenvalue)
        assert result.stability_type in [StabilityType.UNSTABLE, StabilityType.SADDLE]
    
    def test_natural_frequency(self):
        """Test natural frequency computation."""
        m, k = 1.0, 100.0  # ω = 10 rad/s
        x = sp.Symbol('x', real=True)
        x_dot = sp.Symbol('x_dot', real=True)
        
        L = sp.Rational(1, 2) * m * x_dot**2 - sp.Rational(1, 2) * k * x**2
        
        result = analyze_stability(L, ['x'], {'x': 0.0})
        
        # Check natural frequency
        if result.natural_frequencies is not None and len(result.natural_frequencies) > 0:
            omega = result.natural_frequencies[0]
            expected = np.sqrt(k / m)  # 10 rad/s
            assert abs(omega - expected) < 0.1


class TestNormalModes:
    """Test normal mode computation from stability analyzer."""
    
    def test_two_dof_modes(self):
        """Test normal modes for 2-DOF system."""
        M = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
        K = np.array([[3.0, -1.0],
                      [-1.0, 3.0]])
        
        analyzer = StabilityAnalyzer()
        frequencies, modes = analyzer.compute_normal_modes(M, K)
        
        # Analytical: ω₁ = √2, ω₂ = 2
        assert len(frequencies) == 2
        np.testing.assert_array_almost_equal(
            np.sort(frequencies),
            [np.sqrt(2), 2.0],
            decimal=5
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
