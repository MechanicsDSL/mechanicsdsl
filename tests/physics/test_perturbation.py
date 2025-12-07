"""
Tests for Perturbation Theory Module
"""
import pytest
import numpy as np
import sympy as sp

from mechanics_dsl.domains.classical import (
    PerturbationExpander,
    PerturbationResult,
    PerturbationType,
    AveragingMethod,
    MultiScaleAnalysis,
    perturbation_expand,
    average_over_angle
)


class TestPerturbationExpander:
    """Test perturbation expansion."""
    
    def test_create_expander(self):
        """Test creating perturbation expander."""
        expander = PerturbationExpander()
        assert expander is not None
    
    def test_first_order_expansion(self):
        """Test first-order perturbation expansion."""
        expander = PerturbationExpander()
        
        p = sp.Symbol('p', real=True)
        x = sp.Symbol('x', real=True)
        m = sp.Symbol('m', positive=True)
        omega = sp.Symbol('omega', positive=True)
        alpha = sp.Symbol('alpha', real=True)
        
        # Harmonic oscillator
        H0 = p**2/(2*m) + m*omega**2*x**2/2
        # Anharmonic perturbation
        H1 = alpha * x**3
        
        result = expander.expand_hamiltonian(H0, H1, ['x'], order=1)
        
        assert isinstance(result, PerturbationResult)
        assert result.order == 1
        assert result.unperturbed == H0
        assert len(result.corrections) == 1
    
    def test_second_order_expansion(self):
        """Test second-order perturbation."""
        expander = PerturbationExpander()
        
        x = sp.Symbol('x', real=True)
        H0 = x**2 / 2
        H1 = x**3
        
        result = expander.expand_hamiltonian(H0, H1, ['x'], order=2)
        
        assert result.order == 2
        assert len(result.corrections) == 2


class TestAveragingMethod:
    """Test method of averaging."""
    
    def test_average_over_fast_angle(self):
        """Test averaging over fast angle."""
        expander = PerturbationExpander()
        
        theta = sp.Symbol('theta', real=True)
        
        # sin(theta) averages to zero
        expr = sp.sin(theta)
        avg = expander.average_over_fast_angle(expr, 'theta')
        
        assert abs(float(avg.evalf())) < 1e-10
    
    def test_average_constant(self):
        """Test that constant passes through."""
        expander = PerturbationExpander()
        
        A = sp.Symbol('A', real=True)
        avg = expander.average_over_fast_angle(A, 'theta')
        
        assert avg == A
    
    def test_average_cos_squared(self):
        """Test average of cos²(θ) = 1/2."""
        expander = PerturbationExpander()
        
        theta = sp.Symbol('theta', real=True)
        expr = sp.cos(theta)**2
        
        avg = expander.average_over_fast_angle(expr, 'theta')
        
        # <cos²(θ)> = 1/2
        assert abs(float(avg.evalf()) - 0.5) < 1e-10


class TestMultiScaleAnalysis:
    """Test multiple time scale analysis."""
    
    def test_define_time_scales(self):
        """Test defining multiple time scales."""
        msa = MultiScaleAnalysis(order=3)
        
        t = sp.Symbol('t', real=True)
        epsilon = sp.Symbol('epsilon', positive=True)
        
        scales = msa.define_time_scales(t, epsilon)
        
        # Should have T_0, T_1, T_2, T_3
        assert len(scales) == 4
    
    def test_lindstedt_poincare(self):
        """Test Lindstedt-Poincaré method setup."""
        expander = PerturbationExpander()
        
        x = sp.Symbol('x', real=True)
        omega0 = sp.Symbol('omega_0', positive=True)
        epsilon = sp.Symbol('epsilon', positive=True)
        
        result = expander.lindstedt_poincare(
            x**2, 'x', omega0, epsilon, order=2
        )
        
        assert 'frequency' in result
        assert 'frequency_corrections' in result
        assert len(result['frequency_corrections']) == 2


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_perturbation_expand(self):
        """Test perturbation_expand function."""
        x = sp.Symbol('x', real=True)
        H0 = x**2
        H1 = x**3
        
        result = perturbation_expand(H0, H1, ['x'], order=1)
        
        assert isinstance(result, PerturbationResult)
    
    def test_average_over_angle(self):
        """Test average_over_angle function."""
        theta = sp.Symbol('theta', real=True)
        
        avg = average_over_angle(sp.sin(theta), 'theta')
        
        assert abs(float(avg.evalf())) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
