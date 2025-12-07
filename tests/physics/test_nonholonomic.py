"""
Tests for Non-Holonomic Constraints Module
"""
import pytest
import numpy as np
import sympy as sp

from mechanics_dsl.domains.classical import (
    NonholonomicSystem,
    NonholonomicConstraint,
    ConstraintType,
    AppellEquations,
    MaggiEquations,
    rolling_constraint,
    knife_edge_constraint
)


class TestNonholonomicConstraint:
    """Test NonholonomicConstraint class."""
    
    def test_create_constraint(self):
        """Test creating a non-holonomic constraint."""
        constraint = NonholonomicConstraint(
            coefficients={'x': sp.S.One, 'theta': sp.Symbol('R')},
            inhomogeneous=sp.S.Zero,
            name="rolling"
        )
        
        assert constraint.name == "rolling"
        assert 'x' in constraint.coefficients
        assert 'theta' in constraint.coefficients
    
    def test_constraint_matrix_form(self):
        """Test converting to matrix form."""
        R = sp.Symbol('R', positive=True)
        constraint = NonholonomicConstraint(
            coefficients={'x': sp.S.One, 'theta': -R}
        )
        
        A, b = constraint.as_matrix_form(['x', 'theta'])
        
        assert A.shape == (1, 2)
        assert A[0, 0] == 1
        assert A[0, 1] == -R


class TestNonholonomicSystem:
    """Test NonholonomicSystem class."""
    
    def test_create_system(self):
        """Test creating a non-holonomic system."""
        system = NonholonomicSystem()
        assert system._lagrangian is None
        assert len(system._constraints) == 0
    
    def test_add_rolling_constraint(self):
        """Test adding rolling constraint."""
        system = NonholonomicSystem()
        
        R = sp.Symbol('R', positive=True)
        lam = system.add_rolling_constraint('x', 'theta', R)
        
        assert len(system._constraints) == 1
        assert isinstance(lam, sp.Symbol)
    
    def test_add_knife_edge_constraint(self):
        """Test adding knife-edge constraint."""
        system = NonholonomicSystem()
        
        lam = system.add_knife_edge_constraint('x', 'y', 'theta')
        
        assert len(system._constraints) == 1
        constraint = system._constraints[0]
        assert 'x' in constraint.coefficients
        assert 'y' in constraint.coefficients
    
    def test_set_lagrangian(self):
        """Test setting Lagrangian."""
        system = NonholonomicSystem()
        
        m = sp.Symbol('m', positive=True)
        x_dot = sp.Symbol('x_dot', real=True)
        
        L = sp.Rational(1, 2) * m * x_dot**2
        system.set_lagrangian(L)
        
        assert system._lagrangian == L
    
    def test_get_constraint_matrix(self):
        """Test getting constraint matrix."""
        system = NonholonomicSystem()
        
        R = sp.Symbol('R', positive=True)
        system.add_rolling_constraint('x', 'theta', R)
        
        A, b = system.get_constraint_matrix(['x', 'theta'])
        
        assert A.shape == (1, 2)
        assert b.shape == (1, 1)


class TestAppellEquations:
    """Test Appell's equations."""
    
    def test_compute_acceleration_energy(self):
        """Test computing Gibbs-Appell function."""
        appell = AppellEquations()
        
        m = sp.Symbol('m', positive=True)
        x_dot = appell.get_symbol('x_dot')
        
        T = sp.Rational(1, 2) * m * x_dot**2
        
        S = appell.compute_acceleration_energy(T, ['x'])
        
        x_ddot = appell.get_symbol('x_ddot')
        # S should contain x_ddot^2
        assert S.has(x_ddot)


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_rolling_constraint(self):
        """Test rolling_constraint function."""
        R = sp.Symbol('R', positive=True)
        
        constraint = rolling_constraint('x', 'theta', R)
        
        assert isinstance(constraint, NonholonomicConstraint)
        assert constraint.coefficients['x'] == 1
        assert constraint.coefficients['theta'] == -R
    
    def test_knife_edge_constraint(self):
        """Test knife_edge_constraint function."""
        constraint = knife_edge_constraint('x', 'y', 'theta')
        
        assert isinstance(constraint, NonholonomicConstraint)
        assert 'x' in constraint.coefficients
        assert 'y' in constraint.coefficients


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
