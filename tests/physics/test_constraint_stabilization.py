"""
Tests for Constraint Stabilization Module

Validates:
- Baumgarte stabilization
- Velocity-level constraints
- ConstrainedLagrangianSystem
"""
import pytest
import numpy as np
import sympy as sp

from mechanics_dsl.domains.classical import (
    ConstraintHandler,
    BaumgarteStabilization,
    ConstrainedLagrangianSystem
)


class TestBaumgarteStabilization:
    """Test Baumgarte stabilization implementation."""
    
    def test_default_parameters(self):
        """Test default alpha and beta values."""
        stab = BaumgarteStabilization()
        
        assert stab.alpha == 5.0
        assert stab.beta == 5.0
    
    def test_custom_parameters(self):
        """Test custom stabilization parameters."""
        stab = BaumgarteStabilization(alpha=10.0, beta=8.0)
        
        assert stab.alpha == 10.0
        assert stab.beta == 8.0
    
    def test_velocity_level_constraint(self):
        """Test differentiation to velocity level."""
        stab = BaumgarteStabilization()
        
        x = stab.get_symbol('x')
        y = stab.get_symbol('y')
        l = sp.Symbol('l', positive=True)
        
        # Position constraint: x^2 + y^2 - l^2 = 0
        g = x**2 + y**2 - l**2
        
        g_dot = stab.velocity_level_constraint(g, ['x', 'y'])
        
        # g_dot = 2*x*x_dot + 2*y*y_dot
        x_dot = stab.get_symbol('x_dot')
        y_dot = stab.get_symbol('y_dot')
        
        expected = 2*x*x_dot + 2*y*y_dot
        assert sp.simplify(g_dot - expected) == 0
    
    def test_acceleration_level_constraint(self):
        """Test differentiation to acceleration level."""
        stab = BaumgarteStabilization()
        
        x = stab.get_symbol('x')
        
        # Simple constraint: x - a = 0 (constant position)
        # g_dot = x_dot
        # g_ddot = x_ddot
        g = x
        
        g_ddot = stab.acceleration_level_constraint(g, ['x'])
        
        x_ddot = stab.get_symbol('x_ddot')
        
        # g_ddot should contain x_ddot
        assert x_ddot in g_ddot.free_symbols
    
    def test_stabilized_constraint(self):
        """Test Baumgarte stabilized form."""
        stab = BaumgarteStabilization(alpha=5.0, beta=5.0)
        
        x = stab.get_symbol('x')
        
        # Constraint: x = 0
        g = x
        
        stabilized = stab.stabilized_constraint(g, ['x'])
        
        # Should be: x_ddot + 2*5*x_dot + 25*x
        x_dot = stab.get_symbol('x_dot')
        x_ddot = stab.get_symbol('x_ddot')
        
        # Verify terms are present
        assert x in stabilized.free_symbols
        assert x_dot in stabilized.free_symbols
        assert x_ddot in stabilized.free_symbols


class TestConstrainedLagrangianSystem:
    """Test ConstrainedLagrangianSystem class."""
    
    def test_system_creation(self):
        """Test creating a constrained system."""
        system = ConstrainedLagrangianSystem()
        
        assert system.constraint_handler is not None
        assert system.stabilization is not None
    
    def test_set_lagrangian(self):
        """Test setting the Lagrangian."""
        system = ConstrainedLagrangianSystem()
        
        m = sp.Symbol('m', positive=True)
        x = sp.Symbol('x', real=True)
        x_dot = sp.Symbol('x_dot', real=True)
        
        L = sp.Rational(1, 2) * m * x_dot**2
        system.set_lagrangian(L)
        
        assert system._lagrangian == L
    
    def test_add_constraint(self):
        """Test adding a holonomic constraint."""
        system = ConstrainedLagrangianSystem()
        
        x = sp.Symbol('x', real=True)
        y = sp.Symbol('y', real=True)
        
        # Add pendulum constraint: x^2 + y^2 = l^2
        lam = system.add_constraint(x**2 + y**2 - 1)
        
        # Should return Lagrange multiplier
        assert isinstance(lam, sp.Symbol)
    
    def test_derive_equations(self):
        """Test deriving constrained equations of motion."""
        system = ConstrainedLagrangianSystem()
        
        m = sp.Symbol('m', positive=True)
        x = system.get_symbol('x')
        y = system.get_symbol('y')
        x_dot = system.get_symbol('x_dot')
        y_dot = system.get_symbol('y_dot')
        g = sp.Symbol('g', positive=True)
        
        # 2D pendulum
        L = sp.Rational(1, 2) * m * (x_dot**2 + y_dot**2) - m * g * y
        system.set_lagrangian(L)
        
        # Constraint: x^2 + y^2 = 1
        system.add_constraint(x**2 + y**2 - 1)
        
        eom = system.derive_equations_of_motion(['x', 'y'])
        
        # Should have equations for x and y accelerations
        assert 'x_ddot_eq' in eom
        assert 'y_ddot_eq' in eom
        
        # Should have stabilized constraint
        assert 'constraint_0' in eom


class TestConstraintHandler:
    """Test basic ConstraintHandler functionality."""
    
    def test_add_holonomic(self):
        """Test adding holonomic constraint."""
        handler = ConstraintHandler()
        
        x = sp.Symbol('x', real=True)
        
        lam = handler.add_holonomic_constraint(x**2 - 1)
        
        assert len(handler.holonomic_constraints) == 1
        assert isinstance(lam, sp.Symbol)
    
    def test_augment_lagrangian(self):
        """Test augmenting Lagrangian with constraint terms."""
        handler = ConstraintHandler()
        
        m = sp.Symbol('m', positive=True)
        x = sp.Symbol('x', real=True)
        x_dot = sp.Symbol('x_dot', real=True)
        
        L = sp.Rational(1, 2) * m * x_dot**2
        
        # Add constraint
        lam = handler.add_holonomic_constraint(x - 1)
        
        L_aug = handler.augment_lagrangian(L)
        
        # Augmented Lagrangian should include lambda * constraint
        assert lam in L_aug.free_symbols
    
    def test_clear(self):
        """Test clearing constraints."""
        handler = ConstraintHandler()
        
        x = sp.Symbol('x', real=True)
        handler.add_holonomic_constraint(x)
        handler.add_nonholonomic_constraint(x)
        
        handler.clear()
        
        assert len(handler.holonomic_constraints) == 0
        assert len(handler.nonholonomic_constraints) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
