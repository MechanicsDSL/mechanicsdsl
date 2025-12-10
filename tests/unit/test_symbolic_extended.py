"""
Extended unit tests for MechanicsDSL symbolic module.

Tests the SymbolicEngine class for symbolic computation functionality.
"""

import pytest
import numpy as np
import sympy as sp

from mechanics_dsl.symbolic import SymbolicEngine


class TestSymbolicEngineInit:
    """Tests for SymbolicEngine initialization."""
    
    def test_init_creates_instance(self):
        """Test that SymbolicEngine can be instantiated."""
        engine = SymbolicEngine()
        assert engine is not None
    
    def test_init_has_sympy(self):
        """Test that engine has sympy reference."""
        engine = SymbolicEngine()
        assert engine.sp is sp
    
    def test_init_has_symbol_map(self):
        """Test that engine has symbol_map."""
        engine = SymbolicEngine()
        assert hasattr(engine, 'symbol_map')
        assert isinstance(engine.symbol_map, dict)
    
    def test_init_has_function_map(self):
        """Test that engine has function_map."""
        engine = SymbolicEngine()
        assert hasattr(engine, 'function_map')


class TestGetSymbol:
    """Tests for SymbolicEngine.get_symbol method."""
    
    def test_get_symbol_creates_symbol(self):
        """Test that get_symbol creates a symbol."""
        engine = SymbolicEngine()
        x = engine.get_symbol('x')
        
        assert isinstance(x, sp.Symbol)
        assert str(x) == 'x'
    
    def test_get_symbol_cached(self):
        """Test that same symbol is returned for same name."""
        engine = SymbolicEngine()
        x1 = engine.get_symbol('x')
        x2 = engine.get_symbol('x')
        
        assert x1 is x2
    
    def test_get_symbol_different_names(self):
        """Test different names create different symbols."""
        engine = SymbolicEngine()
        x = engine.get_symbol('x')
        y = engine.get_symbol('y')
        
        assert x != y
    
    def test_get_symbol_with_assumptions(self):
        """Test symbol creation with assumptions."""
        engine = SymbolicEngine()
        t = engine.get_symbol('t', real=True, positive=True)
        
        assert isinstance(t, sp.Symbol)


class TestGetFunction:
    """Tests for SymbolicEngine.get_function method."""
    
    def test_get_function_creates_function(self):
        """Test that get_function creates a function."""
        engine = SymbolicEngine()
        f = engine.get_function('f')
        
        assert callable(f)
    
    def test_get_function_cached(self):
        """Test that same function is returned for same name."""
        engine = SymbolicEngine()
        f1 = engine.get_function('f')
        f2 = engine.get_function('f')
        
        assert f1 is f2


class TestDeriveEquationsOfMotion:
    """Tests for SymbolicEngine.derive_equations_of_motion method."""
    
    @pytest.fixture
    def engine(self):
        return SymbolicEngine()
    
    def test_simple_oscillator(self, engine):
        """Test equations of motion for harmonic oscillator."""
        t = sp.Symbol('t')
        x = sp.Function('x')(t)
        x_dot = sp.diff(x, t)
        
        m, k = sp.symbols('m k', positive=True)
        
        # L = T - V = 1/2 m x_dot^2 - 1/2 k x^2
        L = sp.Rational(1, 2) * m * x_dot**2 - sp.Rational(1, 2) * k * x**2
        
        equations = engine.derive_equations_of_motion(L, ['x'])
        
        assert isinstance(equations, list)
        assert len(equations) >= 1
    
    def test_simple_pendulum(self, engine):
        """Test equations of motion for simple pendulum."""
        t = sp.Symbol('t')
        theta = sp.Function('theta')(t)
        theta_dot = sp.diff(theta, t)
        
        m, l, g = sp.symbols('m l g', positive=True)
        
        # L = 1/2 m l^2 theta_dot^2 + m g l cos(theta)
        L = sp.Rational(1, 2) * m * l**2 * theta_dot**2 + m * g * l * sp.cos(theta)
        
        equations = engine.derive_equations_of_motion(L, ['theta'])
        
        assert isinstance(equations, list)


class TestDeriveHamiltonianEquations:
    """Tests for SymbolicEngine.derive_hamiltonian_equations method."""
    
    @pytest.fixture
    def engine(self):
        return SymbolicEngine()
    
    def test_simple_hamiltonian(self, engine):
        """Test Hamilton's equations for simple system."""
        q, p = sp.symbols('q p')
        m, k = sp.symbols('m k', positive=True)
        
        # H = p^2/(2m) + 1/2 k q^2
        H = p**2 / (2*m) + sp.Rational(1, 2) * k * q**2
        
        q_dots, p_dots = engine.derive_hamiltonian_equations(H, ['q'])
        
        assert isinstance(q_dots, list)
        assert isinstance(p_dots, list)
    
    def test_multiple_coordinates(self, engine):
        """Test Hamilton's equations with multiple coordinates."""
        q1, q2, p1, p2 = sp.symbols('q1 q2 p1 p2')
        
        H = p1**2 + p2**2 + q1**2 + q2**2
        
        q_dots, p_dots = engine.derive_hamiltonian_equations(H, ['q1', 'q2'])
        
        assert len(q_dots) == 2
        assert len(p_dots) == 2


class TestLagrangianToHamiltonian:
    """Tests for SymbolicEngine.lagrangian_to_hamiltonian method."""
    
    @pytest.fixture
    def engine(self):
        return SymbolicEngine()
    
    def test_oscillator_lagrangian_to_hamiltonian(self, engine):
        """Test Legendre transform for oscillator."""
        t = sp.Symbol('t')
        x = sp.Function('x')(t)
        x_dot = sp.diff(x, t)
        
        m, k = sp.symbols('m k', positive=True)
        
        L = sp.Rational(1, 2) * m * x_dot**2 - sp.Rational(1, 2) * k * x**2
        
        H = engine.lagrangian_to_hamiltonian(L, ['x'])
        
        # Hamiltonian should be a SymPy expression
        assert isinstance(H, sp.Expr)


class TestSolveForAccelerations:
    """Tests for SymbolicEngine.solve_for_accelerations method."""
    
    @pytest.fixture
    def engine(self):
        return SymbolicEngine()
    
    def test_simple_system(self, engine):
        """Test solving for accelerations in simple system."""
        x, x_dot, x_ddot = sp.symbols('x x_dot x_ddot')
        k, m = sp.symbols('k m', positive=True)
        
        # Simple equation: m*x_ddot = -k*x
        equation = m * x_ddot + k * x
        
        result = engine.solve_for_accelerations([equation], ['x'])
        
        assert isinstance(result, dict)
    
    def test_coupled_system(self, engine):
        """Test solving for accelerations in coupled system."""
        x1, x2 = sp.symbols('x1 x2')
        x1_ddot, x2_ddot = sp.symbols('x1_ddot x2_ddot')
        k, m = sp.symbols('k m', positive=True)
        
        # Coupled oscillators
        eq1 = m * x1_ddot + k * (x1 - x2)
        eq2 = m * x2_ddot + k * (x2 - x1)
        
        result = engine.solve_for_accelerations([eq1, eq2], ['x1', 'x2'])
        
        assert isinstance(result, dict)
