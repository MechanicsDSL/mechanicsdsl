"""
Extended unit tests for the symbolic module.

Tests the SymbolicEngine class with more coverage for edge cases
and different symbolic computation scenarios.
"""

import numpy as np
import pytest
import sympy as sp

from mechanics_dsl.symbolic import SymbolicEngine


@pytest.fixture
def engine():
    """Create a SymbolicEngine instance."""
    return SymbolicEngine()


class TestSymbolicEngineInit:
    """Tests for SymbolicEngine initialization."""

    def test_init(self):
        engine = SymbolicEngine()
        assert engine is not None

    def test_init_has_sympy(self):
        engine = SymbolicEngine()
        assert engine.sp is sp

    def test_init_has_symbol_map(self):
        engine = SymbolicEngine()
        assert hasattr(engine, "symbol_map")
        assert isinstance(engine.symbol_map, dict)

    def test_init_has_function_map(self):
        engine = SymbolicEngine()
        assert hasattr(engine, "function_map")


class TestGetSymbol:
    """Tests for get_symbol method."""

    def test_creates_symbol(self, engine):
        x = engine.get_symbol("x")
        assert isinstance(x, sp.Symbol)
        assert str(x) == "x"

    def test_cached(self, engine):
        x1 = engine.get_symbol("x")
        x2 = engine.get_symbol("x")
        assert x1 is x2

    def test_different_names(self, engine):
        x = engine.get_symbol("x")
        y = engine.get_symbol("y")
        assert x != y

    def test_greek_letters(self, engine):
        theta = engine.get_symbol("theta")
        assert isinstance(theta, sp.Symbol)

    def test_subscripted(self, engine):
        x1 = engine.get_symbol("x1")
        x2 = engine.get_symbol("x2")
        assert x1 != x2


class TestGetFunction:
    """Tests for get_function method."""

    def test_creates_function(self, engine):
        f = engine.get_function("f")
        assert callable(f)

    def test_cached(self, engine):
        f1 = engine.get_function("f")
        f2 = engine.get_function("f")
        assert f1 is f2

    def test_different_functions(self, engine):
        f = engine.get_function("f")
        g = engine.get_function("g")
        assert f != g


class TestDeriveEquationsOfMotion:
    """Tests for derive_equations_of_motion method."""

    def test_oscillator(self, engine):
        t = sp.Symbol("t")
        x = sp.Function("x")(t)
        x_dot = sp.diff(x, t)
        m, k = sp.symbols("m k", positive=True)

        L = sp.Rational(1, 2) * m * x_dot**2 - sp.Rational(1, 2) * k * x**2

        equations = engine.derive_equations_of_motion(L, ["x"])

        assert isinstance(equations, list)
        assert len(equations) >= 1

    def test_pendulum(self, engine):
        t = sp.Symbol("t")
        theta = sp.Function("theta")(t)
        theta_dot = sp.diff(theta, t)
        m, l, g = sp.symbols("m l g", positive=True)

        L = sp.Rational(1, 2) * m * l**2 * theta_dot**2 + m * g * l * sp.cos(theta)

        equations = engine.derive_equations_of_motion(L, ["theta"])

        assert isinstance(equations, list)


class TestSymbolicOperations:
    """Tests for symbolic mathematical operations."""

    def test_differentiation(self, engine):
        x = engine.get_symbol("x")
        expr = x**2
        deriv = sp.diff(expr, x)
        assert deriv == 2 * x

    def test_trig_differentiation(self, engine):
        theta = engine.get_symbol("theta")
        expr = sp.sin(theta)
        deriv = sp.diff(expr, theta)
        assert deriv == sp.cos(theta)

    def test_product_rule(self, engine):
        x = engine.get_symbol("x")
        expr = x * sp.sin(x)
        deriv = sp.diff(expr, x)
        expected = sp.sin(x) + x * sp.cos(x)
        assert sp.simplify(deriv - expected) == 0

    def test_integration(self, engine):
        x = engine.get_symbol("x")
        expr = x**2
        integral = sp.integrate(expr, x)
        assert integral == x**3 / 3

    def test_trig_integration(self, engine):
        x = engine.get_symbol("x")
        expr = sp.cos(x)
        integral = sp.integrate(expr, x)
        assert integral == sp.sin(x)

    def test_simplification(self, engine):
        x = engine.get_symbol("x")
        expr = x**2 - x**2
        result = sp.simplify(expr)
        assert result == 0

    def test_expand(self, engine):
        x = engine.get_symbol("x")
        expr = (x + 1) ** 2
        result = sp.expand(expr)
        assert result == x**2 + 2 * x + 1

    def test_factor(self, engine):
        x = engine.get_symbol("x")
        expr = x**2 + 2 * x + 1
        result = sp.factor(expr)
        assert result == (x + 1) ** 2


class TestLagrangianMechanics:
    """Tests for Lagrangian mechanics calculations."""

    def test_kinetic_energy_1d(self, engine):
        m = engine.get_symbol("m")
        v = engine.get_symbol("v")
        T = sp.Rational(1, 2) * m * v**2

        dT_dv = sp.diff(T, v)
        assert dT_dv == m * v

    def test_potential_energy_spring(self, engine):
        k = engine.get_symbol("k")
        x = engine.get_symbol("x")
        V = sp.Rational(1, 2) * k * x**2

        dV_dx = sp.diff(V, x)
        assert dV_dx == k * x

    def test_potential_energy_gravity(self, engine):
        m = engine.get_symbol("m")
        g = engine.get_symbol("g")
        h = engine.get_symbol("h")
        V = m * g * h

        dV_dh = sp.diff(V, h)
        assert dV_dh == m * g


class TestHamiltonianMechanics:
    """Tests for Hamiltonian mechanics calculations."""

    def test_hamilton_equations_simple(self, engine):
        q, p = sp.symbols("q p")
        m, k = sp.symbols("m k", positive=True)

        H = p**2 / (2 * m) + sp.Rational(1, 2) * k * q**2

        # dH/dp = q_dot
        q_dot = sp.diff(H, p)
        assert q_dot == p / m

        # -dH/dq = p_dot
        p_dot = -sp.diff(H, q)
        assert p_dot == -k * q

    def test_legendre_transform(self, engine):
        m = engine.get_symbol("m")
        k = engine.get_symbol("k")
        q = engine.get_symbol("q")
        q_dot = engine.get_symbol("q_dot")

        L = sp.Rational(1, 2) * m * q_dot**2 - sp.Rational(1, 2) * k * q**2

        # p = dL/d(q_dot)
        p = sp.diff(L, q_dot)
        assert p == m * q_dot


class TestSymbolicSubstitution:
    """Tests for symbolic substitution."""

    def test_substitute_single(self, engine):
        x = engine.get_symbol("x")
        expr = x**2 + 2 * x + 1
        result = expr.subs(x, 2)
        assert result == 9

    def test_substitute_multiple(self, engine):
        x = engine.get_symbol("x")
        y = engine.get_symbol("y")
        expr = x**2 + y**2
        result = expr.subs([(x, 1), (y, 2)])
        assert result == 5

    def test_substitute_expression(self, engine):
        x = engine.get_symbol("x")
        y = engine.get_symbol("y")
        expr = x**2
        result = expr.subs(x, y + 1)
        assert sp.expand(result) == y**2 + 2 * y + 1


class TestSeriesExpansion:
    """Tests for series expansion."""

    def test_taylor_sin(self, engine):
        x = engine.get_symbol("x")
        expr = sp.sin(x)
        series = sp.series(expr, x, 0, n=5)
        # sin(x) ≈ x - x^3/6 + ...
        assert series.coeff(x, 1) == 1
        assert series.coeff(x, 3) == sp.Rational(-1, 6)

    def test_taylor_cos(self, engine):
        x = engine.get_symbol("x")
        expr = sp.cos(x)
        series = sp.series(expr, x, 0, n=5)
        # cos(x) ≈ 1 - x^2/2 + ...
        assert series.coeff(x, 0) == 1
        assert series.coeff(x, 2) == sp.Rational(-1, 2)

    def test_taylor_exp(self, engine):
        x = engine.get_symbol("x")
        expr = sp.exp(x)
        series = sp.series(expr, x, 0, n=5)
        # e^x ≈ 1 + x + x^2/2 + ...
        assert series.coeff(x, 0) == 1
        assert series.coeff(x, 1) == 1


class TestSymbolicSolving:
    """Tests for symbolic equation solving."""

    def test_solve_linear(self, engine):
        x = engine.get_symbol("x")
        eq = 2 * x - 6
        solutions = sp.solve(eq, x)
        assert solutions == [3]

    def test_solve_quadratic(self, engine):
        x = engine.get_symbol("x")
        eq = x**2 - 4
        solutions = sp.solve(eq, x)
        assert set(solutions) == {-2, 2}

    def test_solve_system(self, engine):
        x = engine.get_symbol("x")
        y = engine.get_symbol("y")
        eq1 = x + y - 3
        eq2 = x - y - 1
        solutions = sp.solve([eq1, eq2], [x, y])
        assert solutions[x] == 2
        assert solutions[y] == 1


class TestComplexExpressions:
    """Tests for complex symbolic expressions."""

    def test_nested_trig(self, engine):
        x = engine.get_symbol("x")
        expr = sp.sin(sp.cos(x))
        deriv = sp.diff(expr, x)
        expected = -sp.cos(sp.cos(x)) * sp.sin(x)
        assert sp.simplify(deriv - expected) == 0

    def test_exponential_product(self, engine):
        x = engine.get_symbol("x")
        expr = x * sp.exp(x)
        deriv = sp.diff(expr, x)
        expected = sp.exp(x) + x * sp.exp(x)
        assert sp.simplify(deriv - expected) == 0

    def test_logarithmic(self, engine):
        x = engine.get_symbol("x")
        expr = sp.log(x)
        deriv = sp.diff(expr, x)
        assert deriv == 1 / x
