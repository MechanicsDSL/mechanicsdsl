"""
Comprehensive tests for symbolic.py to achieve 90%+ coverage

Tests all SymbolicEngine methods including:
- ast_to_sympy and _ast_to_sympy_impl for all expression types
- derive_equations_of_motion
- derive_equations_with_constraints
- derive_hamiltonian_equations
- lagrangian_to_hamiltonian
- solve_for_accelerations
"""

import pytest
import sympy as sp

from mechanics_dsl.parser import (
    BinaryOpExpr,
    DerivativeExpr,
    DerivativeVarExpr,
    Expression,
    FractionExpr,
    FunctionCallExpr,
    GreekLetterExpr,
    IdentExpr,
    NumberExpr,
    UnaryOpExpr,
    VectorExpr,
    VectorOpExpr,
)
from mechanics_dsl.symbolic import SymbolicEngine
from mechanics_dsl.utils import config


@pytest.fixture
def engine():
    """Create a SymbolicEngine instance."""
    return SymbolicEngine()


@pytest.fixture
def engine_no_cache():
    """Create SymbolicEngine with caching disabled."""
    original = config.cache_symbolic_results
    config.cache_symbolic_results = False
    eng = SymbolicEngine()
    config.cache_symbolic_results = original
    return eng


# ============================================================================
# AST TO SYMPY CONVERSION TESTS
# ============================================================================


class TestAstToSympyNumberExpr:
    """Tests for NumberExpr conversion."""

    def test_integer(self, engine):
        expr = NumberExpr(42)
        result = engine.ast_to_sympy(expr)
        assert result == sp.Float(42)

    def test_float(self, engine):
        expr = NumberExpr(3.14159)
        result = engine.ast_to_sympy(expr)
        assert abs(float(result) - 3.14159) < 1e-10

    def test_negative(self, engine):
        expr = NumberExpr(-5.5)
        result = engine.ast_to_sympy(expr)
        assert result == sp.Float(-5.5)

    def test_zero(self, engine):
        expr = NumberExpr(0)
        result = engine.ast_to_sympy(expr)
        assert result == sp.Float(0)


class TestAstToSympyIdentExpr:
    """Tests for IdentExpr conversion."""

    def test_simple_variable(self, engine):
        expr = IdentExpr("x")
        result = engine.ast_to_sympy(expr)
        assert isinstance(result, sp.Symbol)
        assert str(result) == "x"

    def test_time_variable(self, engine):
        expr = IdentExpr("t")
        result = engine.ast_to_sympy(expr)
        assert result == engine.time_symbol

    def test_subscripted(self, engine):
        expr = IdentExpr("x1")
        result = engine.ast_to_sympy(expr)
        assert str(result) == "x1"


class TestAstToSympyGreekLetterExpr:
    """Tests for GreekLetterExpr conversion."""

    def test_theta(self, engine):
        expr = GreekLetterExpr("theta")
        result = engine.ast_to_sympy(expr)
        assert isinstance(result, sp.Symbol)
        assert str(result) == "theta"

    def test_phi(self, engine):
        expr = GreekLetterExpr("phi")
        result = engine.ast_to_sympy(expr)
        assert str(result) == "phi"

    def test_omega(self, engine):
        expr = GreekLetterExpr("omega")
        result = engine.ast_to_sympy(expr)
        assert str(result) == "omega"


class TestAstToSympyBinaryOpExpr:
    """Tests for BinaryOpExpr conversion."""

    def test_addition(self, engine):
        left = NumberExpr(2)
        right = NumberExpr(3)
        expr = BinaryOpExpr(left, "+", right)
        result = engine.ast_to_sympy(expr)
        assert result == sp.Float(5)

    def test_subtraction(self, engine):
        left = NumberExpr(5)
        right = NumberExpr(3)
        expr = BinaryOpExpr(left, "-", right)
        result = engine.ast_to_sympy(expr)
        assert result == sp.Float(2)

    def test_multiplication(self, engine):
        left = NumberExpr(4)
        right = NumberExpr(3)
        expr = BinaryOpExpr(left, "*", right)
        result = engine.ast_to_sympy(expr)
        assert result == sp.Float(12)

    def test_division(self, engine):
        left = NumberExpr(10)
        right = NumberExpr(2)
        expr = BinaryOpExpr(left, "/", right)
        result = engine.ast_to_sympy(expr)
        assert result == sp.Float(5)

    def test_power(self, engine):
        left = NumberExpr(2)
        right = NumberExpr(3)
        expr = BinaryOpExpr(left, "^", right)
        result = engine.ast_to_sympy(expr)
        assert result == sp.Float(8)

    def test_unknown_operator(self, engine):
        left = NumberExpr(2)
        right = NumberExpr(3)
        expr = BinaryOpExpr(left, "%", right)
        with pytest.raises(ValueError, match="Unknown operator"):
            engine.ast_to_sympy(expr)


class TestAstToSympyUnaryOpExpr:
    """Tests for UnaryOpExpr conversion."""

    def test_negation(self, engine):
        operand = NumberExpr(5)
        expr = UnaryOpExpr("-", operand)
        result = engine.ast_to_sympy(expr)
        assert result == sp.Float(-5)

    def test_positive(self, engine):
        operand = NumberExpr(5)
        expr = UnaryOpExpr("+", operand)
        result = engine.ast_to_sympy(expr)
        assert result == sp.Float(5)

    def test_unknown_unary_operator(self, engine):
        operand = NumberExpr(5)
        expr = UnaryOpExpr("!", operand)
        with pytest.raises(ValueError, match="Unknown unary operator"):
            engine.ast_to_sympy(expr)


class TestAstToSympyFractionExpr:
    """Tests for FractionExpr conversion."""

    def test_simple_fraction(self, engine):
        num = NumberExpr(1)
        denom = NumberExpr(2)
        expr = FractionExpr(num, denom)
        result = engine.ast_to_sympy(expr)
        assert result == sp.Float(0.5)

    def test_symbolic_fraction(self, engine):
        num = IdentExpr("a")
        denom = IdentExpr("b")
        expr = FractionExpr(num, denom)
        result = engine.ast_to_sympy(expr)
        a = engine.get_symbol("a")
        b = engine.get_symbol("b")
        assert result == a / b


class TestAstToSympyDerivativeVarExpr:
    """Tests for DerivativeVarExpr conversion."""

    def test_first_derivative(self, engine):
        expr = DerivativeVarExpr("x", 1)
        result = engine.ast_to_sympy(expr)
        assert str(result) == "x_dot"

    def test_second_derivative(self, engine):
        expr = DerivativeVarExpr("theta", 2)
        result = engine.ast_to_sympy(expr)
        assert str(result) == "theta_ddot"

    def test_unsupported_order(self, engine):
        expr = DerivativeVarExpr("x", 3)
        with pytest.raises(ValueError, match="Derivative order 3 not supported"):
            engine.ast_to_sympy(expr)


class TestAstToSympyDerivativeExpr:
    """Tests for DerivativeExpr conversion."""

    def test_partial_derivative(self, engine):
        inner = IdentExpr("f")
        expr = DerivativeExpr(inner, "x", 1, partial=True)
        result = engine.ast_to_sympy(expr)
        # Should be symbolic derivative
        assert result is not None

    def test_time_derivative(self, engine):
        inner = IdentExpr("q")
        expr = DerivativeExpr(inner, "t", 1, partial=False)
        result = engine.ast_to_sympy(expr)
        assert result is not None

    def test_second_order_derivative(self, engine):
        inner = IdentExpr("y")
        expr = DerivativeExpr(inner, "x", 2, partial=True)
        result = engine.ast_to_sympy(expr)
        assert result is not None


class TestAstToSympyFunctionCallExpr:
    """Tests for FunctionCallExpr conversion."""

    def test_sin(self, engine):
        arg = IdentExpr("x")
        expr = FunctionCallExpr("sin", [arg])
        result = engine.ast_to_sympy(expr)
        x = engine.get_symbol("x")
        assert result == sp.sin(x)

    def test_cos(self, engine):
        arg = IdentExpr("x")
        expr = FunctionCallExpr("cos", [arg])
        result = engine.ast_to_sympy(expr)
        x = engine.get_symbol("x")
        assert result == sp.cos(x)

    def test_tan(self, engine):
        arg = IdentExpr("x")
        expr = FunctionCallExpr("tan", [arg])
        result = engine.ast_to_sympy(expr)
        x = engine.get_symbol("x")
        assert result == sp.tan(x)

    def test_exp(self, engine):
        arg = IdentExpr("x")
        expr = FunctionCallExpr("exp", [arg])
        result = engine.ast_to_sympy(expr)
        x = engine.get_symbol("x")
        assert result == sp.exp(x)

    def test_log(self, engine):
        arg = IdentExpr("x")
        expr = FunctionCallExpr("log", [arg])
        result = engine.ast_to_sympy(expr)
        x = engine.get_symbol("x")
        assert result == sp.log(x)

    def test_ln(self, engine):
        arg = IdentExpr("x")
        expr = FunctionCallExpr("ln", [arg])
        result = engine.ast_to_sympy(expr)
        x = engine.get_symbol("x")
        assert result == sp.log(x)

    def test_sqrt(self, engine):
        arg = IdentExpr("x")
        expr = FunctionCallExpr("sqrt", [arg])
        result = engine.ast_to_sympy(expr)
        x = engine.get_symbol("x")
        assert result == sp.sqrt(x)

    def test_sinh(self, engine):
        arg = IdentExpr("x")
        expr = FunctionCallExpr("sinh", [arg])
        result = engine.ast_to_sympy(expr)
        x = engine.get_symbol("x")
        assert result == sp.sinh(x)

    def test_cosh(self, engine):
        arg = IdentExpr("x")
        expr = FunctionCallExpr("cosh", [arg])
        result = engine.ast_to_sympy(expr)
        x = engine.get_symbol("x")
        assert result == sp.cosh(x)

    def test_tanh(self, engine):
        arg = IdentExpr("x")
        expr = FunctionCallExpr("tanh", [arg])
        result = engine.ast_to_sympy(expr)
        x = engine.get_symbol("x")
        assert result == sp.tanh(x)

    def test_arcsin(self, engine):
        arg = IdentExpr("x")
        expr = FunctionCallExpr("arcsin", [arg])
        result = engine.ast_to_sympy(expr)
        x = engine.get_symbol("x")
        assert result == sp.asin(x)

    def test_arccos(self, engine):
        arg = IdentExpr("x")
        expr = FunctionCallExpr("arccos", [arg])
        result = engine.ast_to_sympy(expr)
        x = engine.get_symbol("x")
        assert result == sp.acos(x)

    def test_arctan(self, engine):
        arg = IdentExpr("x")
        expr = FunctionCallExpr("arctan", [arg])
        result = engine.ast_to_sympy(expr)
        x = engine.get_symbol("x")
        assert result == sp.atan(x)

    def test_abs(self, engine):
        arg = IdentExpr("x")
        expr = FunctionCallExpr("abs", [arg])
        result = engine.ast_to_sympy(expr)
        x = engine.get_symbol("x")
        assert result == sp.Abs(x)

    def test_custom_function(self, engine):
        arg = IdentExpr("x")
        expr = FunctionCallExpr("custom_func", [arg])
        result = engine.ast_to_sympy(expr)
        assert result is not None


class TestAstToSympyVectorExpr:
    """Tests for VectorExpr conversion."""

    def test_vector_creation(self, engine):
        components = [NumberExpr(1), NumberExpr(2), NumberExpr(3)]
        expr = VectorExpr(components)
        result = engine.ast_to_sympy(expr)
        assert isinstance(result, sp.Matrix)
        assert result.shape == (3, 1)


class TestAstToSympyVectorOpExpr:
    """Tests for VectorOpExpr conversion."""

    def test_gradient(self, engine):
        inner = IdentExpr("f")
        expr = VectorOpExpr("grad", inner, None)
        result = engine.ast_to_sympy(expr)
        assert isinstance(result, sp.Matrix)

    def test_gradient_no_operand(self, engine):
        expr = VectorOpExpr("grad", None, None)
        result = engine.ast_to_sympy(expr)
        assert str(result) == "nabla"

    def test_dot_product_vectors(self, engine):
        v1 = VectorExpr([NumberExpr(1), NumberExpr(0), NumberExpr(0)])
        v2 = VectorExpr([NumberExpr(0), NumberExpr(1), NumberExpr(0)])
        expr = VectorOpExpr("dot", v1, v2)
        result = engine.ast_to_sympy(expr)
        assert result == 0

    def test_dot_product_scalars(self, engine):
        s1 = NumberExpr(3)
        s2 = NumberExpr(4)
        expr = VectorOpExpr("dot", s1, s2)
        result = engine.ast_to_sympy(expr)
        assert result == sp.Float(12)

    def test_cross_product(self, engine):
        v1 = VectorExpr([NumberExpr(1), NumberExpr(0), NumberExpr(0)])
        v2 = VectorExpr([NumberExpr(0), NumberExpr(1), NumberExpr(0)])
        expr = VectorOpExpr("cross", v1, v2)
        result = engine.ast_to_sympy(expr)
        assert isinstance(result, sp.Matrix)

    def test_cross_product_non_vector(self, engine):
        s1 = NumberExpr(3)
        s2 = NumberExpr(4)
        expr = VectorOpExpr("cross", s1, s2)
        with pytest.raises(ValueError, match="Cross product requires vector"):
            engine.ast_to_sympy(expr)

    def test_magnitude_vector(self, engine):
        v = VectorExpr([NumberExpr(3), NumberExpr(4), NumberExpr(0)])
        expr = VectorOpExpr("magnitude", v, None)
        result = engine.ast_to_sympy(expr)
        assert result == sp.Float(5)

    def test_magnitude_scalar(self, engine):
        s = NumberExpr(-5)
        expr = VectorOpExpr("magnitude", s, None)
        result = engine.ast_to_sympy(expr)
        assert result == sp.Float(5)


class TestAstToSympyUnknownExpr:
    """Tests for unknown expression types."""

    def test_unknown_expression_type(self, engine):
        class UnknownExpr(Expression):
            pass

        expr = UnknownExpr()
        with pytest.raises(ValueError, match="Cannot convert"):
            engine.ast_to_sympy(expr)


class TestAstToSympyCaching:
    """Tests for AST to SymPy caching behavior."""

    def test_cache_hit(self, engine):
        expr = NumberExpr(42)
        result1 = engine.ast_to_sympy(expr)
        result2 = engine.ast_to_sympy(expr)
        assert result1 == result2

    def test_cache_disabled(self, engine_no_cache):
        expr = NumberExpr(42)
        result = engine_no_cache.ast_to_sympy(expr)
        assert result == sp.Float(42)


# ============================================================================
# DERIVE EQUATIONS WITH CONSTRAINTS
# ============================================================================


class TestDeriveEquationsWithConstraints:
    """Tests for derive_equations_with_constraints method."""

    def test_simple_constraint(self, engine):
        t = sp.Symbol("t")
        x = sp.Function("x")(t)
        x_dot = sp.diff(x, t)
        m = sp.Symbol("m", positive=True)

        # Simple Lagrangian
        L = sp.Rational(1, 2) * m * x_dot**2

        # Constraint: x = 0
        constraint = engine.get_symbol("x")

        equations, coords = engine.derive_equations_with_constraints(L, ["x"], [constraint])

        assert len(equations) > 0
        assert len(coords) > len(["x"])  # Should include lambda

    def test_multiple_constraints(self, engine):
        m = engine.get_symbol("m")
        x_dot = engine.get_symbol("x_dot")
        y_dot = engine.get_symbol("y_dot")

        L = sp.Rational(1, 2) * m * (x_dot**2 + y_dot**2)

        c1 = engine.get_symbol("x")
        c2 = engine.get_symbol("y")

        equations, coords = engine.derive_equations_with_constraints(L, ["x", "y"], [c1, c2])

        assert "lambda_0" in coords
        assert "lambda_1" in coords


# ============================================================================
# DERIVE HAMILTONIAN EQUATIONS
# ============================================================================


class TestDeriveHamiltonianEquations:
    """Tests for derive_hamiltonian_equations method."""

    def test_oscillator_hamiltonian(self, engine):
        q = engine.get_symbol("q")
        p_q = engine.get_symbol("p_q")
        m = engine.get_symbol("m")
        k = engine.get_symbol("k")

        H = p_q**2 / (2 * m) + sp.Rational(1, 2) * k * q**2

        q_dots, p_dots = engine.derive_hamiltonian_equations(H, ["q"])

        assert len(q_dots) == 1
        assert len(p_dots) == 1
        # dq/dt = p/m
        assert q_dots[0] == p_q / m
        # dp/dt = -kq
        assert p_dots[0] == -k * q

    def test_multiple_coordinates(self, engine):
        engine.get_symbol("q1")
        engine.get_symbol("q2")
        p_q1 = engine.get_symbol("p_q1")
        p_q2 = engine.get_symbol("p_q2")
        m = engine.get_symbol("m")

        H = (p_q1**2 + p_q2**2) / (2 * m)

        q_dots, p_dots = engine.derive_hamiltonian_equations(H, ["q1", "q2"])

        assert len(q_dots) == 2
        assert len(p_dots) == 2


# ============================================================================
# LAGRANGIAN TO HAMILTONIAN
# ============================================================================


class TestLagrangianToHamiltonian:
    """Tests for lagrangian_to_hamiltonian method."""

    def test_simple_oscillator(self, engine):
        m = engine.get_symbol("m")
        k = engine.get_symbol("k")
        q = engine.get_symbol("q")
        q_dot = engine.get_symbol("q_dot")

        L = sp.Rational(1, 2) * m * q_dot**2 - sp.Rational(1, 2) * k * q**2

        H = engine.lagrangian_to_hamiltonian(L, ["q"])

        assert H is not None

    def test_pendulum(self, engine):
        m = engine.get_symbol("m")
        l = engine.get_symbol("l")
        g = engine.get_symbol("g")
        theta = engine.get_symbol("theta")
        theta_dot = engine.get_symbol("theta_dot")

        L = sp.Rational(1, 2) * m * l**2 * theta_dot**2 + m * g * l * sp.cos(theta)

        H = engine.lagrangian_to_hamiltonian(L, ["theta"])

        assert H is not None


# ============================================================================
# SOLVE FOR ACCELERATIONS
# ============================================================================


class TestSolveForAccelerations:
    """Tests for solve_for_accelerations method."""

    def test_simple_oscillator(self, engine):
        m = engine.get_symbol("m")
        k = engine.get_symbol("k")
        x = engine.get_symbol("x")
        x_ddot = engine.get_symbol("x_ddot")

        # m*x_ddot + k*x = 0
        equation = m * x_ddot + k * x

        accelerations = engine.solve_for_accelerations([equation], ["x"])

        assert "x_ddot" in accelerations
        assert accelerations["x_ddot"] == -k * x / m

    def test_multiple_coordinates(self, engine):
        m = engine.get_symbol("m")
        engine.get_symbol("x")
        engine.get_symbol("y")
        x_ddot = engine.get_symbol("x_ddot")
        y_ddot = engine.get_symbol("y_ddot")

        eq1 = m * x_ddot
        eq2 = m * y_ddot

        accelerations = engine.solve_for_accelerations([eq1, eq2], ["x", "y"])

        assert "x_ddot" in accelerations
        assert "y_ddot" in accelerations

    def test_acceleration_already_present(self, engine):
        m = engine.get_symbol("m")
        k = engine.get_symbol("k")
        x = engine.get_symbol("x")
        x_ddot = engine.get_symbol("x_ddot")

        equation = m * x_ddot + k * x

        accelerations = engine.solve_for_accelerations([equation], ["x"])

        assert "x_ddot" in accelerations

    def test_zero_coefficient(self, engine):
        x = engine.get_symbol("x")

        # Equation without acceleration term
        equation = x + 5

        accelerations = engine.solve_for_accelerations([equation], ["x"])

        assert "x_ddot" in accelerations


# ============================================================================
# ENGINE INITIALIZATION TESTS
# ============================================================================


class TestEngineInitialization:
    """Tests for SymbolicEngine initialization."""

    def test_cache_enabled(self):
        original = config.cache_symbolic_results
        config.cache_symbolic_results = True
        engine = SymbolicEngine()
        assert engine._cache is not None
        config.cache_symbolic_results = original

    def test_cache_disabled(self):
        original = config.cache_symbolic_results
        config.cache_symbolic_results = False
        engine = SymbolicEngine()
        assert engine._cache is None
        config.cache_symbolic_results = original

    def test_perf_monitor_enabled(self):
        original = config.enable_performance_monitoring
        config.enable_performance_monitoring = True
        engine = SymbolicEngine()
        assert engine._perf_monitor is not None
        config.enable_performance_monitoring = original

    def test_perf_monitor_disabled(self):
        original = config.enable_performance_monitoring
        config.enable_performance_monitoring = False
        engine = SymbolicEngine()
        assert engine._perf_monitor is None
        config.enable_performance_monitoring = original


# ============================================================================
# TIMEOUT HANDLING TESTS
# ============================================================================


class TestTimeoutHandling:
    """Tests for timeout handling in symbolic operations."""

    def test_simplification_timeout_zero(self, engine):
        original = config.simplification_timeout
        config.simplification_timeout = 0

        m = engine.get_symbol("m")
        engine.get_symbol("theta")
        theta_dot = engine.get_symbol("theta_dot")

        L = sp.Rational(1, 2) * m * theta_dot**2

        # Should work without timeout
        equations = engine.derive_equations_of_motion(L, ["theta"])
        assert len(equations) > 0

        config.simplification_timeout = original


# ============================================================================
# EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Edge case tests for symbolic engine."""

    def test_empty_coordinates(self, engine):
        L = sp.Symbol("L")
        equations = engine.derive_equations_of_motion(L, [])
        assert equations == []

    def test_symbol_with_assumptions(self, engine):
        x = engine.get_symbol("x", positive=True)
        assert x.is_positive

    def test_function_caching(self, engine):
        f1 = engine.get_function("f")
        f2 = engine.get_function("f")
        assert f1 is f2
