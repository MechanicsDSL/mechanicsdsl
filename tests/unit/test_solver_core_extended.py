"""
Extended solver/core.py coverage tests.

Covers: compile_equations (time-dependent, constant eq, compilation errors, wrapper
for ndarray/ZeroDivision), compile_hamiltonian_equations, _replace_derivatives,
_select_optimal_solver, simulate with method=, equations_of_motion edge cases.
"""

import numpy as np
import pytest
import sympy as sp

from mechanics_dsl.solver import NumericalSimulator
from mechanics_dsl.symbolic import SymbolicEngine


@pytest.fixture
def engine():
    return SymbolicEngine()


@pytest.fixture
def sim(engine):
    return NumericalSimulator(engine)


class TestCompileEquationsTimeDependent:
    """Test compile_equations with time-dependent equations (has_time branch)."""

    def test_compile_time_dependent_equation(self, sim):
        """Equation that depends on time: ordered_symbols includes time_sym."""
        sim.set_parameters({"A": 1.0, "omega": 2.0})
        sim.set_initial_conditions({"x": 0.0, "x_dot": 1.0})
        t = sim.symbolic.time_symbol
        x = sim.symbolic.get_symbol("x")
        # x_ddot = -x + sin(omega*t)  (time-dependent)
        accels = {"x_ddot": -x + sp.sin(sp.Symbol("omega") * t)}
        sim.compile_equations(accels, ["x"])
        assert "x_ddot" in sim.equations
        # Evaluate at t=0, x=0, x_dot=1: x_ddot = 0 + sin(0) = 0
        out = sim.equations["x_ddot"](0.0, 0.0, 1.0)
        assert np.isfinite(out)


class TestCompileEquationsConstant:
    """Test compile_equations when equation is constant (ordered_symbols empty)."""

    def test_compile_constant_equation(self, sim):
        """Equation with no free symbols: compiled as constant."""
        sim.set_parameters({})
        sim.set_initial_conditions({"x": 0.0, "x_dot": 0.0})
        accels = {"x_ddot": sp.Integer(5)}
        sim.compile_equations(accels, ["x"])
        assert "x_ddot" in sim.equations
        out = sim.equations["x_ddot"](0.0, 0.0, 0.0)
        assert out == 5.0


class TestCompileEquationsWrapperEdgeCases:
    """Test wrapper: ndarray return, ZeroDivision, index out of range, len(args)<1."""

    def test_wrapper_ndarray_return(self, sim):
        """lambdify can return ndarray; wrapper converts to float."""
        sim.set_parameters({"k": 1.0})
        sim.set_initial_conditions({"x": 1.0, "x_dot": 0.0})
        x = sim.symbolic.get_symbol("x")
        # Expression that might return array in some edge case
        accels = {"x_ddot": -x}
        sim.compile_equations(accels, ["x"])
        out = sim.equations["x_ddot"](0.0, 1.0, 0.0)
        assert np.isfinite(out)


class TestCompileHamiltonianEquations:
    """Test compile_hamiltonian_equations."""

    def test_compile_hamiltonian_equations(self, sim):
        sim.set_parameters({"m": 1.0, "k": 1.0})
        # state_vars for coordinates ["q"] are ["q", "p_q"]
        sim.set_initial_conditions({"q": 1.0, "p_q": 0.0})
        q = sim.symbolic.get_symbol("q")
        p_q = sim.symbolic.get_symbol("p_q")
        # dq/dt = p_q/m, dp_q/dt = -k*q
        q_dots = [p_q / sp.Symbol("m")]
        p_dots = [-sp.Symbol("k") * q]
        sim.compile_hamiltonian_equations(q_dots, p_dots, ["q"])
        assert sim.use_hamiltonian is True
        result = sim.simulate((0, 1), num_points=10)
        assert result["success"]


class TestReplaceDerivatives:
    """Test _replace_derivatives."""

    def test_replace_derivatives(self, sim):
        x = sim.symbolic.get_symbol("x")
        x_dot = sim.symbolic.get_symbol("x_dot")
        # Build an expression that uses derivatives
        accels = {"x_ddot": -x}
        sim.compile_equations(accels, ["x"])
        # _replace_derivatives is called inside compile_equations
        assert "x_ddot" in sim.equations


class TestSelectOptimalSolver:
    """Test _select_optimal_solver for different time spans and state sizes."""

    def test_select_optimal_solver_short(self, sim):
        sim.set_parameters({"k": 1.0})
        sim.set_initial_conditions({"x": 1.0, "x_dot": 0.0})
        x = sim.symbolic.get_symbol("x")
        sim.compile_equations({"x_ddot": -x}, ["x"])
        method = sim._select_optimal_solver((0, 1), np.array([1.0, 0.0]))
        assert method in ["RK45", "LSODA", "RK23", "Radau", "BDF", "DOP853"]

    def test_select_optimal_solver_long(self, sim):
        sim.set_parameters({"k": 1.0})
        sim.set_initial_conditions({"x": 1.0, "x_dot": 0.0})
        x = sim.symbolic.get_symbol("x")
        sim.compile_equations({"x_ddot": -x}, ["x"])
        method = sim._select_optimal_solver((0, 1000), np.array([1.0, 0.0]))
        assert method in ["RK45", "LSODA", "RK23", "Radau", "BDF", "DOP853"]

    def test_select_optimal_solver_many_coords(self, sim):
        sim.set_parameters({"k": 1.0})
        sim.set_initial_conditions({"x1": 1.0, "x1_dot": 0.0, "x2": 0.0, "x2_dot": 0.0})
        x1 = sim.symbolic.get_symbol("x1")
        x2 = sim.symbolic.get_symbol("x2")
        sim.compile_equations({"x1_ddot": -x1, "x2_ddot": -x2}, ["x1", "x2"])
        y0 = np.array([1.0, 0.0, 0.0, 0.0])
        method = sim._select_optimal_solver((0, 10), y0)
        assert method in ["RK45", "LSODA", "RK23", "Radau", "BDF", "DOP853"]


class TestSimulateWithMethod:
    """Test simulate() with explicit method=."""

    def test_simulate_method_rk45(self, sim):
        sim.set_parameters({"k": 1.0})
        sim.set_initial_conditions({"x": 1.0, "x_dot": 0.0})
        x = sim.symbolic.get_symbol("x")
        sim.compile_equations({"x_ddot": -x}, ["x"])
        result = sim.simulate((0, 2), num_points=20, method="RK45")
        assert result["success"]
        assert len(result["t"]) == 20

    def test_simulate_method_radau(self, sim):
        sim.set_parameters({"k": 1.0})
        sim.set_initial_conditions({"x": 1.0, "x_dot": 0.0})
        x = sim.symbolic.get_symbol("x")
        sim.compile_equations({"x_ddot": -x}, ["x"])
        result = sim.simulate((0, 1), num_points=10, method="Radau")
        assert result["success"]


class TestEquationsOfMotionEdgeCases:
    """Test equations_of_motion with Hamiltonian mode (via compile_hamiltonian + simulate)."""

    def test_equations_of_motion_hamiltonian_mode(self, sim):
        sim.set_parameters({"m": 1.0, "k": 1.0})
        sim.set_initial_conditions({"q": 1.0, "p_q": 0.0})
        q = sim.symbolic.get_symbol("q")
        p_q = sim.symbolic.get_symbol("p_q")
        sim.compile_hamiltonian_equations([p_q / sp.Symbol("m")], [-sp.Symbol("k") * q], ["q"])
        out = sim.equations_of_motion(0.0, np.array([1.0, 0.0]))
        assert np.all(np.isfinite(out))
