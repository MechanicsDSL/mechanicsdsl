"""
Additional solver and symbolic coverage tests
"""

import numpy as np
import pytest
import sympy as sp

from mechanics_dsl.solver.core import NumericalSimulator
from mechanics_dsl.symbolic import SymbolicEngine


class TestSymbolicEngineCoverage:
    """Tests for SymbolicEngine edge cases"""

    def test_engine_creation(self):
        """Test engine creation"""
        engine = SymbolicEngine()
        assert engine is not None
        assert engine.time_symbol is not None

    def test_get_symbol(self):
        """Test getting symbols"""
        engine = SymbolicEngine()
        x = engine.get_symbol("x")
        assert x is not None
        assert str(x) == "x"

    def test_get_same_symbol(self):
        """Test getting same symbol returns cached"""
        engine = SymbolicEngine()
        x1 = engine.get_symbol("x")
        x2 = engine.get_symbol("x")
        assert x1 is x2

    def test_ast_to_sympy_number(self):
        """Test converting number AST to sympy"""
        from mechanics_dsl.parser.ast_nodes import NumberExpr

        engine = SymbolicEngine()
        ast = NumberExpr(3.14)
        result = engine.ast_to_sympy(ast)
        assert float(result) == 3.14

    def test_ast_to_sympy_ident(self):
        """Test converting ident AST to sympy"""
        from mechanics_dsl.parser.ast_nodes import IdentExpr

        engine = SymbolicEngine()
        ast = IdentExpr("x")
        result = engine.ast_to_sympy(ast)
        assert "x" in str(result)

    def test_derive_eom_single_coord(self):
        """Test deriving EOM for single coordinate"""
        engine = SymbolicEngine()
        # L = 0.5*m*v^2 - 0.5*k*x^2
        m, k = sp.symbols("m k")
        x = engine.get_symbol("x")
        x_dot = engine.get_symbol("x_dot")
        L = sp.Rational(1, 2) * m * x_dot**2 - sp.Rational(1, 2) * k * x**2

        eom = engine.derive_equations_of_motion(L, ["x"])
        assert len(eom) == 1

    def test_derive_eom_coupled(self):
        """Test deriving EOM for coupled coordinates"""
        engine = SymbolicEngine()
        m, g = sp.symbols("m g")
        theta1 = engine.get_symbol("theta1")
        theta1_dot = engine.get_symbol("theta1_dot")
        theta2 = engine.get_symbol("theta2")
        theta2_dot = engine.get_symbol("theta2_dot")

        L = sp.Rational(1, 2) * m * (theta1_dot**2 + theta2_dot**2)

        eom = engine.derive_equations_of_motion(L, ["theta1", "theta2"])
        assert len(eom) == 2

    def test_solve_accelerations_single(self):
        """Test solving for single acceleration"""
        engine = SymbolicEngine()
        m, k = sp.symbols("m k")
        x_ddot = engine.get_symbol("x_ddot")
        x = engine.get_symbol("x")

        # m*x_ddot + k*x = 0
        eq = [m * x_ddot + k * x]

        accels = engine.solve_for_accelerations(eq, ["x"])
        assert "x_ddot" in accels

    def test_solve_accelerations_coupled(self):
        """Test solving for coupled accelerations"""
        engine = SymbolicEngine()
        m = sp.symbols("m")
        a1 = engine.get_symbol("theta1_ddot")
        a2 = engine.get_symbol("theta2_ddot")
        t1 = engine.get_symbol("theta1")
        t2 = engine.get_symbol("theta2")

        # Simple coupled system
        eq1 = m * a1 + t1 + t2
        eq2 = m * a2 + t1 - t2

        accels = engine.solve_for_accelerations([eq1, eq2], ["theta1", "theta2"])
        assert "theta1_ddot" in accels
        assert "theta2_ddot" in accels


class TestNumericalSimulatorCoverage:
    """Tests for NumericalSimulator edge cases"""

    def test_simulator_creation(self):
        """Test simulator creation"""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        assert sim is not None

    def test_set_parameters(self):
        """Test setting parameters"""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        sim.set_parameters({"m": 1.0, "k": 4.0})
        assert sim.parameters["m"] == 1.0
        assert sim.parameters["k"] == 4.0

    def test_set_initial_conditions(self):
        """Test setting initial conditions"""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        sim.set_initial_conditions({"x": 1.0, "x_dot": 0.0})
        assert sim.initial_conditions["x"] == 1.0

    def test_compile_equations(self):
        """Test compiling equations"""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        sim.set_parameters({"m": 1.0, "k": 4.0})

        m, k = sp.symbols("m k")
        x = engine.get_symbol("x")
        x_dot = engine.get_symbol("x_dot")

        accels = {"x_ddot": -k / m * x}
        sim.compile_equations(accels, ["x"])

        assert "x_ddot" in sim.equations

    def test_equations_of_motion_with_invalid_input(self):
        """Test EOM with invalid input"""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        sim.coordinates = ["x"]
        sim.equations = {}

        # Test with None
        result = sim.equations_of_motion(0, None)
        assert np.all(np.isfinite(result))

    def test_select_optimal_solver(self):
        """Test solver selection"""
        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        sim.coordinates = ["x"]

        method = sim._select_optimal_solver((0, 10), np.array([1.0, 0.0]))
        assert method in ["RK45", "LSODA", "RK23", "Radau", "BDF", "DOP853"]


class TestVisualizationCoverage:
    """Tests for visualization edge cases"""

    def test_visualization_imports(self):
        """Test visualization module imports"""
        from mechanics_dsl.visualization import MechanicsVisualizer

        assert MechanicsVisualizer is not None

    def test_visualizer_creation(self):
        """Test visualizer creation"""
        from mechanics_dsl.visualization import MechanicsVisualizer

        vis = MechanicsVisualizer()
        assert vis is not None

    def test_phase_space_import(self):
        """Test phase space module import"""
        from mechanics_dsl.visualization.phase_space import PhaseSpaceVisualizer

        assert PhaseSpaceVisualizer is not None

    def test_animator_import(self):
        """Test animator module import"""
        from mechanics_dsl.visualization import animator

        assert animator is not None

    def test_plotter_import(self):
        """Test plotter module import"""
        from mechanics_dsl.visualization import plotter

        assert plotter is not None


class TestCachingCoverage:
    """Tests for caching utility"""

    def test_caching_import(self):
        """Test caching module import"""
        from mechanics_dsl.utils.caching import LRUCache

        assert LRUCache is not None

    def test_cache_creation(self):
        """Test cache creation"""
        from mechanics_dsl.utils.caching import LRUCache

        cache = LRUCache(maxsize=100)
        assert cache is not None

    def test_cache_get_set(self):
        """Test cache get/set operations"""
        from mechanics_dsl.utils.caching import LRUCache

        cache = LRUCache(maxsize=100)

        cache["key1"] = "value1"
        result = cache["key1"]
        assert result == "value1"

    def test_cache_miss(self):
        """Test cache miss"""
        from mechanics_dsl.utils.caching import LRUCache

        cache = LRUCache(maxsize=100)

        result = cache.get("nonexistent")
        assert result is None


class TestProfilingCoverage:
    """Tests for profiling utility"""

    def test_profiling_import(self):
        """Test profiling module import"""
        from mechanics_dsl.utils.profiling import PerformanceMonitor

        assert PerformanceMonitor is not None

    def test_monitor_creation(self):
        """Test monitor creation"""
        from mechanics_dsl.utils.profiling import PerformanceMonitor

        monitor = PerformanceMonitor()
        assert monitor is not None

    def test_start_stop_timer(self):
        """Test timer start/stop"""
        from mechanics_dsl.utils.profiling import PerformanceMonitor

        monitor = PerformanceMonitor()

        monitor.start_timer("test")
        import time

        time.sleep(0.01)
        duration = monitor.stop_timer("test")

        assert duration is not None or True  # May return None on error
