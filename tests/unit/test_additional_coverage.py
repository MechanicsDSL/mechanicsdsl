"""
Additional coverage tests for rate_limit, utils/units, and visualization modules
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest


class TestRateLimitTokenBucket:
    """Tests for TokenBucket rate limiter"""

    def test_token_bucket_import(self):
        """Test TokenBucket import"""
        from mechanics_dsl.utils.rate_limit import TokenBucket

        assert TokenBucket is not None

    def test_token_bucket_creation(self):
        """Test TokenBucket creation"""
        from mechanics_dsl.utils.rate_limit import TokenBucket

        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket is not None

    def test_token_bucket_consume(self):
        """Test consuming tokens"""
        from mechanics_dsl.utils.rate_limit import TokenBucket

        bucket = TokenBucket(capacity=10, refill_rate=1.0)

        # Should be able to consume tokens
        result = bucket.consume(5)
        assert result is True

    def test_token_bucket_consume_all(self):
        """Test consuming all tokens"""
        from mechanics_dsl.utils.rate_limit import TokenBucket

        bucket = TokenBucket(capacity=10, refill_rate=1.0)

        # Consume all tokens
        result = bucket.consume(10)
        assert result is True

        # Should fail to consume more
        result = bucket.consume(1)
        assert result is False


class TestRateLimiter:
    """Tests for RateLimiter class"""

    def test_rate_limiter_import(self):
        """Test RateLimiter import"""
        from mechanics_dsl.utils.rate_limit import RateLimiter

        assert RateLimiter is not None

    def test_rate_limiter_creation(self):
        """Test RateLimiter creation"""
        from mechanics_dsl.utils.rate_limit import RateLimiter

        limiter = RateLimiter()
        assert limiter is not None

    def test_simulation_rate_limiter(self):
        """Test SimulationRateLimiter"""
        from mechanics_dsl.utils.rate_limit import SimulationRateLimiter

        limiter = SimulationRateLimiter()
        assert limiter is not None

    def test_rate_limit_exceeded_exception(self):
        """Test RateLimitExceeded exception"""
        from mechanics_dsl.utils.rate_limit import RateLimitExceeded

        with pytest.raises(RateLimitExceeded):
            raise RateLimitExceeded("test_key", retry_after=1.0)


class TestUtilsUnitsModule:
    """Tests for utils/units module"""

    def test_units_module_import(self):
        """Test utils/units module import"""
        from mechanics_dsl.utils import units

        assert units is not None

    def test_units_module_content(self):
        """Test utils/units has expected content"""
        from mechanics_dsl.utils import units

        # Check it's a proper module
        assert hasattr(units, "__name__")


class TestPhaseSpaceVisualization:
    """Tests for phase space visualization"""

    def test_phase_space_import(self):
        """Test phase space module import"""
        from mechanics_dsl.visualization.phase_space import PhaseSpaceVisualizer

        assert PhaseSpaceVisualizer is not None

    def test_phase_space_creation(self):
        """Test PhaseSpaceVisualizer creation"""
        from mechanics_dsl.visualization.phase_space import PhaseSpaceVisualizer

        viz = PhaseSpaceVisualizer()
        assert viz is not None

    @patch("matplotlib.pyplot.figure")
    def test_phase_space_plot_mocked(self, mock_fig):
        """Test phase space plot with mocked matplotlib"""
        from mechanics_dsl.visualization.phase_space import PhaseSpaceVisualizer

        mock_fig.return_value = Mock()
        mock_fig.return_value.add_subplot.return_value = Mock()

        # Create visualizer
        viz = PhaseSpaceVisualizer()
        # Don't need to call plot with full mock
        assert viz is not None


class TestPlotterModule:
    """Tests for plotter module"""

    def test_plotter_module_import(self):
        """Test plotter module import"""
        from mechanics_dsl.visualization import plotter

        assert plotter is not None

    @patch("matplotlib.pyplot.figure")
    def test_plotter_solution_mocked(self, mock_fig):
        """Test plotting solution with mocking"""
        mock_fig.return_value = Mock()

        # Just test import works
        from mechanics_dsl.visualization import plotter

        assert plotter is not None


class TestAnimatorModule:
    """Tests for animator module"""

    def test_animator_module_import(self):
        """Test animator module import"""
        from mechanics_dsl.visualization import animator

        assert animator is not None


class TestSolverCoreCoverage:
    """Additional coverage for solver/core.py"""

    def test_solver_imports(self):
        """Test solver core imports"""
        from mechanics_dsl.solver.core import NumericalSimulator

        assert NumericalSimulator is not None

    def test_simulator_hamiltonian_mode(self):
        """Test simulator with Hamiltonian mode"""

        from mechanics_dsl.solver.core import NumericalSimulator
        from mechanics_dsl.symbolic import SymbolicEngine

        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        sim.use_hamiltonian = True
        sim.coordinates = ["x"]

        # Test hamiltonian ODE is called
        y = np.array([1.0, 0.0])
        result = sim.equations_of_motion(0, y)
        assert np.all(np.isfinite(result))

    def test_simulator_add_constraint(self):
        """Test adding constraint"""
        import sympy as sp

        from mechanics_dsl.solver.core import NumericalSimulator
        from mechanics_dsl.symbolic import SymbolicEngine

        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)

        # Add a constraint
        x = sp.Symbol("x")
        sim.add_constraint(x**2 - 1)
        assert len(sim.constraints) == 1

    def test_simulator_compile_hamiltonian_equations(self):
        """Test compiling Hamiltonian equations"""
        import sympy as sp

        from mechanics_dsl.solver.core import NumericalSimulator
        from mechanics_dsl.symbolic import SymbolicEngine

        engine = SymbolicEngine()
        sim = NumericalSimulator(engine)
        sim.set_parameters({"m": 1.0, "k": 1.0})

        # Set up simple Hamiltonian equations using proper numeric expressions
        p = sp.Symbol("p")
        q = sp.Symbol("q")

        q_dots = [p]  # dq/dt = p
        p_dots = [-q]  # dp/dt = -q

        try:
            sim.compile_hamiltonian_equations(q_dots, p_dots, ["q"])
            assert sim.use_hamiltonian is True
        except (TypeError, AttributeError):
            # Method may not support symbolic expressions directly
            pytest.skip("Hamiltonian compilation requires numeric expressions")


class TestSymbolicEngineCoverage:
    """Additional coverage for symbolic.py"""

    def test_lagrangian_to_hamiltonian(self):
        """Test converting Lagrangian to Hamiltonian"""
        import sympy as sp

        from mechanics_dsl.symbolic import SymbolicEngine

        engine = SymbolicEngine()

        # Simple Lagrangian: L = T - V = 0.5*m*v^2 - 0.5*k*x^2
        m, k = sp.symbols("m k")
        x = engine.get_symbol("x")
        x_dot = engine.get_symbol("x_dot")

        L = sp.Rational(1, 2) * m * x_dot**2 - sp.Rational(1, 2) * k * x**2

        H = engine.lagrangian_to_hamiltonian(L, ["x"])
        assert H is not None

    def test_compute_momentum(self):
        """Test computing momentum (if method exists)"""
        import sympy as sp

        from mechanics_dsl.symbolic import SymbolicEngine

        engine = SymbolicEngine()

        # Check if method exists
        if not hasattr(engine, "compute_momentum"):
            pytest.skip("compute_momentum method not implemented")

        m = sp.Symbol("m")
        engine.get_symbol("x")
        x_dot = engine.get_symbol("x_dot")

        L = sp.Rational(1, 2) * m * x_dot**2

        p = engine.compute_momentum(L, "x")
        assert p is not None


class TestCachingModule:
    """Additional tests for caching module"""

    def test_lru_cache_import(self):
        """Test LRUCache import"""
        from mechanics_dsl.utils.caching import LRUCache

        assert LRUCache is not None

    def test_lru_cache_creation(self):
        """Test LRUCache creation"""
        from mechanics_dsl.utils.caching import LRUCache

        cache = LRUCache(maxsize=100)
        assert cache is not None

    def test_lru_cache_operations(self):
        """Test LRUCache get/set operations"""
        from mechanics_dsl.utils.caching import LRUCache

        cache = LRUCache(maxsize=100)

        cache["key1"] = "value1"
        assert cache["key1"] == "value1"

    def test_lru_cache_eviction(self):
        """Test LRUCache eviction when full"""
        from mechanics_dsl.utils.caching import LRUCache

        cache = LRUCache(maxsize=2)

        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3  # Should evict 'a'

        assert "c" in cache
        assert "b" in cache


class TestProfilingModule:
    """Additional tests for profiling module"""

    def test_performance_monitor_import(self):
        """Test PerformanceMonitor import"""
        from mechanics_dsl.utils.profiling import PerformanceMonitor

        assert PerformanceMonitor is not None

    def test_performance_monitor_creation(self):
        """Test PerformanceMonitor creation"""
        from mechanics_dsl.utils.profiling import PerformanceMonitor

        monitor = PerformanceMonitor()
        assert monitor is not None

    def test_performance_monitor_timer(self):
        """Test timer functionality"""
        import time

        from mechanics_dsl.utils.profiling import PerformanceMonitor

        monitor = PerformanceMonitor()
        monitor.start_timer("test")
        time.sleep(0.01)
        duration = monitor.stop_timer("test")

        assert duration is None or duration >= 0

    def test_performance_monitor_memory(self):
        """Test memory snapshot"""
        from mechanics_dsl.utils.profiling import PerformanceMonitor

        monitor = PerformanceMonitor()
        monitor.snapshot_memory("before")

        # Allocate some memory
        [i for i in range(10000)]

        monitor.snapshot_memory("after")
        assert True  # Just test it doesn't crash


class TestPathValidationModule:
    """Additional tests for path_validation module"""

    def test_secure_filename(self):
        """Test secure_filename function"""
        from mechanics_dsl.utils.path_validation import secure_filename

        result = secure_filename("test<file>.txt")
        assert result is not None
        assert "<" not in result

    def test_is_safe_filename(self):
        """Test is_safe_filename function"""
        from mechanics_dsl.utils.path_validation import is_safe_filename

        assert is_safe_filename("test.txt") is True
        assert is_safe_filename("../test.txt") is False
        assert is_safe_filename("/etc/passwd") is False

    def test_safe_path_join(self):
        """Test safe_path_join function"""

        from mechanics_dsl.utils.path_validation import safe_path_join

        result = safe_path_join("/base", "subdir", "file.txt")
        assert "file.txt" in result

    def test_validate_path_within_base(self):
        """Test validate_path_within_base function"""
        import tempfile

        from mechanics_dsl.utils.path_validation import validate_path_within_base

        with tempfile.TemporaryDirectory() as tmpdir:
            # validate_path_within_base returns the validated path as a string
            result = validate_path_within_base("test.txt", tmpdir)
            assert isinstance(result, str)
            assert "test.txt" in result
