"""Tests for mechanics_dsl.protocols module."""

import pytest

from mechanics_dsl.protocols import (
    CacheProtocol,
    CodeGeneratorProtocol,
    CompilableProtocol,
    PhysicsDomainProtocol,
    SimulatableProtocol,
    SymbolicExpressionProtocol,
    ValidatorProtocol,
    VisualizableProtocol,
)


class TestSymbolicExpressionProtocol:
    """Tests for SymbolicExpressionProtocol."""

    def test_conforming_class(self):
        class MockExpr:
            @property
            def free_symbols(self):
                return set()

            def subs(self, substitutions):
                return self

        assert isinstance(MockExpr(), SymbolicExpressionProtocol)

    def test_non_conforming(self):
        assert not isinstance("not an expr", SymbolicExpressionProtocol)
        assert not isinstance(42, SymbolicExpressionProtocol)

    def test_sympy_expr_conforms(self):
        """Real sympy expressions should satisfy the protocol."""
        import sympy as sp

        x = sp.Symbol("x")
        expr = sp.sin(x) + x**2
        assert isinstance(expr, SymbolicExpressionProtocol)


class TestSimulatableProtocol:
    """Tests for SimulatableProtocol."""

    def test_conforming_class(self):
        class MockSimulator:
            def set_parameters(self, params):
                pass

            def set_initial_conditions(self, conditions):
                pass

            def simulate(self, t_span, num_points=1000):
                return {"t": [], "y": [], "success": True}

        assert isinstance(MockSimulator(), SimulatableProtocol)

    def test_non_conforming(self):
        class Incomplete:
            def simulate(self, t_span):
                pass

        # Missing set_parameters and set_initial_conditions
        assert not isinstance(Incomplete(), SimulatableProtocol)


class TestCompilableProtocol:
    """Tests for CompilableProtocol."""

    def test_conforming_class(self):
        class MockCompiler:
            def compile_dsl(self, source, **kwargs):
                return {"success": True}

        assert isinstance(MockCompiler(), CompilableProtocol)

    def test_physics_compiler_conforms(self):
        """The real PhysicsCompiler should satisfy CompilableProtocol."""
        from mechanics_dsl import PhysicsCompiler

        compiler = PhysicsCompiler()
        assert isinstance(compiler, CompilableProtocol)


class TestVisualizableProtocol:
    """Tests for VisualizableProtocol."""

    def test_conforming_class(self):
        class MockViz:
            def plot(self, solution, **kwargs):
                pass

            def animate(self, solution, **kwargs):
                pass

        assert isinstance(MockViz(), VisualizableProtocol)

    def test_non_conforming(self):
        class PlotOnly:
            def plot(self, solution):
                pass

        assert not isinstance(PlotOnly(), VisualizableProtocol)


class TestCodeGeneratorProtocol:
    """Tests for CodeGeneratorProtocol."""

    def test_conforming_class(self):
        class MockGen:
            @property
            def target_name(self):
                return "test"

            @property
            def file_extension(self):
                return ".test"

            def generate(self, output_file):
                return output_file

            def generate_equations(self):
                return ""

        assert isinstance(MockGen(), CodeGeneratorProtocol)


class TestPhysicsDomainProtocol:
    """Tests for PhysicsDomainProtocol."""

    def test_conforming_class(self):
        class MockDomain:
            @property
            def name(self):
                return "test"

            @property
            def coordinates(self):
                return ["x"]

            @property
            def parameters(self):
                return {"m": 1.0}

            def define_lagrangian(self):
                return None

            def derive_equations_of_motion(self):
                return {}

        assert isinstance(MockDomain(), PhysicsDomainProtocol)


class TestCacheProtocol:
    """Tests for CacheProtocol."""

    def test_conforming_class(self):
        class MockCache:
            def get(self, key):
                return None

            def set(self, key, value):
                pass

            def clear(self):
                pass

            @property
            def hit_rate(self):
                return 0.0

        assert isinstance(MockCache(), CacheProtocol)

    def test_lru_cache_partial_conformance(self):
        """LRUCache has get/set/clear but uses get_stats() instead of hit_rate property."""
        from mechanics_dsl.utils.caching import LRUCache

        cache = LRUCache(maxsize=10)
        # LRUCache has get, set, clear but not hit_rate property
        assert hasattr(cache, "get")
        assert hasattr(cache, "set")
        assert hasattr(cache, "clear")


class TestValidatorProtocol:
    """Tests for ValidatorProtocol."""

    def test_conforming_class(self):
        class MockValidator:
            def validate(self, value):
                return (True, None)

        assert isinstance(MockValidator(), ValidatorProtocol)

    def test_non_conforming(self):
        assert not isinstance({}, ValidatorProtocol)
        assert not isinstance([], ValidatorProtocol)


class TestRuntimeCheckable:
    """Verify all protocols are runtime_checkable."""

    @pytest.mark.parametrize(
        "protocol",
        [
            SymbolicExpressionProtocol,
            SimulatableProtocol,
            CompilableProtocol,
            VisualizableProtocol,
            CodeGeneratorProtocol,
            PhysicsDomainProtocol,
            CacheProtocol,
            ValidatorProtocol,
        ],
    )
    def test_is_runtime_checkable(self, protocol):
        # runtime_checkable protocols support isinstance checks
        assert not isinstance("arbitrary string", protocol)
