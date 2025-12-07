"""
Unit tests for MechanicsDSL compiler module.

Tests the PhysicsCompiler and SystemSerializer classes.
"""

import pytest
import json
import os
import tempfile
import numpy as np

from mechanics_dsl.compiler import PhysicsCompiler, SystemSerializer


class TestPhysicsCompilerInit:
    """Tests for PhysicsCompiler initialization."""

    def test_init_creates_instance(self):
        """Test that PhysicsCompiler can be instantiated."""
        compiler = PhysicsCompiler()
        assert compiler is not None

    def test_init_has_ast_attribute(self):
        """Test that compiler has ast attribute."""
        compiler = PhysicsCompiler()
        assert hasattr(compiler, 'ast')

    def test_init_symbolic_engine(self):
        """Test that compiler has symbolic engine."""
        compiler = PhysicsCompiler()
        assert compiler.symbolic is not None

    def test_init_simulator(self):
        """Test that compiler has simulator."""
        compiler = PhysicsCompiler()
        assert compiler.simulator is not None

    def test_init_visualizer(self):
        """Test that compiler has visualizer."""
        compiler = PhysicsCompiler()
        assert compiler.visualizer is not None


class TestPhysicsCompilerContextManager:
    """Tests for PhysicsCompiler as context manager."""

    def test_context_manager_enter(self):
        """Test context manager entry."""
        with PhysicsCompiler() as compiler:
            assert compiler is not None

    def test_context_manager_cleanup(self):
        """Test that context manager calls cleanup."""
        compiler = PhysicsCompiler()
        with compiler:
            pass
        # Should not raise

    def test_context_manager_exception(self):
        """Test context manager with exception."""
        try:
            with PhysicsCompiler() as compiler:
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected


class TestCompileDSL:
    """Tests for PhysicsCompiler.compile_dsl method."""

    def test_compile_simple_pendulum(self):
        """Test compiling simple pendulum DSL."""
        compiler = PhysicsCompiler()
        dsl = r"""
        \system{pendulum}
        
        \defvar{theta}{Angle}{rad}
        
        \parameter{m}{1.0}{kg}
        \parameter{L}{1.0}{m}
        \parameter{g}{9.81}{m/s^2}
        
        \lagrangian{\frac{1}{2} * m * L^2 * \dot{theta}^2 - m * g * L * (1 - \cos{theta})}
        
        \initial{theta=0.5, theta_dot=0.0}
        """
        result = compiler.compile_dsl(dsl)
        assert result['success'] is True

    def test_compile_harmonic_oscillator(self):
        """Test compiling harmonic oscillator DSL."""
        compiler = PhysicsCompiler()
        dsl = r"""
        \system{oscillator}
        
        \defvar{x}{Position}{m}
        
        \parameter{m}{1.0}{kg}
        \parameter{k}{10.0}{N/m}
        
        \lagrangian{\frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2}
        
        \initial{x=1.0, x_dot=0.0}
        """
        result = compiler.compile_dsl(dsl)
        assert result['success'] is True

    def test_compile_returns_dict(self):
        """Test that compile_dsl returns a dictionary."""
        compiler = PhysicsCompiler()
        dsl = r"""
        \system{test}
        \defvar{x}{Position}{m}
        \parameter{m}{1.0}{kg}
        \lagrangian{\frac{1}{2} * m * \dot{x}^2}
        \initial{x=0.0, x_dot=1.0}
        """
        result = compiler.compile_dsl(dsl)
        assert isinstance(result, dict)

    def test_compile_empty_dsl_fails(self):
        """Test that empty DSL raises error."""
        compiler = PhysicsCompiler()
        with pytest.raises((ValueError, TypeError)):
            compiler.compile_dsl("")

    def test_compile_invalid_dsl_type(self):
        """Test that non-string DSL raises TypeError."""
        compiler = PhysicsCompiler()
        with pytest.raises(TypeError):
            compiler.compile_dsl(123)

    def test_compile_contains_success_key(self):
        """Test that result contains success key."""
        compiler = PhysicsCompiler()
        dsl = r"""
        \system{test}
        \defvar{x}{Position}{m}
        \parameter{m}{1.0}{kg}
        \lagrangian{\frac{1}{2} * m * \dot{x}^2}
        \initial{x=0.0, x_dot=1.0}
        """
        result = compiler.compile_dsl(dsl)
        assert 'success' in result


class TestSimulate:
    """Tests for PhysicsCompiler.simulate method."""

    def test_simulate_after_compile(self):
        """Test simulation after successful compilation."""
        compiler = PhysicsCompiler()
        dsl = r"""
        \system{pendulum}
        \defvar{theta}{Angle}{rad}
        \parameter{m}{1.0}{kg}
        \parameter{L}{1.0}{m}
        \parameter{g}{9.81}{m/s^2}
        \lagrangian{\frac{1}{2} * m * L^2 * \dot{theta}^2 - m * g * L * (1 - \cos{theta})}
        \initial{theta=0.3, theta_dot=0.0}
        """
        result = compiler.compile_dsl(dsl)
        assert result['success']
        
        solution = compiler.simulate((0, 5), num_points=100)
        assert solution is not None

    def test_simulate_returns_dict(self):
        """Test that simulate returns a dictionary."""
        compiler = PhysicsCompiler()
        dsl = r"""
        \system{oscillator}
        \defvar{x}{Position}{m}
        \parameter{m}{1.0}{kg}
        \parameter{k}{1.0}{N/m}
        \lagrangian{\frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2}
        \initial{x=1.0, x_dot=0.0}
        """
        result = compiler.compile_dsl(dsl)
        if result['success']:
            solution = compiler.simulate((0, 2), num_points=50)
            assert isinstance(solution, dict)

    def test_simulate_custom_time_span(self):
        """Test simulation with custom time span."""
        compiler = PhysicsCompiler()
        dsl = r"""
        \system{oscillator}
        \defvar{x}{Position}{m}
        \parameter{m}{1.0}{kg}
        \parameter{k}{1.0}{N/m}
        \lagrangian{\frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2}
        \initial{x=1.0, x_dot=0.0}
        """
        result = compiler.compile_dsl(dsl)
        if result['success']:
            solution = compiler.simulate((0, 20), num_points=200)
            assert solution['t'][-1] == pytest.approx(20.0)


class TestGetCoordinates:
    """Tests for PhysicsCompiler.get_coordinates method."""

    def test_get_single_coordinate(self):
        """Test getting single coordinate."""
        compiler = PhysicsCompiler()
        dsl = r"""
        \system{test}
        \defvar{x}{Position}{m}
        \parameter{m}{1.0}{kg}
        \lagrangian{\frac{1}{2} * m * \dot{x}^2}
        \initial{x=0.0, x_dot=1.0}
        """
        result = compiler.compile_dsl(dsl)
        if result['success']:
            coords = compiler.get_coordinates()
            assert len(coords) >= 1

    def test_get_coordinates_returns_list(self):
        """Test that get_coordinates returns a list."""
        compiler = PhysicsCompiler()
        dsl = r"""
        \system{test}
        \defvar{x}{Position}{m}
        \parameter{m}{1.0}{kg}
        \lagrangian{\frac{1}{2} * m * \dot{x}^2}
        \initial{x=0.0, x_dot=1.0}
        """
        result = compiler.compile_dsl(dsl)
        if result['success']:
            coords = compiler.get_coordinates()
            assert isinstance(coords, list)


class TestSystemSerializer:
    """Tests for SystemSerializer class."""

    def test_export_json_format(self):
        """Test exporting system to JSON format."""
        compiler = PhysicsCompiler()
        dsl = r"""
        \system{test}
        \defvar{x}{Position}{m}
        \parameter{m}{1.0}{kg}
        \lagrangian{\frac{1}{2} * m * \dot{x}^2}
        \initial{x=1.0, x_dot=0.0}
        """
        result = compiler.compile_dsl(dsl)
        if not result['success']:
            pytest.skip("Compilation failed")
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filename = f.name
        
        try:
            result = SystemSerializer.export_system(compiler, filename, format='json')
            assert result is True
            assert os.path.exists(filename)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_export_creates_file(self):
        """Test that export creates a file."""
        compiler = PhysicsCompiler()
        dsl = r"""
        \system{test}
        \defvar{x}{Position}{m}
        \parameter{m}{1.0}{kg}
        \lagrangian{\frac{1}{2} * m * \dot{x}^2}
        \initial{x=1.0, x_dot=0.0}
        """
        result = compiler.compile_dsl(dsl)
        if not result['success']:
            pytest.skip("Compilation failed")
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filename = f.name
        
        try:
            SystemSerializer.export_system(compiler, filename)
            assert os.path.exists(filename)
            assert os.path.getsize(filename) > 0
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_import_nonexistent_file(self):
        """Test importing from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            SystemSerializer.import_system("nonexistent_file_12345.json")

    def test_import_invalid_filename_type(self):
        """Test that invalid filename type raises TypeError."""
        with pytest.raises(TypeError):
            SystemSerializer.import_system(12345)


class TestCleanup:
    """Tests for PhysicsCompiler.cleanup method."""

    def test_cleanup_does_not_raise(self):
        """Test that cleanup doesn't raise exceptions."""
        compiler = PhysicsCompiler()
        compiler.cleanup()  # Should not raise

    def test_cleanup_after_compile(self):
        """Test cleanup after compilation."""
        compiler = PhysicsCompiler()
        dsl = r"""
        \system{test}
        \defvar{x}{Position}{m}
        \parameter{m}{1.0}{kg}
        \lagrangian{\frac{1}{2} * m * \dot{x}^2}
        \initial{x=1.0, x_dot=0.0}
        """
        result = compiler.compile_dsl(dsl)
        compiler.cleanup()  # Should not raise
