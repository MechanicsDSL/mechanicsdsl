"""
ARM Performance Benchmarks

Benchmark suite for measuring performance on ARM platforms.
Run with: pytest benchmarks/benchmark_arm.py --benchmark-only
"""

import pytest
import numpy as np
import time


# Check if we're running on ARM
def is_arm_platform():
    """Check if current platform is ARM."""
    import platform
    machine = platform.machine().lower()
    return 'arm' in machine or 'aarch64' in machine


ARM_AVAILABLE = is_arm_platform()


class BenchmarkARMSimulation:
    """Benchmarks for ARM-optimized simulations."""
    
    @pytest.fixture
    def simple_pendulum_code(self):
        """Simple pendulum DSL code."""
        return r"""
        \system{pendulum}
        \defvar{theta}{Angle}{rad}
        \parameter{m}{1.0}{kg}
        \parameter{l}{1.0}{m}
        \parameter{g}{9.81}{m/s^2}
        \lagrangian{0.5 * m * l^2 * \dot{theta}^2 - m * g * l * (1 - \cos{theta})}
        \initial{theta=0.5, theta_dot=0.0}
        """
    
    @pytest.fixture
    def double_pendulum_code(self):
        """Double pendulum DSL code."""
        return r"""
        \system{double_pendulum}
        \defvar{theta1}{Angle}{rad}
        \defvar{theta2}{Angle}{rad}
        \parameter{m}{1.0}{kg}
        \parameter{l}{1.0}{m}
        \parameter{g}{9.81}{m/s^2}
        \lagrangian{
            0.5 * m * l^2 * (2 * \dot{theta1}^2 + \dot{theta2}^2 
            + 2 * \dot{theta1} * \dot{theta2} * \cos{theta1 - theta2})
            + m * g * l * (2 * \cos{theta1} + \cos{theta2})
        }
        \initial{theta1=2.0, theta1_dot=0, theta2=2.5, theta2_dot=0}
        """
    
    @pytest.mark.benchmark(group="compilation")
    def test_benchmark_compilation(self, benchmark, simple_pendulum_code):
        """Benchmark DSL compilation time."""
        from mechanics_dsl import PhysicsCompiler
        
        def compile_dsl():
            compiler = PhysicsCompiler()
            compiler.compile_dsl(simple_pendulum_code)
            return compiler
        
        result = benchmark(compile_dsl)
        assert result is not None
    
    @pytest.mark.benchmark(group="simulation")
    def test_benchmark_simulation_short(self, benchmark, simple_pendulum_code):
        """Benchmark short simulation (1 second)."""
        from mechanics_dsl import PhysicsCompiler
        
        compiler = PhysicsCompiler()
        compiler.compile_dsl(simple_pendulum_code)
        
        def run_simulation():
            return compiler.simulate(t_span=(0, 1), num_points=100)
        
        result = benchmark(run_simulation)
        assert 't' in result
    
    @pytest.mark.benchmark(group="simulation")
    def test_benchmark_simulation_long(self, benchmark, simple_pendulum_code):
        """Benchmark longer simulation (10 seconds, 1000 points)."""
        from mechanics_dsl import PhysicsCompiler
        
        compiler = PhysicsCompiler()
        compiler.compile_dsl(simple_pendulum_code)
        
        def run_simulation():
            return compiler.simulate(t_span=(0, 10), num_points=1000)
        
        result = benchmark(run_simulation)
        assert len(result['t']) == 1000
    
    @pytest.mark.benchmark(group="codegen")
    def test_benchmark_cpp_codegen(self, benchmark, simple_pendulum_code):
        """Benchmark C++ code generation."""
        from mechanics_dsl import PhysicsCompiler
        from mechanics_dsl.codegen.cpp import CppGenerator
        import tempfile
        import os
        
        compiler = PhysicsCompiler()
        compiler.compile_dsl(simple_pendulum_code)
        
        def generate_cpp():
            with tempfile.TemporaryDirectory() as tmpdir:
                gen = CppGenerator(
                    system_name="pendulum",
                    coordinates=list(compiler.simulator.coordinates.keys()) if hasattr(compiler.simulator.coordinates, 'keys') else compiler.simulator.coordinates,
                    parameters=compiler.simulator.parameters,
                    initial_conditions=compiler.simulator.initial_conditions,
                    equations=compiler.simulator.equations
                )
                return gen.generate(os.path.join(tmpdir, "pendulum.cpp"))
        
        result = benchmark(generate_cpp)
        assert result is not None
    
    @pytest.mark.benchmark(group="codegen")
    @pytest.mark.skipif(not ARM_AVAILABLE, reason="Not on ARM platform")
    def test_benchmark_arm_codegen(self, benchmark, simple_pendulum_code):
        """Benchmark ARM code generation."""
        from mechanics_dsl import PhysicsCompiler
        from mechanics_dsl.codegen.arm import ARMGenerator
        import tempfile
        
        compiler = PhysicsCompiler()
        compiler.compile_dsl(simple_pendulum_code)
        
        def generate_arm():
            with tempfile.TemporaryDirectory() as tmpdir:
                gen = ARMGenerator(
                    system_name="pendulum",
                    coordinates=list(compiler.simulator.coordinates.keys()) if hasattr(compiler.simulator.coordinates, 'keys') else compiler.simulator.coordinates,
                    parameters=compiler.simulator.parameters,
                    initial_conditions=compiler.simulator.initial_conditions,
                    equations=compiler.simulator.equations,
                    target="raspberry_pi",
                    use_neon=True
                )
                return gen.generate_project(tmpdir)
        
        result = benchmark(generate_arm)
        assert result is not None


class BenchmarkNumericalOperations:
    """Benchmarks for numerical operations."""
    
    @pytest.mark.benchmark(group="numpy")
    def test_benchmark_matrix_multiply(self, benchmark):
        """Benchmark matrix multiplication (OpenBLAS on ARM)."""
        A = np.random.randn(100, 100)
        B = np.random.randn(100, 100)
        
        result = benchmark(np.dot, A, B)
        assert result.shape == (100, 100)
    
    @pytest.mark.benchmark(group="numpy")
    def test_benchmark_eigenvalues(self, benchmark):
        """Benchmark eigenvalue computation."""
        A = np.random.randn(50, 50)
        A = A @ A.T  # Make symmetric
        
        result = benchmark(np.linalg.eigvalsh, A)
        assert len(result) == 50
    
    @pytest.mark.benchmark(group="numpy")
    def test_benchmark_fft(self, benchmark):
        """Benchmark FFT (important for signal processing)."""
        x = np.random.randn(1024)
        
        result = benchmark(np.fft.fft, x)
        assert len(result) == 1024


class BenchmarkPlatformInfo:
    """Report platform information for benchmarks."""
    
    def test_platform_info(self):
        """Print platform information for context."""
        import platform
        import sys
        
        print("\n" + "="*60)
        print("PLATFORM INFORMATION")
        print("="*60)
        print(f"Machine: {platform.machine()}")
        print(f"Platform: {platform.platform()}")
        print(f"Python: {sys.version}")
        print(f"NumPy: {np.__version__}")
        
        # Check for BLAS
        try:
            np.__config__.show()
        except:
            pass
        
        print(f"ARM Platform: {ARM_AVAILABLE}")
        print("="*60)
        
        assert True  # Always pass
