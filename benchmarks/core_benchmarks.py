"""
Core simulation benchmarks for MechanicsDSL.

Benchmarks for simulation speed across different systems and backends.
"""
import time
import gc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import numpy as np

try:
    from mechanics_dsl import PhysicsCompiler
except ImportError:
    PhysicsCompiler = None


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    backend: str
    time_ms: float
    memory_mb: float
    num_evaluations: int
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationBenchmark:
    """
    Configuration for a simulation benchmark.
    
    Defines the DSL code, time span, and expected characteristics.
    """
    name: str
    dsl_code: str
    t_span: tuple = (0, 100)
    num_points: int = 10000
    description: str = ""
    expected_dof: int = 1
    is_stiff: bool = False
    tags: List[str] = field(default_factory=list)


# Standard benchmark cases
SIMPLE_PENDULUM = SimulationBenchmark(
    name="simple_pendulum",
    description="Single degree of freedom pendulum - baseline benchmark",
    dsl_code=r"""
\system{simple_pendulum}
\defvar{theta}{Angle}{rad}
\defvar{m}{Mass}{kg}
\defvar{l}{Length}{m}
\defvar{g}{Acceleration}{m/s^2}

\parameter{m}{1.0}{kg}
\parameter{l}{1.0}{m}
\parameter{g}{9.81}{m/s^2}

\lagrangian{
    \frac{1}{2} * m * l^2 * \dot{theta}^2 
    - m * g * l * (1 - \cos{theta})
}

\initial{theta=2.5, theta_dot=0.0}
""",
    t_span=(0, 100),
    num_points=10000,
    expected_dof=1,
    tags=["baseline", "pendulum"]
)


DOUBLE_PENDULUM = SimulationBenchmark(
    name="double_pendulum",
    description="Chaotic double pendulum system",
    dsl_code=r"""
\system{double_pendulum}
\defvar{theta1}{Angle}{rad}
\defvar{theta2}{Angle}{rad}
\defvar{m1}{Mass}{kg}
\defvar{m2}{Mass}{kg}
\defvar{l1}{Length}{m}
\defvar{l2}{Length}{m}
\defvar{g}{Acceleration}{m/s^2}

\parameter{m1}{1.0}{kg}
\parameter{m2}{1.0}{kg}
\parameter{l1}{1.0}{m}
\parameter{l2}{1.0}{m}
\parameter{g}{9.81}{m/s^2}

\lagrangian{
    \frac{1}{2} * (m1 + m2) * l1^2 * \dot{theta1}^2
    + \frac{1}{2} * m2 * l2^2 * \dot{theta2}^2
    + m2 * l1 * l2 * \dot{theta1} * \dot{theta2} * \cos{theta1 - theta2}
    - (m1 + m2) * g * l1 * (1 - \cos{theta1})
    - m2 * g * l2 * (1 - \cos{theta2})
}

\initial{theta1=2.5, theta1_dot=0.0, theta2=2.0, theta2_dot=0.0}
""",
    t_span=(0, 100),
    num_points=10000,
    expected_dof=2,
    tags=["chaotic", "pendulum"]
)


COUPLED_OSCILLATORS = SimulationBenchmark(
    name="coupled_oscillators",
    description="Three coupled spring-mass oscillators",
    dsl_code=r"""
\system{coupled_oscillators}
\defvar{x1}{Position}{m}
\defvar{x2}{Position}{m}
\defvar{x3}{Position}{m}
\defvar{m}{Mass}{kg}
\defvar{k}{Spring Constant}{N/m}
\defvar{k_c}{Coupling Constant}{N/m}

\parameter{m}{1.0}{kg}
\parameter{k}{10.0}{N/m}
\parameter{k_c}{2.0}{N/m}

\lagrangian{
    \frac{1}{2} * m * (\dot{x1}^2 + \dot{x2}^2 + \dot{x3}^2)
    - \frac{1}{2} * k * (x1^2 + x2^2 + x3^2)
    - \frac{1}{2} * k_c * ((x1 - x2)^2 + (x2 - x3)^2)
}

\initial{x1=1.0, x1_dot=0.0, x2=0.0, x2_dot=0.0, x3=0.0, x3_dot=0.0}
""",
    t_span=(0, 100),
    num_points=10000,
    expected_dof=3,
    tags=["coupled", "oscillator"]
)


DAMPED_DRIVEN = SimulationBenchmark(
    name="damped_driven",
    description="Damped driven oscillator (non-conservative)",
    dsl_code=r"""
\system{damped_driven}
\defvar{x}{Position}{m}
\defvar{m}{Mass}{kg}
\defvar{k}{Spring Constant}{N/m}
\defvar{b}{Damping Coeff}{N*s/m}
\defvar{F0}{Force Amplitude}{N}
\defvar{omega}{Drive Frequency}{rad/s}

\parameter{m}{1.0}{kg}
\parameter{k}{10.0}{N/m}
\parameter{b}{0.5}{N*s/m}
\parameter{F0}{2.0}{N}
\parameter{omega}{3.0}{rad/s}

\lagrangian{
    \frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2
}

\force{-b * x_dot + F0 * \cos{omega * t}}

\initial{x=0.0, x_dot=0.0}
""",
    t_span=(0, 100),
    num_points=10000,
    expected_dof=1,
    tags=["damped", "driven", "non-conservative"]
)


# All standard benchmarks
STANDARD_BENCHMARKS = [
    SIMPLE_PENDULUM,
    DOUBLE_PENDULUM,
    COUPLED_OSCILLATORS,
    DAMPED_DRIVEN,
]


def run_benchmark(
    benchmark: SimulationBenchmark,
    backend: str = "scipy",
    warmup: int = 1,
    repeats: int = 3,
) -> BenchmarkResult:
    """
    Run a simulation benchmark.
    
    Args:
        benchmark: Benchmark configuration
        backend: Solver backend ("scipy", "numba", "jax")
        warmup: Number of warmup runs
        repeats: Number of timed runs
        
    Returns:
        BenchmarkResult with timing and memory info
    """
    if PhysicsCompiler is None:
        return BenchmarkResult(
            name=benchmark.name,
            backend=backend,
            time_ms=0,
            memory_mb=0,
            num_evaluations=0,
            success=False,
            error="PhysicsCompiler not available"
        )
    
    try:
        # Compile
        compiler = PhysicsCompiler()
        result = compiler.compile_dsl(benchmark.dsl_code)
        
        if not result['success']:
            return BenchmarkResult(
                name=benchmark.name,
                backend=backend,
                time_ms=0,
                memory_mb=0,
                num_evaluations=0,
                success=False,
                error=result.get('error', 'Compilation failed')
            )
        
        # Warmup runs
        for _ in range(warmup):
            compiler.simulate(
                t_span=benchmark.t_span,
                num_points=benchmark.num_points // 10  # Shorter warmup
            )
        
        # Timed runs
        gc.collect()
        times = []
        nfevs = []
        
        for _ in range(repeats):
            start = time.perf_counter()
            solution = compiler.simulate(
                t_span=benchmark.t_span,
                num_points=benchmark.num_points
            )
            end = time.perf_counter()
            
            times.append((end - start) * 1000)  # ms
            nfevs.append(solution.get('nfev', 0))
        
        # Memory estimate (rough)
        import sys
        memory_mb = sys.getsizeof(solution.get('y', np.array([]))) / (1024 * 1024)
        
        return BenchmarkResult(
            name=benchmark.name,
            backend=backend,
            time_ms=np.median(times),
            memory_mb=memory_mb,
            num_evaluations=int(np.median(nfevs)),
            success=True,
            metadata={
                'times_ms': times,
                'num_points': benchmark.num_points,
                't_span': benchmark.t_span,
            }
        )
        
    except Exception as e:
        return BenchmarkResult(
            name=benchmark.name,
            backend=backend,
            time_ms=0,
            memory_mb=0,
            num_evaluations=0,
            success=False,
            error=str(e)
        )


def benchmark_simple_pendulum(backend: str = "scipy") -> BenchmarkResult:
    """Run simple pendulum benchmark."""
    return run_benchmark(SIMPLE_PENDULUM, backend)


def benchmark_double_pendulum(backend: str = "scipy") -> BenchmarkResult:
    """Run double pendulum benchmark."""
    return run_benchmark(DOUBLE_PENDULUM, backend)


def benchmark_n_body(n: int = 10, backend: str = "scipy") -> BenchmarkResult:
    """
    Run N-body gravitational benchmark.
    
    Args:
        n: Number of bodies
        backend: Solver backend
    """
    # Generate N-body DSL code dynamically
    coords = ", ".join([f"x{i}, y{i}" for i in range(n)])
    dsl_code = f"""
\\system{{n_body_{n}}}
"""
    # This would need dynamic generation - placeholder for now
    return BenchmarkResult(
        name=f"n_body_{n}",
        backend=backend,
        time_ms=0,
        memory_mb=0,
        num_evaluations=0,
        success=False,
        error="N-body benchmark not yet implemented"
    )


def benchmark_sph_fluid(n_particles: int = 1000, backend: str = "scipy") -> BenchmarkResult:
    """
    Run SPH fluid simulation benchmark.
    
    Args:
        n_particles: Number of fluid particles
        backend: Solver backend
    """
    # Placeholder for SPH benchmark
    return BenchmarkResult(
        name=f"sph_fluid_{n_particles}",
        backend=backend,
        time_ms=0,
        memory_mb=0,
        num_evaluations=0,
        success=False,
        error="SPH benchmark not yet implemented"
    )


def run_all_core_benchmarks(
    backends: List[str] = ["scipy"],
) -> List[BenchmarkResult]:
    """
    Run all standard benchmarks across specified backends.
    
    Args:
        backends: List of backends to benchmark
        
    Returns:
        List of all benchmark results
    """
    results = []
    
    for benchmark in STANDARD_BENCHMARKS:
        for backend in backends:
            result = run_benchmark(benchmark, backend)
            results.append(result)
            print(f"  {benchmark.name}/{backend}: {result.time_ms:.2f}ms")
    
    return results


__all__ = [
    'BenchmarkResult',
    'SimulationBenchmark',
    'SIMPLE_PENDULUM',
    'DOUBLE_PENDULUM',
    'COUPLED_OSCILLATORS',
    'DAMPED_DRIVEN',
    'STANDARD_BENCHMARKS',
    'run_benchmark',
    'benchmark_simple_pendulum',
    'benchmark_double_pendulum',
    'benchmark_n_body',
    'benchmark_sph_fluid',
    'run_all_core_benchmarks',
]
