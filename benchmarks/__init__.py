"""
MechanicsDSL Benchmark Suite

Performance benchmarks for simulation, code generation, and memory usage.

Usage:
    python -m benchmarks.run_benchmarks
    python -m benchmarks.run_benchmarks --backend numba --output results.json
"""

from .core_benchmarks import (
    SimulationBenchmark,
    benchmark_simple_pendulum,
    benchmark_double_pendulum,
    benchmark_n_body,
    benchmark_sph_fluid,
)
from .codegen_benchmarks import (
    CodegenBenchmark,
    benchmark_all_generators,
)
from .memory_benchmarks import (
    MemoryBenchmark,
    benchmark_memory_usage,
)
from .report import (
    BenchmarkReport,
    generate_report,
    compare_reports,
)
from .runner import (
    BenchmarkRunner,
    run_all_benchmarks,
)

__all__ = [
    # Core benchmarks
    'SimulationBenchmark',
    'benchmark_simple_pendulum',
    'benchmark_double_pendulum',
    'benchmark_n_body',
    'benchmark_sph_fluid',
    # Codegen benchmarks
    'CodegenBenchmark',
    'benchmark_all_generators',
    # Memory benchmarks
    'MemoryBenchmark',
    'benchmark_memory_usage',
    # Reporting
    'BenchmarkReport',
    'generate_report',
    'compare_reports',
    # Runner
    'BenchmarkRunner',
    'run_all_benchmarks',
]
