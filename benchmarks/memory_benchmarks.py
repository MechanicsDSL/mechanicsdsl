"""
Memory usage benchmarks for MechanicsDSL.

Tracks memory consumption during compilation and simulation.
"""
import gc
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import tracemalloc

try:
    from mechanics_dsl import PhysicsCompiler
except ImportError:
    PhysicsCompiler = None


@dataclass
class MemoryResult:
    """Result of a memory benchmark."""
    phase: str
    current_mb: float
    peak_mb: float
    allocated_blocks: int
    success: bool
    error: Optional[str] = None


@dataclass
class MemoryBenchmark:
    """Full memory profile of a compilation and simulation."""
    name: str
    compilation: MemoryResult
    simulation: MemoryResult
    total_peak_mb: float
    gc_collected: int


def measure_memory(func, *args, **kwargs) -> tuple:
    """
    Measure memory usage of a function call.
    
    Returns:
        Tuple of (result, current_mb, peak_mb, blocks)
    """
    gc.collect()
    tracemalloc.start()
    
    try:
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()
        blocks = len(snapshot.statistics('lineno'))
        
        return (
            result,
            current / (1024 * 1024),
            peak / (1024 * 1024),
            blocks
        )
    finally:
        tracemalloc.stop()


def benchmark_memory_usage(
    dsl_code: str,
    t_span: tuple = (0, 100),
    num_points: int = 10000
) -> MemoryBenchmark:
    """
    Profile memory usage for compilation and simulation.
    
    Args:
        dsl_code: DSL code to compile
        t_span: Simulation time span
        num_points: Number of output points
        
    Returns:
        MemoryBenchmark with detailed memory stats
    """
    if PhysicsCompiler is None:
        return MemoryBenchmark(
            name="unknown",
            compilation=MemoryResult(
                phase="compilation",
                current_mb=0, peak_mb=0, allocated_blocks=0,
                success=False, error="PhysicsCompiler not available"
            ),
            simulation=MemoryResult(
                phase="simulation",
                current_mb=0, peak_mb=0, allocated_blocks=0,
                success=False, error="PhysicsCompiler not available"
            ),
            total_peak_mb=0,
            gc_collected=0
        )
    
    compiler = PhysicsCompiler()
    
    # Measure compilation
    try:
        comp_result, comp_current, comp_peak, comp_blocks = measure_memory(
            compiler.compile_dsl, dsl_code
        )
        compilation = MemoryResult(
            phase="compilation",
            current_mb=comp_current,
            peak_mb=comp_peak,
            allocated_blocks=comp_blocks,
            success=comp_result.get('success', False)
        )
        system_name = comp_result.get('system_name', 'unknown')
    except Exception as e:
        compilation = MemoryResult(
            phase="compilation",
            current_mb=0, peak_mb=0, allocated_blocks=0,
            success=False, error=str(e)
        )
        system_name = "error"
    
    # Measure simulation
    if compilation.success:
        try:
            sim_result, sim_current, sim_peak, sim_blocks = measure_memory(
                lambda: compiler.simulate(t_span=t_span, num_points=num_points)
            )
            simulation = MemoryResult(
                phase="simulation",
                current_mb=sim_current,
                peak_mb=sim_peak,
                allocated_blocks=sim_blocks,
                success=sim_result.get('success', False)
            )
        except Exception as e:
            simulation = MemoryResult(
                phase="simulation",
                current_mb=0, peak_mb=0, allocated_blocks=0,
                success=False, error=str(e)
            )
    else:
        simulation = MemoryResult(
            phase="simulation",
            current_mb=0, peak_mb=0, allocated_blocks=0,
            success=False, error="Compilation failed"
        )
    
    # Force GC and count collected
    gc_collected = gc.collect()
    
    return MemoryBenchmark(
        name=system_name,
        compilation=compilation,
        simulation=simulation,
        total_peak_mb=max(compilation.peak_mb, simulation.peak_mb),
        gc_collected=gc_collected
    )


def benchmark_memory_scaling(
    dsl_code: str,
    point_counts: List[int] = [100, 1000, 10000, 100000]
) -> Dict[int, MemoryBenchmark]:
    """
    Benchmark memory scaling with output size.
    
    Args:
        dsl_code: DSL code to compile
        point_counts: List of num_points values to test
        
    Returns:
        Dictionary of num_points -> memory benchmark
    """
    results = {}
    
    for n in point_counts:
        results[n] = benchmark_memory_usage(dsl_code, num_points=n)
        print(f"  {n} points: {results[n].simulation.peak_mb:.2f} MB peak")
    
    return results


__all__ = [
    'MemoryResult',
    'MemoryBenchmark',
    'measure_memory',
    'benchmark_memory_usage',
    'benchmark_memory_scaling',
]
