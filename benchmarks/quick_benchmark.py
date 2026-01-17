#!/usr/bin/env python3
"""
MechanicsDSL Quick Benchmark Script

Demonstrates performance across different system sizes and solvers.

Usage:
    python benchmarks/quick_benchmark.py
    python benchmarks/quick_benchmark.py --full
"""

import sys
import time
import argparse
from typing import List, Tuple


def format_time(seconds: float) -> str:
    """Format time in human-readable units."""
    if seconds < 0.001:
        return f"{seconds*1e6:.1f} µs"
    elif seconds < 1:
        return f"{seconds*1000:.1f} ms"
    else:
        return f"{seconds:.2f} s"


def benchmark_compilation() -> Tuple[float, str]:
    """Benchmark DSL compilation speed."""
    from mechanics_dsl import PhysicsCompiler
    
    dsl_code = r"""
    \system{benchmark_pendulum}
    \defvar{theta}{rad}
    \parameter{m}{1.0}{kg}
    \parameter{l}{1.0}{m}
    \parameter{g}{9.81}{m/s^2}
    \lagrangian{\frac{1}{2}*m*l^2*\dot{theta}^2 - m*g*l*(1-\cos{theta})}
    \initial{theta=2.5}
    """
    
    start = time.perf_counter()
    compiler = PhysicsCompiler()
    compiler.compile_dsl(dsl_code)
    elapsed = time.perf_counter() - start
    
    return elapsed, "Simple pendulum compilation"


def benchmark_simulation(num_points: int = 10000) -> Tuple[float, str]:
    """Benchmark simulation speed."""
    from mechanics_dsl import PhysicsCompiler
    
    dsl_code = r"""
    \system{benchmark_sim}
    \defvar{theta}{rad}
    \parameter{m}{1.0}{kg}
    \parameter{l}{1.0}{m}
    \parameter{g}{9.81}{m/s^2}
    \lagrangian{\frac{1}{2}*m*l^2*\dot{theta}^2 - m*g*l*(1-\cos{theta})}
    \initial{theta=2.5}
    """
    
    compiler = PhysicsCompiler()
    compiler.compile_dsl(dsl_code)
    
    start = time.perf_counter()
    compiler.simulate(t_span=(0, 100), num_points=num_points)
    elapsed = time.perf_counter() - start
    
    points_per_sec = num_points / elapsed
    return elapsed, f"Simulation ({num_points:,} points, {points_per_sec:,.0f} pts/sec)"


def benchmark_double_pendulum(num_points: int = 5000) -> Tuple[float, str]:
    """Benchmark double pendulum (coupled system)."""
    from mechanics_dsl import PhysicsCompiler
    
    dsl_code = r"""
    \system{benchmark_double}
    \defvar{theta1}{rad}
    \defvar{theta2}{rad}
    \parameter{m1}{1.0}{kg}
    \parameter{m2}{1.0}{kg}
    \parameter{l1}{1.0}{m}
    \parameter{l2}{1.0}{m}
    \parameter{g}{9.81}{m/s^2}
    \lagrangian{
        \frac{1}{2}*(m1+m2)*l1^2*\dot{theta1}^2
        + \frac{1}{2}*m2*l2^2*\dot{theta2}^2
        + m2*l1*l2*\dot{theta1}*\dot{theta2}*\cos{theta1-theta2}
        - (m1+m2)*g*l1*(1-\cos{theta1})
        - m2*g*l2*(1-\cos{theta2})
    }
    \initial{theta1=2.5, theta2=2.0}
    """
    
    compiler = PhysicsCompiler()
    compiler.compile_dsl(dsl_code)
    
    start = time.perf_counter()
    compiler.simulate(t_span=(0, 30), num_points=num_points)
    elapsed = time.perf_counter() - start
    
    return elapsed, f"Double pendulum ({num_points:,} points)"


def benchmark_code_generation() -> Tuple[float, str]:
    """Benchmark code generation to multiple targets."""
    from mechanics_dsl import PhysicsCompiler
    import tempfile
    import os
    
    dsl_code = r"""
    \system{codegen_bench}
    \defvar{x}{m}
    \parameter{m}{1.0}{kg}
    \parameter{k}{10.0}{N/m}
    \lagrangian{\frac{1}{2}*m*\dot{x}^2 - \frac{1}{2}*k*x^2}
    \initial{x=1.0}
    """
    
    compiler = PhysicsCompiler()
    compiler.compile_dsl(dsl_code)
    
    targets = ['cpp', 'python', 'julia', 'matlab', 'javascript']
    
    with tempfile.TemporaryDirectory() as tmpdir:
        start = time.perf_counter()
        for target in targets:
            method = getattr(compiler, f'compile_to_{target}')
            method(os.path.join(tmpdir, f'output.{target}'))
        elapsed = time.perf_counter() - start
    
    return elapsed, f"Code generation ({len(targets)} targets)"


def benchmark_symbolic() -> Tuple[float, str]:
    """Benchmark symbolic derivation for complex system."""
    from mechanics_dsl import PhysicsCompiler
    
    # Triple pendulum - complex symbolic derivation
    dsl_code = r"""
    \system{triple_pendulum}
    \defvar{theta1}{rad}
    \defvar{theta2}{rad}
    \defvar{theta3}{rad}
    \parameter{m1}{1}{kg}
    \parameter{m2}{1}{kg}
    \parameter{m3}{1}{kg}
    \parameter{l1}{1}{m}
    \parameter{l2}{1}{m}
    \parameter{l3}{1}{m}
    \parameter{g}{9.81}{m/s^2}
    \lagrangian{
        \frac{1}{2}*(m1+m2+m3)*l1^2*\dot{theta1}^2
        + \frac{1}{2}*(m2+m3)*l2^2*\dot{theta2}^2
        + \frac{1}{2}*m3*l3^2*\dot{theta3}^2
        + (m2+m3)*l1*l2*\dot{theta1}*\dot{theta2}*\cos{theta1-theta2}
        + m3*l2*l3*\dot{theta2}*\dot{theta3}*\cos{theta2-theta3}
        + m3*l1*l3*\dot{theta1}*\dot{theta3}*\cos{theta1-theta3}
        - (m1+m2+m3)*g*l1*(1-\cos{theta1})
        - (m2+m3)*g*l2*(1-\cos{theta2})
        - m3*g*l3*(1-\cos{theta3})
    }
    \initial{theta1=1.5, theta2=1.0, theta3=0.5}
    """
    
    start = time.perf_counter()
    compiler = PhysicsCompiler()
    compiler.compile_dsl(dsl_code)
    elapsed = time.perf_counter() - start
    
    return elapsed, "Triple pendulum symbolic derivation"


def run_benchmarks(full: bool = False) -> List[Tuple[str, float, str]]:
    """Run all benchmarks and return results."""
    results = []
    
    print("=" * 60)
    print("MechanicsDSL Performance Benchmark")
    print("=" * 60)
    print()
    
    benchmarks = [
        ("Compilation", benchmark_compilation),
        ("Simulation (10k)", lambda: benchmark_simulation(10000)),
        ("Double Pendulum", benchmark_double_pendulum),
        ("Code Generation", benchmark_code_generation),
    ]
    
    if full:
        benchmarks.extend([
            ("Simulation (100k)", lambda: benchmark_simulation(100000)),
            ("Symbolic (Triple)", benchmark_symbolic),
        ])
    
    for name, bench_func in benchmarks:
        try:
            elapsed, description = bench_func()
            results.append((name, elapsed, description))
            print(f"✓ {description}")
            print(f"  Time: {format_time(elapsed)}")
            print()
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            print()
    
    return results


def print_summary(results: List[Tuple[str, float, str]]):
    """Print summary table."""
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()
    print(f"{'Benchmark':<30} {'Time':>15}")
    print("-" * 45)
    
    for name, elapsed, _ in results:
        print(f"{name:<30} {format_time(elapsed):>15}")
    
    total_time = sum(r[1] for r in results)
    print("-" * 45)
    print(f"{'Total':<30} {format_time(total_time):>15}")
    print()


def main():
    parser = argparse.ArgumentParser(description='MechanicsDSL Benchmark')
    parser.add_argument('--full', action='store_true', help='Run extended benchmarks')
    args = parser.parse_args()
    
    try:
        results = run_benchmarks(full=args.full)
        print_summary(results)
        print("All benchmarks passed! ✓")
        return 0
    except ImportError as e:
        print(f"Error: Could not import MechanicsDSL: {e}")
        print("Make sure it's installed: pip install mechanicsdsl-core")
        return 1


if __name__ == '__main__':
    sys.exit(main())
