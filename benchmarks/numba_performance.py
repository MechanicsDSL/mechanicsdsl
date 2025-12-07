#!/usr/bin/env python3
"""
Benchmark: Numba vs SciPy Solver Performance

Compares execution time for increasing problem sizes.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import numpy as np
import sympy as sp
from tabulate import tabulate

from mechanics_dsl.solver_numba import NumbaSimulator, is_numba_available
from mechanics_dsl.solver import NumericalSimulator
from mechanics_dsl.symbolic import SymbolicEngine


def create_simple_pendulum():
    """Create simple pendulum equations."""
    theta = sp.Symbol('theta', real=True)
    theta_dot = sp.Symbol('theta_dot', real=True)
    g = sp.Symbol('g', positive=True)
    l = sp.Symbol('l', positive=True)
    
    return {
        'accelerations': {'theta_ddot': -g/l * sp.sin(theta)},
        'coordinates': ['theta'],
        'parameters': {'g': 9.81, 'l': 1.0},
        'initial_conditions': {'theta': 0.3, 'theta_dot': 0.0}
    }


def benchmark_numba(system, num_points):
    """Benchmark Numba solver."""
    sim = NumbaSimulator()
    sim.set_parameters(system['parameters'])
    sim.set_initial_conditions(system['initial_conditions'])
    sim.compile_equations(system['accelerations'], system['coordinates'])
    
    start = time.perf_counter()
    solution = sim.simulate_numba(t_span=(0, 10), num_points=num_points, method='rk4')
    elapsed = time.perf_counter() - start
    
    return elapsed, solution


def benchmark_scipy(system, num_points):
    """Benchmark SciPy solver."""
    engine = SymbolicEngine("pendulum")
    engine.coordinates = system['coordinates']
    engine.parameters = system['parameters']
    
    sim = NumericalSimulator(engine)
    sim.set_parameters(system['parameters'])
    sim.set_initial_conditions(system['initial_conditions'])
    
    # Compile equations
    sim.compile_equations(system['accelerations'], system['coordinates'])
    
    start = time.perf_counter()
    solution = sim.simulate(t_span=(0, 10), num_points=num_points, method='RK45')
    elapsed = time.perf_counter() - start
    
    return elapsed, solution


def main():
    print("=" * 60)
    print("NUMBA vs SCIPY PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    if not is_numba_available():
        print("\nNumba not available. Install with: pip install numba")
        return
    
    system = create_simple_pendulum()
    
    # Warmup Numba JIT
    print("\nWarming up Numba JIT...")
    benchmark_numba(system, 100)
    
    # Test different problem sizes
    test_sizes = [100, 1000, 5000, 10000, 50000]
    results = []
    
    print("\nRunning benchmarks...")
    for n in test_sizes:
        try:
            numba_time, numba_sol = benchmark_numba(system, n)
            scipy_time, scipy_sol = benchmark_scipy(system, n)
            
            speedup = scipy_time / numba_time if numba_time > 0 else float('inf')
            
            results.append([
                n,
                f"{numba_time*1000:.2f} ms",
                f"{scipy_time*1000:.2f} ms",
                f"{speedup:.1f}x"
            ])
        except Exception as e:
            results.append([n, "ERROR", "ERROR", str(e)[:20]])
    
    print("\n" + tabulate(results, 
                          headers=["Points", "Numba", "SciPy", "Speedup"],
                          tablefmt="grid"))
    
    # Verify accuracy
    print("\nAccuracy verification (last valid run):")
    if scipy_sol and numba_sol:
        max_diff = np.max(np.abs(scipy_sol['y'][0] - numba_sol['y'][0]))
        print(f"  Max difference: {max_diff:.2e}")
        if max_diff < 1e-3:
            print("  ✅ Solutions match!")
        else:
            print("  ⚠️  Solutions differ significantly")


if __name__ == '__main__':
    try:
        from tabulate import tabulate
    except ImportError:
        print("Installing tabulate...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tabulate', '-q'])
        from tabulate import tabulate
    
    main()
