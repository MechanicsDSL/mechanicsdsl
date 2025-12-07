#!/usr/bin/env python3
"""
Performance Comparison Matrix

Compares all backends on a standard benchmark problem.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import tempfile
import sympy as sp

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


def create_pendulum():
    """Standard pendulum for benchmarking."""
    theta = sp.Symbol('theta', real=True)
    g = sp.Symbol('g', positive=True)
    l = sp.Symbol('l', positive=True)
    
    return {
        'system_name': 'benchmark_pendulum',
        'coordinates': ['theta'],
        'parameters': {'g': 9.81, 'l': 1.0},
        'initial_conditions': {'theta': 0.3, 'theta_dot': 0.0},
        'equations': {'theta_ddot': -g/l * sp.sin(theta)}
    }


def benchmark_generation():
    """Benchmark code generation time for all backends."""
    from mechanics_dsl.codegen import (
        CppGenerator, PythonGenerator, JuliaGenerator,
        RustGenerator, MatlabGenerator, FortranGenerator,
        JavaScriptGenerator, CudaGenerator, OpenMPGenerator,
        WasmGenerator, ArduinoGenerator
    )
    
    backends = [
        ('C++', CppGenerator, '.cpp'),
        ('Python', PythonGenerator, '.py'),
        ('Julia', JuliaGenerator, '.jl'),
        ('Rust', RustGenerator, '.rs'),
        ('MATLAB', MatlabGenerator, '.m'),
        ('Fortran', FortranGenerator, '.f90'),
        ('JavaScript', JavaScriptGenerator, '.js'),
        ('OpenMP', OpenMPGenerator, '.cpp'),
        ('Arduino', ArduinoGenerator, '.ino'),
    ]
    
    pendulum = create_pendulum()
    results = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for name, GenClass, ext in backends:
            try:
                start = time.perf_counter()
                gen = GenClass(**pendulum)
                output = os.path.join(tmpdir, f'test{ext}')
                gen.generate(output)
                elapsed = time.perf_counter() - start
                
                # Get file size
                size = os.path.getsize(output)
                
                results.append([
                    name,
                    f"{elapsed*1000:.1f} ms",
                    f"{size} bytes",
                    "✓"
                ])
            except Exception as e:
                results.append([name, "-", "-", f"✗ {str(e)[:20]}"])
        
        # Special handling for CUDA (generates multiple files)
        try:
            start = time.perf_counter()
            cuda_dir = os.path.join(tmpdir, 'cuda')
            gen = CudaGenerator(**pendulum, generate_cpu_fallback=True)
            gen.generate(cuda_dir)
            elapsed = time.perf_counter() - start
            
            total_size = sum(
                os.path.getsize(os.path.join(cuda_dir, f))
                for f in os.listdir(cuda_dir)
            )
            
            results.append([
                'CUDA',
                f"{elapsed*1000:.1f} ms",
                f"{total_size} bytes",
                "✓"
            ])
        except Exception as e:
            results.append(['CUDA', "-", "-", f"✗ {str(e)[:20]}"])
        
        # Special handling for WASM
        try:
            start = time.perf_counter()
            wasm_dir = os.path.join(tmpdir, 'wasm')
            gen = WasmGenerator(**pendulum)
            gen.generate(wasm_dir)
            elapsed = time.perf_counter() - start
            
            total_size = sum(
                os.path.getsize(os.path.join(wasm_dir, f))
                for f in os.listdir(wasm_dir)
            )
            
            results.append([
                'WASM',
                f"{elapsed*1000:.1f} ms",
                f"{total_size} bytes",
                "✓"
            ])
        except Exception as e:
            results.append(['WASM', "-", "-", f"✗ {str(e)[:20]}"])
    
    return results


def benchmark_numba_vs_scipy():
    """Compare Numba and SciPy simulation."""
    from mechanics_dsl.solver_numba import NumbaSimulator, is_numba_available
    import numpy as np
    
    if not is_numba_available():
        return None
    
    theta = sp.Symbol('theta', real=True)
    g = sp.Symbol('g', positive=True)
    l = sp.Symbol('l', positive=True)
    
    accelerations = {'theta_ddot': -g/l * sp.sin(theta)}
    
    # Warmup
    sim = NumbaSimulator()
    sim.set_parameters({'g': 9.81, 'l': 1.0})
    sim.set_initial_conditions({'theta': 0.3, 'theta_dot': 0.0})
    sim.compile_equations(accelerations, ['theta'])
    sim.simulate_numba(t_span=(0, 1), num_points=100)
    
    # Benchmark
    results = []
    for points in [1000, 10000]:
        start = time.perf_counter()
        sim.simulate_numba(t_span=(0, 10), num_points=points, method='rk4')
        elapsed = time.perf_counter() - start
        
        results.append([
            f"Numba ({points} pts)",
            f"{elapsed*1000:.1f} ms"
        ])
    
    return results


def main():
    print("=" * 70)
    print("MECHANICSDSL PERFORMANCE COMPARISON MATRIX")
    print("=" * 70)
    
    print("\n## Code Generation Performance\n")
    gen_results = benchmark_generation()
    
    if HAS_TABULATE:
        print(tabulate(gen_results, 
                      headers=["Backend", "Gen Time", "Output Size", "Status"],
                      tablefmt="github"))
    else:
        print("Backend      | Gen Time  | Output Size | Status")
        print("-------------|-----------|-------------|-------")
        for row in gen_results:
            print(f"{row[0]:12} | {row[1]:9} | {row[2]:11} | {row[3]}")
    
    print("\n## Simulation Performance (Numba)\n")
    numba_results = benchmark_numba_vs_scipy()
    
    if numba_results:
        if HAS_TABULATE:
            print(tabulate(numba_results, 
                          headers=["Method", "Time"],
                          tablefmt="github"))
        else:
            for row in numba_results:
                print(f"{row[0]}: {row[1]}")
    else:
        print("Numba not available")
    
    print("\n## Estimated Runtime Performance\n")
    runtime_estimates = [
        ["CUDA", "Fastest", "10-50x vs CPU", "GPU required"],
        ["OpenMP", "Very Fast", "4-8x vs single", "Multi-core"],
        ["C++/Rust", "Very Fast", "1x baseline", "Compiled"],
        ["Fortran", "Very Fast", "~1x", "Compiled"],
        ["Numba", "Fast", "5-10x vs SciPy", "JIT"],
        ["Julia", "Fast", "~1x C++", "JIT compiled"],
        ["JavaScript", "Medium", "0.5-1x", "V8 engine"],
        ["WASM", "Fast", "0.8-1x C++", "Browser"],
        ["Python", "Slow", "0.1x", "Interpreted"],
        ["MATLAB", "Medium", "0.3-0.5x", "Interpreted+JIT"],
        ["Arduino", "Slow", "MCU limited", "Embedded"],
    ]
    
    if HAS_TABULATE:
        print(tabulate(runtime_estimates,
                      headers=["Backend", "Speed", "vs Baseline", "Notes"],
                      tablefmt="github"))
    else:
        print("Backend    | Speed     | vs Baseline  | Notes")
        print("-----------|-----------|--------------|-------")
        for row in runtime_estimates:
            print(f"{row[0]:10} | {row[1]:9} | {row[2]:12} | {row[3]}")
    
    print("\n" + "=" * 70)
    print("Run individual benchmarks:")
    print("  python benchmarks/numba_performance.py")
    print("  python benchmarks/cuda_performance.py")
    print("=" * 70)


if __name__ == '__main__':
    main()
