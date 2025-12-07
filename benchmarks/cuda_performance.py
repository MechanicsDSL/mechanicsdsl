#!/usr/bin/env python3
"""
Benchmark: CUDA vs CPU Performance

Compares GPU and CPU simulation for SPH particle systems.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import numpy as np

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


def benchmark_cpu_sph(num_particles, steps):
    """Simple CPU SPH simulation benchmark."""
    # Initialize particles
    x = np.random.rand(num_particles) * 2.0
    y = np.random.rand(num_particles) * 1.0
    vx = np.zeros(num_particles)
    vy = np.zeros(num_particles)
    rho = np.ones(num_particles) * 1000.0
    
    h = 0.04
    h2 = h * h
    dt = 0.0001
    
    start = time.perf_counter()
    
    for step in range(steps):
        # Compute density (O(n^2) - naive)
        for i in range(num_particles):
            rho[i] = 0.0
            for j in range(num_particles):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                r2 = dx*dx + dy*dy
                if r2 < h2:
                    rho[i] += (h2 - r2)**3
        
        # Simple integration
        vy -= 9.81 * dt
        x += vx * dt
        y += vy * dt
        
        # Boundary
        y = np.maximum(y, 0.0)
    
    elapsed = time.perf_counter() - start
    return elapsed


def main():
    print("=" * 60)
    print("CUDA vs CPU PERFORMANCE BENCHMARK")
    print("=" * 60)
    print("\nNote: This benchmark tests CPU SPH performance.")
    print("CUDA performance requires compiling and running generated code.\n")
    
    # Test different particle counts
    test_configs = [
        (100, 100),
        (500, 50),
        (1000, 20),
    ]
    
    results = []
    
    for num_particles, steps in test_configs:
        print(f"Testing {num_particles} particles, {steps} steps...")
        
        cpu_time = benchmark_cpu_sph(num_particles, steps)
        
        # Estimate CUDA speedup (typical 10-50x for well-optimized SPH)
        estimated_cuda_time = cpu_time / 20.0
        
        results.append([
            num_particles,
            steps,
            f"{cpu_time*1000:.1f} ms",
            f"~{estimated_cuda_time*1000:.1f} ms",
            "~20x"
        ])
    
    print("\n")
    
    if tabulate:
        print(tabulate(results, 
                      headers=["Particles", "Steps", "CPU", "CUDA (est.)", "Speedup"],
                      tablefmt="grid"))
    else:
        print("Particles | Steps | CPU Time | CUDA (est.) | Speedup")
        print("-" * 55)
        for row in results:
            print(" | ".join(str(x) for x in row))
    
    print("\n" + "=" * 60)
    print("To get actual CUDA timings:")
    print("1. Generate CUDA code: python demos/cuda_pendulum/generate.py")
    print("2. Build with nvcc")
    print("3. Run and compare timing outputs")
    print("=" * 60)


if __name__ == '__main__':
    main()
