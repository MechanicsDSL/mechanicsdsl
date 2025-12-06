"""
Tutorial 19: Performance Tuning

Optimizing simulations for speed and accuracy.

Topics:
- Solver selection (RK45, LSODA, DOP853, etc.)
- Tolerance tuning (rtol, atol)
- Stiffness detection
- Profiling and benchmarking
- When to use C++ code generation
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# Define a system to benchmark
# ============================================================================

dsl_code = r"""
\system{double_pendulum}

\defvar{theta1}{Angle of first pendulum}{rad}
\defvar{theta2}{Angle of second pendulum}{rad}

\parameter{m1}{1.0}{kg}
\parameter{m2}{1.0}{kg}
\parameter{L1}{1.0}{m}
\parameter{L2}{1.0}{m}
\parameter{g}{9.81}{m/s^2}

\lagrangian{
    \frac{1}{2} * (m1 + m2) * L1^2 * \dot{theta1}^2 +
    \frac{1}{2} * m2 * L2^2 * \dot{theta2}^2 +
    m2 * L1 * L2 * \dot{theta1} * \dot{theta2} * \cos{theta1 - theta2} +
    (m1 + m2) * g * L1 * \cos{theta1} +
    m2 * g * L2 * \cos{theta2}
}

\initial{theta1=2.5, theta2=2.5, theta1_dot=0.0, theta2_dot=0.0}
"""

# ============================================================================
# Benchmark different solver methods
# ============================================================================

print("="*60)
print("PERFORMANCE BENCHMARKING")
print("="*60)

methods = ['RK45', 'RK23', 'DOP853', 'LSODA', 'BDF', 'Radau']
results = {}

for method in methods:
    compiler = PhysicsCompiler()
    compiler.compile_dsl(dsl_code)
    
    start_time = time.time()
    solution = compiler.simulate(
        t_span=(0, 20), 
        num_points=2000,
    )
    elapsed = time.time() - start_time
    
    if solution['success']:
        nfev = solution.get('nfev', 'N/A')
        results[method] = {
            'time': elapsed,
            'nfev': nfev,
            'success': True,
            'y_final': solution['y'][:, -1]
        }
        print(f"{method:10s}: {elapsed:.4f}s, {nfev} evaluations")
    else:
        results[method] = {'success': False}
        print(f"{method:10s}: FAILED")

# ============================================================================
# Benchmark different tolerances
# ============================================================================

print("\n" + "="*60)
print("TOLERANCE STUDY (RK45)")
print("="*60)

tolerances = [1e-3, 1e-6, 1e-9, 1e-12]
tol_results = {}

for tol in tolerances:
    compiler = PhysicsCompiler()
    compiler.compile_dsl(dsl_code)
    
    start_time = time.time()
    solution = compiler.simulate(
        t_span=(0, 20), 
        num_points=2000,
        rtol=tol,
        atol=tol
    )
    elapsed = time.time() - start_time
    
    if solution['success']:
        nfev = solution.get('nfev', 'N/A')
        tol_results[tol] = {
            'time': elapsed,
            'nfev': nfev,
            'y_final': solution['y'][:, -1]
        }
        print(f"tol={tol:.0e}: {elapsed:.4f}s, {nfev} evaluations")

# ============================================================================
# Visualize results
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Solver comparison
successful_methods = [m for m in methods if results[m]['success']]
times = [results[m]['time'] for m in successful_methods]
nfevs = [results[m]['nfev'] if results[m]['nfev'] != 'N/A' else 0 
         for m in successful_methods]

axes[0, 0].bar(successful_methods, times, color='steelblue')
axes[0, 0].set_ylabel('Time (s)')
axes[0, 0].set_title('Solver Comparison: Execution Time')
axes[0, 0].tick_params(axis='x', rotation=45)

axes[0, 1].bar(successful_methods, nfevs, color='coral')
axes[0, 1].set_ylabel('Function Evaluations')
axes[0, 1].set_title('Solver Comparison: Efficiency')
axes[0, 1].tick_params(axis='x', rotation=45)

# Tolerance study
tols = list(tol_results.keys())
tol_times = [tol_results[t]['time'] for t in tols]
tol_nfevs = [tol_results[t]['nfev'] if tol_results[t]['nfev'] != 'N/A' else 0 
             for t in tols]

axes[1, 0].semilogx(tols, tol_times, 'bo-', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('Tolerance')
axes[1, 0].set_ylabel('Time (s)')
axes[1, 0].set_title('Tolerance vs Execution Time')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].invert_xaxis()

axes[1, 1].semilogx(tols, tol_nfevs, 'ro-', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('Tolerance')
axes[1, 1].set_ylabel('Function Evaluations')
axes[1, 1].set_title('Tolerance vs Evaluations')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].invert_xaxis()

plt.tight_layout()
plt.savefig('19_performance_tuning.png', dpi=150)
print("\n[OK] Saved: 19_performance_tuning.png")

# ============================================================================
# Key insights
# ============================================================================

print("\n" + "="*60)
print("PERFORMANCE TIPS:")
print("="*60)
print("1. RK45: Good default, efficient for most problems")
print("2. DOP853: Higher order, better for smooth problems")
print("3. LSODA: Auto-switches between stiff/non-stiff")
print("4. BDF/Radau: Best for stiff problems (rare in mechanics)")
print("5. Looser tolerances = faster but less accurate")
print("6. For production: use C++ code generation")
print("7. Profile before optimizing!")
print("="*60)

plt.show()


