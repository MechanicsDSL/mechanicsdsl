#!/usr/bin/env python3
"""
Generate figures and benchmarks for CPC paper submission.
"""
import sys
import os
import time

# Ensure we're using the local source
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

# Create figures directory
FIG_DIR = os.path.join(os.path.dirname(__file__), 'paper_figures')
os.makedirs(FIG_DIR, exist_ok=True)

print("="*60)
print("GENERATING CPC PAPER FIGURES AND BENCHMARKS")
print("="*60)

#============================================================================
# FIGURE 1: Double Pendulum Phase Space
#============================================================================
print("\n[1/5] Generating double pendulum phase space...")

from mechanics_dsl import PhysicsCompiler

double_pendulum_code = r"""
\system{double_pendulum}
\defvar{theta1}{Angle}{rad}
\defvar{theta2}{Angle}{rad}
\parameter{m1}{1.0}{kg}
\parameter{m2}{1.0}{kg}
\parameter{l1}{1.0}{m}
\parameter{l2}{1.0}{m}
\parameter{g}{9.81}{m/s^2}

\lagrangian{
    0.5*(m1+m2)*l1^2*\dot{theta1}^2
    + 0.5*m2*l2^2*\dot{theta2}^2
    + m2*l1*l2*\dot{theta1}*\dot{theta2}*cos(theta1-theta2)
    + (m1+m2)*g*l1*cos(theta1)
    + m2*g*l2*cos(theta2)
}
"""

compiler = PhysicsCompiler()
result = compiler.compile_dsl(double_pendulum_code)
compiler.simulator.set_initial_conditions({
    'theta1': np.pi/2, 'theta1_dot': 0,
    'theta2': np.pi/2, 'theta2_dot': 0
})
solution = compiler.simulate(t_span=(0, 20), num_points=2000)

if solution['success']:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Trajectory
    t = solution['t']
    theta1 = solution['y'][0, :]
    theta2 = solution['y'][2, :]
    theta1_dot = solution['y'][1, :]
    theta2_dot = solution['y'][3, :]
    
    # Phase space theta1
    axes[0].plot(theta1, theta1_dot, 'b-', linewidth=0.3, alpha=0.7)
    axes[0].set_xlabel(r'$\theta_1$ (rad)', fontsize=12)
    axes[0].set_ylabel(r'$\dot{\theta}_1$ (rad/s)', fontsize=12)
    axes[0].set_title('Phase Space: Mass 1', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Phase space theta2
    axes[1].plot(theta2, theta2_dot, 'r-', linewidth=0.3, alpha=0.7)
    axes[1].set_xlabel(r'$\theta_2$ (rad)', fontsize=12)
    axes[1].set_ylabel(r'$\dot{\theta}_2$ (rad/s)', fontsize=12)
    axes[1].set_title('Phase Space: Mass 2', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig1_double_pendulum_phase.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'fig1_double_pendulum_phase.pdf'), bbox_inches='tight')
    plt.close()
    print("  -> Saved fig1_double_pendulum_phase.png/.pdf")
else:
    print("  -> ERROR: Simulation failed")

#============================================================================
# FIGURE 2: Energy Conservation
#============================================================================
print("\n[2/5] Generating energy conservation plot...")

# Simple pendulum for energy test
pendulum_code = r"""
\system{pendulum}
\defvar{theta}{Angle}{rad}
\parameter{m}{1.0}{kg}
\parameter{l}{1.0}{m}
\parameter{g}{9.81}{m/s^2}

\lagrangian{0.5*m*l^2*\dot{theta}^2 + m*g*l*cos(theta)}
"""

compiler2 = PhysicsCompiler()
compiler2.compile_dsl(pendulum_code)
compiler2.simulator.set_initial_conditions({'theta': 0.5, 'theta_dot': 0})
solution2 = compiler2.simulate(t_span=(0, 50), num_points=5000)

if solution2['success']:
    t = solution2['t']
    theta = solution2['y'][0, :]
    theta_dot = solution2['y'][1, :]
    
    # Energy calculation
    m, l, g = 1.0, 1.0, 9.81
    KE = 0.5 * m * l**2 * theta_dot**2
    PE = -m * g * l * np.cos(theta)
    E_total = KE + PE
    E_initial = E_total[0]
    E_error = (E_total - E_initial) / np.abs(E_initial)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    axes[0].plot(t, KE, 'b-', label='Kinetic Energy', linewidth=1)
    axes[0].plot(t, PE, 'r-', label='Potential Energy', linewidth=1)
    axes[0].plot(t, E_total, 'k--', label='Total Energy', linewidth=1.5)
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('Energy (J)', fontsize=12)
    axes[0].set_title('Energy Components', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t, E_error, 'g-', linewidth=1)
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel(r'$(E - E_0) / |E_0|$', fontsize=12)
    axes[1].set_title(f'Relative Energy Error (max: {np.max(np.abs(E_error)):.2e})', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig2_energy_conservation.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'fig2_energy_conservation.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  -> Saved fig2_energy_conservation.png/.pdf (max error: {np.max(np.abs(E_error)):.2e})")

#============================================================================
# FIGURE 3: Architecture Diagram
#============================================================================
print("\n[3/5] Generating architecture diagram...")

fig, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

# Box parameters
box_style = dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='navy', linewidth=2)
arrow_style = dict(arrowstyle='->', color='navy', linewidth=2)

# Pipeline boxes
boxes = [
    (1, 3, 'DSL\nSource', 'lavender'),
    (3, 3, 'Lexer\n$\\mathcal{L}$', 'lightcyan'),
    (5, 3, 'Parser\n$\\mathcal{P}$', 'lightcyan'),
    (7, 3, 'Symbolic\n$\\mathcal{S}$', 'lightyellow'),
    (9, 3, 'Derivation\n$\\mathcal{D}_{EL}$', 'lightyellow'),
    (11, 3, 'Extraction\n$\\mathcal{A}$', 'lightgreen'),
    (13, 3, 'Numerical\n$\\mathcal{N}$', 'lightgreen'),
]

for x, y, label, color in boxes:
    rect = mpatches.FancyBboxPatch((x-0.8, y-0.5), 1.6, 1.2,
                                    boxstyle='round,pad=0.1',
                                    facecolor=color, edgecolor='navy', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')

# Arrows between boxes
for i in range(len(boxes)-1):
    x1 = boxes[i][0] + 0.8
    x2 = boxes[i+1][0] - 0.8
    ax.annotate('', xy=(x2, 3), xytext=(x1, 3),
                arrowprops=dict(arrowstyle='->', color='navy', lw=2))

# Integration and output
rect = mpatches.FancyBboxPatch((10-0.8, 1-0.5), 1.6, 1.2,
                                boxstyle='round,pad=0.1',
                                facecolor='lightsalmon', edgecolor='navy', linewidth=1.5)
ax.add_patch(rect)
ax.text(10, 1, 'Integration\n$\\mathcal{I}$', ha='center', va='center', fontsize=9, fontweight='bold')

rect = mpatches.FancyBboxPatch((13-0.8, 1-0.5), 1.6, 1.2,
                                boxstyle='round,pad=0.1',
                                facecolor='lightpink', edgecolor='navy', linewidth=1.5)
ax.add_patch(rect)
ax.text(13, 1, 'Trajectory\n$\\mathbf{y}(t)$', ha='center', va='center', fontsize=9, fontweight='bold')

# Connecting arrows
ax.annotate('', xy=(13, 2.5), xytext=(13, 1.5),
            arrowprops=dict(arrowstyle='->', color='navy', lw=2))
ax.annotate('', xy=(11.2, 1), xytext=(10.8, 1),
            arrowprops=dict(arrowstyle='->', color='navy', lw=2))

# Title
ax.text(7, 5.5, 'MechanicsDSL Compilation Pipeline', ha='center', va='center', 
        fontsize=16, fontweight='bold')
ax.text(7, 5, '$\\mathcal{C} = \\mathcal{I} \\circ \\mathcal{N} \\circ \\mathcal{A} \\circ \\mathcal{D}_{EL} \\circ \\mathcal{S} \\circ \\mathcal{P} \\circ \\mathcal{L}$',
        ha='center', va='center', fontsize=12)

# Legend
legend_items = [
    (2, 0.3, 'Parsing', 'lightcyan'),
    (5, 0.3, 'Symbolic', 'lightyellow'),
    (8, 0.3, 'Numerical', 'lightgreen'),
    (11, 0.3, 'Output', 'lightsalmon'),
]
for x, y, label, color in legend_items:
    rect = mpatches.Rectangle((x-0.5, y-0.15), 1, 0.3, facecolor=color, edgecolor='navy')
    ax.add_patch(rect)
    ax.text(x+0.7, y, label, ha='left', va='center', fontsize=9)

plt.savefig(os.path.join(FIG_DIR, 'fig3_architecture.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'fig3_architecture.pdf'), bbox_inches='tight')
plt.close()
print("  -> Saved fig3_architecture.png/.pdf")

#============================================================================
# BENCHMARKS
#============================================================================
print("\n[4/5] Running benchmarks...")

benchmark_results = {}

# Benchmark 1: Compilation time
systems = [
    ('Simple Pendulum (1 DOF)', pendulum_code),
    ('Double Pendulum (2 DOF)', double_pendulum_code),
]

print("\n  Compilation Time Benchmarks:")
print("  " + "-"*50)

for name, code in systems:
    times = []
    for _ in range(5):
        start = time.perf_counter()
        c = PhysicsCompiler()
        c.compile_dsl(code)
        times.append(time.perf_counter() - start)
    avg_time = np.mean(times)
    benchmark_results[f'compile_{name}'] = avg_time
    print(f"  {name}: {avg_time*1000:.1f} ms (avg of 5)")

# Benchmark 2: Simulation time
print("\n  Simulation Time Benchmarks (10s, 1000 points):")
print("  " + "-"*50)

for name, code in systems:
    c = PhysicsCompiler()
    c.compile_dsl(code)
    c.simulator.set_initial_conditions({'theta': 0.5, 'theta_dot': 0} if 'theta1' not in code 
                                       else {'theta1': 0.5, 'theta1_dot': 0, 'theta2': 0.5, 'theta2_dot': 0})
    
    times = []
    for _ in range(5):
        start = time.perf_counter()
        c.simulate(t_span=(0, 10), num_points=1000)
        times.append(time.perf_counter() - start)
    avg_time = np.mean(times)
    benchmark_results[f'simulate_{name}'] = avg_time
    print(f"  {name}: {avg_time*1000:.1f} ms (avg of 5)")

# Benchmark 3: Code generation (if available)
print("\n  Code Generation Benchmarks:")
print("  " + "-"*50)

try:
    from mechanics_dsl.codegen import CppGenerator, PythonGenerator, JuliaGenerator
    import tempfile
    import sympy as sp
    
    theta = sp.Symbol('theta', real=True)
    g = sp.Symbol('g', positive=True)
    l = sp.Symbol('l', positive=True)
    
    pendulum_data = {
        'system_name': 'pendulum',
        'coordinates': ['theta'],
        'parameters': {'g': 9.81, 'l': 1.0},
        'initial_conditions': {'theta': 0.3, 'theta_dot': 0.0},
        'equations': {'theta_ddot': -g/l * sp.sin(theta)}
    }
    
    generators = [
        ('C++', CppGenerator, '.cpp'),
        ('Python', PythonGenerator, '.py'),
        ('Julia', JuliaGenerator, '.jl'),
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for gen_name, GenClass, ext in generators:
            times = []
            for _ in range(5):
                start = time.perf_counter()
                gen = GenClass(**pendulum_data)
                gen.generate(os.path.join(tmpdir, f'test{ext}'))
                times.append(time.perf_counter() - start)
            avg_time = np.mean(times)
            benchmark_results[f'codegen_{gen_name}'] = avg_time
            print(f"  {gen_name}: {avg_time*1000:.1f} ms (avg of 5)")

except Exception as e:
    print(f"  Code generation benchmarks skipped: {e}")

#============================================================================
# VALIDATION
#============================================================================
print("\n[5/5] Running validation tests...")

# Harmonic oscillator period test
print("\n  Validation Tests:")
print("  " + "-"*50)

ho_code = r"""
\system{harmonic_oscillator}
\defvar{x}{Position}{m}
\parameter{m}{1.0}{kg}
\parameter{k}{4.0}{N/m}

\lagrangian{0.5*m*\dot{x}^2 - 0.5*k*x^2}
"""

c = PhysicsCompiler()
c.compile_dsl(ho_code)
c.simulator.set_initial_conditions({'x': 1.0, 'x_dot': 0})
sol = c.simulate(t_span=(0, 20), num_points=10000)

if sol['success']:
    x = sol['y'][0, :]
    t = sol['t']
    
    # Find zero crossings (going positive)
    crossings = []
    for i in range(1, len(x)):
        if x[i-1] < 0 and x[i] >= 0:
            # Linear interpolation for precise crossing
            t_cross = t[i-1] + (0 - x[i-1]) * (t[i] - t[i-1]) / (x[i] - x[i-1])
            crossings.append(t_cross)
    
    if len(crossings) >= 2:
        measured_period = np.mean(np.diff(crossings))
        m, k = 1.0, 4.0
        expected_period = 2 * np.pi * np.sqrt(m / k)
        period_error = abs(measured_period - expected_period) / expected_period
        print(f"  Harmonic oscillator period error: {period_error:.2e}")
        benchmark_results['validation_ho_period_error'] = period_error

# Energy conservation over long time
c2 = PhysicsCompiler()
c2.compile_dsl(pendulum_code)
c2.simulator.set_initial_conditions({'theta': 0.5, 'theta_dot': 0})
sol2 = c2.simulate(t_span=(0, 100), num_points=10000)

if sol2['success']:
    theta = sol2['y'][0, :]
    theta_dot = sol2['y'][1, :]
    m, l, g = 1.0, 1.0, 9.81
    E = 0.5 * m * l**2 * theta_dot**2 - m * g * l * np.cos(theta)
    E_error = np.max(np.abs((E - E[0]) / np.abs(E[0])))
    print(f"  Energy conservation (100s): max relative error = {E_error:.2e}")
    benchmark_results['validation_energy_error'] = E_error

#============================================================================
# SAVE BENCHMARK RESULTS
#============================================================================
print("\n" + "="*60)
print("BENCHMARK RESULTS SUMMARY")
print("="*60)

with open(os.path.join(FIG_DIR, 'benchmarks.txt'), 'w') as f:
    f.write("MechanicsDSL Benchmark Results\n")
    f.write("="*50 + "\n\n")
    for key, value in benchmark_results.items():
        if 'error' in key:
            line = f"{key}: {value:.2e}\n"
        else:
            line = f"{key}: {value*1000:.2f} ms\n"
        f.write(line)
        print(line.strip())

print("\n" + "="*60)
print(f"All outputs saved to: {FIG_DIR}")
print("="*60)
