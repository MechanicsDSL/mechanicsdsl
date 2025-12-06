"""
Tutorial 13: Hamiltonian Formulation

The Hamiltonian approach is an alternative to Lagrangian mechanics.
Instead of second-order equations, we get first-order equations.

Advantages:
- Symplectic structure (better for numerical integration)
- Direct access to momenta
- Natural for quantum mechanics
- Phase space is more natural

Physics:
- Hamiltonian: H = Σ(pᵢ·q̇ᵢ) - L
- Hamilton's equations: q̇ = ∂H/∂p, ṗ = -∂H/∂q
- For conservative systems: H = T + V
"""

import numpy as np
import matplotlib.pyplot as plt
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# Method 1: Let MechanicsDSL derive Hamiltonian from Lagrangian
# ============================================================================

print("="*60)
print("METHOD 1: Automatic Hamiltonian from Lagrangian")
print("="*60)

compiler1 = PhysicsCompiler()

dsl_lagrangian = """
\\system{harmonic_oscillator_hamiltonian}

\\defvar{x}{Position}{m}

\\parameter{m}{1.0}{kg}
\\parameter{k}{10.0}{N/m}

\\lagrangian{\\frac{1}{2} * m * \\dot{x}^2 - \\frac{1}{2} * k * x^2}

\\initial{x=1.0, x_dot=0.0}
"""

# Compile with Hamiltonian formulation
result1 = compiler1.compile_dsl(dsl_lagrangian, use_hamiltonian=True)

if not result1['success']:
    print(f"❌ Compilation failed: {result1.get('error')}")
    exit(1)

print("✅ Compilation successful (Hamiltonian formulation)!")

solution1 = compiler1.simulate(t_span=(0, 10), num_points=500)

# ============================================================================
# Method 2: Explicitly define Hamiltonian
# ============================================================================

print("\n" + "="*60)
print("METHOD 2: Explicit Hamiltonian Definition")
print("="*60)

compiler2 = PhysicsCompiler()

dsl_hamiltonian = """
\\system{explicit_hamiltonian}

\\defvar{x}{Position}{m}

\\parameter{m}{1.0}{kg}
\\parameter{k}{10.0}{N/m}

\\hamiltonian{\\frac{1}{2} * p_x^2 / m + \\frac{1}{2} * k * x^2}

\\initial{x=1.0, p_x=0.0}
"""

result2 = compiler2.compile_dsl(dsl_hamiltonian)

if not result2['success']:
    print(f"❌ Compilation failed: {result2.get('error')}")
    exit(1)

print("✅ Compilation successful (explicit Hamiltonian)!")

solution2 = compiler2.simulate(t_span=(0, 10), num_points=500)

# ============================================================================
# Extract results
# ============================================================================

t1 = solution1['t']
x1 = solution1['y'][0]
p1 = solution1['y'][1]  # Momentum

t2 = solution2['t']
x2 = solution2['y'][0]
p2 = solution2['y'][1]  # Momentum

# ============================================================================
# Compare results
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Position comparison
axes[0, 0].plot(t1, x1, 'b-', linewidth=2, label='From Lagrangian', alpha=0.7)
axes[0, 0].plot(t2, x2, 'r--', linewidth=2, label='Explicit Hamiltonian', alpha=0.7)
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Position (m)')
axes[0, 0].set_title('Position: Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Momentum
axes[0, 1].plot(t1, p1, 'b-', linewidth=2, label='From Lagrangian', alpha=0.7)
axes[0, 1].plot(t2, p2, 'r--', linewidth=2, label='Explicit Hamiltonian', alpha=0.7)
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Momentum (kg·m/s)')
axes[0, 1].set_title('Momentum vs Time')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Phase space (position vs momentum)
axes[1, 0].plot(x1, p1, 'b-', linewidth=1.5, label='From Lagrangian', alpha=0.7)
axes[1, 0].plot(x2, p2, 'r--', linewidth=1.5, label='Explicit Hamiltonian', alpha=0.7)
axes[1, 0].set_xlabel('Position (m)')
axes[1, 0].set_ylabel('Momentum (kg·m/s)')
axes[1, 0].set_title('Phase Space (x, p)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axis('equal')

# Energy (Hamiltonian should be conserved)
m, k = 1.0, 10.0
H1 = p1**2 / (2*m) + 0.5 * k * x1**2
H2 = p2**2 / (2*m) + 0.5 * k * x2**2

axes[1, 1].plot(t1, H1, 'b-', linewidth=2, label='From Lagrangian', alpha=0.7)
axes[1, 1].plot(t2, H2, 'r--', linewidth=2, label='Explicit Hamiltonian', alpha=0.7)
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Hamiltonian (J)')
axes[1, 1].set_title('Hamiltonian Conservation')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('13_hamiltonian_comparison.png', dpi=150)
print("\n✅ Saved: 13_hamiltonian_comparison.png")

# ============================================================================
# Key insights
# ============================================================================

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. Hamiltonian = T + V for conservative systems")
print("2. State variables are (q, p) not (q, q̇)")
print("3. Hamilton's equations are first-order (not second)")
print("4. Phase space is natural: (position, momentum)")
print("5. Use use_hamiltonian=True or define \\hamiltonian{}")
print("6. Better numerical properties (symplectic)")
print("="*60)

plt.show()
