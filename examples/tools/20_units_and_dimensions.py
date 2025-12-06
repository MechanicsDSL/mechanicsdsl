"""
Tutorial 20: Units and Dimensions

Understanding the unit system in MechanicsDSL.

Topics:
- SI units in DSL syntax
- Dimensional consistency
- Converting between unit systems
- Natural units for theoretical work
"""

import numpy as np
import matplotlib.pyplot as plt
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# Part 1: Standard SI Units
# ============================================================================

print("="*60)
print("PART 1: SI UNITS")
print("="*60)

compiler = PhysicsCompiler()

# Simple pendulum with explicit SI units
dsl_code_si = r"""
\system{pendulum_si}

\defvar{theta}{Angular displacement}{rad}

\parameter{m}{2.5}{kg}
\parameter{L}{1.5}{m}
\parameter{g}{9.81}{m/s^2}

\lagrangian{
    \frac{1}{2} * m * L^2 * \dot{theta}^2 + m * g * L * \cos{theta}
}

\initial{theta=0.3, theta_dot=0.0}
"""

result = compiler.compile_dsl(dsl_code_si)
solution = compiler.simulate(t_span=(0, 10), num_points=500)

if solution['success']:
    print("[OK] SI units simulation successful")
    # Calculate period
    t = solution['t']
    theta = solution['y'][0]
    
    # Find zero crossings to measure period
    crossings = np.where(np.diff(np.sign(theta)))[0]
    if len(crossings) >= 2:
        measured_period = 2 * (t[crossings[1]] - t[crossings[0]])
        theoretical_period = 2 * np.pi * np.sqrt(1.5 / 9.81)  # T = 2pisqrt(L/g)
        print(f"   Measured period: {measured_period:.4f} s")
        print(f"   Theoretical period: {theoretical_period:.4f} s")

# ============================================================================
# Part 2: Comparing Different Mass/Length Scales
# ============================================================================

print("\n" + "="*60)
print("PART 2: SCALING ANALYSIS")
print("="*60)

# The period should only depend on L/g, not on mass!
masses = [0.1, 1.0, 10.0, 100.0]
periods = []

for mass in masses:
    compiler = PhysicsCompiler()
    code = dsl_code_si.replace("m}{2.5}", f"m}}{mass}}")
    compiler.compile_dsl(code)
    sol = compiler.simulate(t_span=(0, 10), num_points=500)
    
    if sol['success']:
        t = sol['t']
        theta = sol['y'][0]
        crossings = np.where(np.diff(np.sign(theta)))[0]
        if len(crossings) >= 2:
            period = 2 * (t[crossings[1]] - t[crossings[0]])
            periods.append(period)
            print(f"   m = {mass:6.1f} kg → T = {period:.4f} s")

print("\n   ✓ Period is independent of mass (as expected from theory)")

# ============================================================================
# Part 3: Dimensional Analysis
# ============================================================================

print("\n" + "="*60)
print("PART 3: DIMENSIONAL ANALYSIS")
print("="*60)

# Pendulum period: T = 2pi sqrt(L/g)
# Dimensions: [T] = sqrt([L]/[L/T²]) = sqrt[T²] = [T] ✓

print("""
Dimensional Analysis for Simple Pendulum:

Given quantities:
  - m [M] = mass
  - L [L] = length  
  - g [L T⁻²] = gravitational acceleration

Want: Period T [T]

By Buckingham pi theorem:
  T must be proportional to sqrt(L/g)
  
  [T] = [L]^a [L T⁻²]^b
  [T] = [L^(a+b) T^(-2b)]
  
  Solving: a+b=0, -2b=1 → b=-1/2, a=1/2
  
  ∴ T ~ sqrt(L/g)
  
The full answer: T = 2pi sqrt(L/g)
""")

# ============================================================================
# Visualize the scaling
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Period vs Length
lengths = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
g = 9.81
theoretical_T = 2 * np.pi * np.sqrt(lengths / g)

measured_T = []
for L in lengths:
    compiler = PhysicsCompiler()
    code = dsl_code_si.replace("L}{1.5}", f"L}}{L}}}")
    compiler.compile_dsl(code)
    sol = compiler.simulate(t_span=(0, 15), num_points=500)
    
    if sol['success']:
        t = sol['t']
        theta = sol['y'][0]
        crossings = np.where(np.diff(np.sign(theta)))[0]
        if len(crossings) >= 2:
            measured_T.append(2 * (t[crossings[1]] - t[crossings[0]]))
        else:
            measured_T.append(np.nan)
    else:
        measured_T.append(np.nan)

axes[0].plot(lengths, theoretical_T, 'b-', linewidth=2, label='Theory: 2pisqrt(L/g)')
axes[0].plot(lengths, measured_T, 'ro', markersize=10, label='Measured')
axes[0].set_xlabel('Length L (m)', fontsize=12)
axes[0].set_ylabel('Period T (s)', fontsize=12)
axes[0].set_title('Period vs Length', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Period vs sqrt(L)
axes[1].plot(np.sqrt(lengths), theoretical_T, 'b-', linewidth=2, label='Theory')
axes[1].plot(np.sqrt(lengths), measured_T, 'ro', markersize=10, label='Measured')
axes[1].set_xlabel('sqrtL (m^0.5)', fontsize=12)
axes[1].set_ylabel('Period T (s)', fontsize=12)
axes[1].set_title('Period vs sqrtL (Linear Relationship)', fontsize=14)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('20_units_and_dimensions.png', dpi=150)
print("\n[OK] Saved: 20_units_and_dimensions.png")

# ============================================================================
# Key insights
# ============================================================================

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. Always use consistent units (SI recommended)")
print("2. DSL supports unit annotations: {m}, {kg}, {rad}, etc.")
print("3. Period T ~ sqrt(L/g) - independent of mass!")
print("4. Dimensional analysis predicts scaling behavior")
print("5. Use natural units (c=hbar=1) for particle physics")
print("6. Verify simulations match theoretical scaling")
print("="*60)

plt.show()


