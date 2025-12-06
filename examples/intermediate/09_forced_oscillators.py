"""
Tutorial 09: Forced Oscillators

When you drive an oscillator with an external force, interesting things happen:
- Resonance: Large amplitude at driving frequency = natural frequency
- Phase shifts: Response lags or leads the driving force
- Beats: When frequencies are close

Physics:
- Driving force: F(t) = F_0·cos(omega_d·t)
- Natural frequency: omega_0 = sqrt(k/m)
- Resonance when omega_d = omega_0
"""
import numpy as np
import matplotlib.pyplot as plt
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# System parameters
# ============================================================================

m = 1.0
k = 10.0
omega0 = np.sqrt(k / m)  # Natural frequency

print("System parameters:")
print(f"   Mass: {m} kg")
print(f"   Spring constant: {k} N/m")
print(f"   Natural frequency: omega_0 = {omega0:.3f} rad/s")
print(f"   Natural period: T_0 = {2*np.pi/omega0:.3f} s")

# ============================================================================
# Below Resonance (omega_d < omega_0)
# ============================================================================

print("\n" + "="*60)
print("BELOW RESONANCE (omega_d = 0.5·omega_0)")
print("="*60)

compiler1 = PhysicsCompiler()

omega_d1 = 0.5 * omega0
F0 = 1.0

dsl_below = r"""
\system{forced_oscillator_below}

\defvar{x}{Position}{m}

\parameter{m}{1.0}{kg}
\parameter{k}{10.0}{N/m}
\parameter{F0}{1.0}{N}
\parameter{omega_d}{%.3f}{rad/s}

\lagrangian{
    \frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2
}

\force{F0 * \cos{omega_d * t}}

\initial{x=0.0, x_dot=0.0}
""" % omega_d1

result1 = compiler1.compile_dsl(dsl_below)
if not result1['success']:
    print(f"[FAIL] Compilation failed: {result1.get('error')}")
    exit(1)

solution1 = compiler1.simulate(t_span=(0, 20), num_points=2000)

# ============================================================================
# At Resonance (omega_d = omega_0)
# ============================================================================

print("\n" + "="*60)
print("AT RESONANCE (omega_d = omega_0)")
print("="*60)

compiler2 = PhysicsCompiler()

omega_d2 = omega0

dsl_resonance = r"""
\system{forced_oscillator_resonance}

\defvar{x}{Position}{m}

\parameter{m}{1.0}{kg}
\parameter{k}{10.0}{N/m}
\parameter{F0}{1.0}{N}
\parameter{omega_d}{%.3f}{rad/s}

\lagrangian{
    \frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2
}

\force{F0 * \cos{omega_d * t}}

\initial{x=0.0, x_dot=0.0}
""" % omega_d2

result2 = compiler2.compile_dsl(dsl_resonance)
if not result2['success']:
    print(f"[FAIL] Compilation failed: {result2.get('error')}")
    exit(1)

solution2 = compiler2.simulate(t_span=(0, 20), num_points=2000)

# ============================================================================
# Above Resonance (omega_d > omega_0)
# ============================================================================

print("\n" + "="*60)
print("ABOVE RESONANCE (omega_d = 2·omega_0)")
print("="*60)

compiler3 = PhysicsCompiler()

omega_d3 = 2.0 * omega0

dsl_above = r"""
\system{forced_oscillator_above}

\defvar{x}{Position}{m}

\parameter{m}{1.0}{kg}
\parameter{k}{10.0}{N/m}
\parameter{F0}{1.0}{N}
\parameter{omega_d}{%.3f}{rad/s}

\lagrangian{
    \frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2
}

\force{F0 * \cos{omega_d * t}}

\initial{x=0.0, x_dot=0.0}
""" % omega_d3

result3 = compiler3.compile_dsl(dsl_above)
if not result3['success']:
    print(f"[FAIL] Compilation failed: {result3.get('error')}")
    exit(1)

solution3 = compiler3.simulate(t_span=(0, 20), num_points=2000)

# ============================================================================
# Extract results
# ============================================================================

t1 = solution1['t']
x1 = solution1['y'][0]

t2 = solution2['t']
x2 = solution2['y'][0]

t3 = solution3['t']
x3 = solution3['y'][0]

# Driving forces
F1 = F0 * np.cos(omega_d1 * t1)
F2 = F0 * np.cos(omega_d2 * t2)
F3 = F0 * np.cos(omega_d3 * t3)

# ============================================================================
# Plot comparison
# ============================================================================

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Below resonance
axes[0, 0].plot(t1, x1, 'b-', linewidth=2, label='Response')
axes[0, 0].plot(t1, F1, 'r--', linewidth=1, alpha=0.5, label='Driving force (scaled)')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Position (m)')
axes[0, 0].set_title(f'Below Resonance (omega_d = {omega_d1:.2f} rad/s)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t1[-500:], x1[-500:], 'b-', linewidth=2)
axes[0, 1].plot(t1[-500:], F1[-500:], 'r--', linewidth=1, alpha=0.5)
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Amplitude')
axes[0, 1].set_title('Steady State (last 25%)')
axes[0, 1].grid(True, alpha=0.3)

# At resonance
axes[1, 0].plot(t2, x2, 'b-', linewidth=2, label='Response')
axes[1, 0].plot(t2, F2, 'r--', linewidth=1, alpha=0.5, label='Driving force (scaled)')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Position (m)')
axes[1, 0].set_title(f'At Resonance (omega_d = {omega_d2:.2f} rad/s)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(t2[-500:], x2[-500:], 'b-', linewidth=2)
axes[1, 1].plot(t2[-500:], F2[-500:], 'r--', linewidth=1, alpha=0.5)
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Amplitude')
axes[1, 1].set_title('Steady State (last 25%)')
axes[1, 1].grid(True, alpha=0.3)

# Above resonance
axes[2, 0].plot(t3, x3, 'b-', linewidth=2, label='Response')
axes[2, 0].plot(t3, F3, 'r--', linewidth=1, alpha=0.5, label='Driving force (scaled)')
axes[2, 0].set_xlabel('Time (s)')
axes[2, 0].set_ylabel('Position (m)')
axes[2, 0].set_title(f'Above Resonance (omega_d = {omega_d3:.2f} rad/s)')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].plot(t3[-500:], x3[-500:], 'b-', linewidth=2)
axes[2, 1].plot(t3[-500:], F3[-500:], 'r--', linewidth=1, alpha=0.5)
axes[2, 1].set_xlabel('Time (s)')
axes[2, 1].set_ylabel('Amplitude')
axes[2, 1].set_title('Steady State (last 25%)')
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('09_forced_oscillators.png', dpi=150)
print("\n[OK] Saved: 09_forced_oscillators.png")

# ============================================================================
# Amplitude comparison
# ============================================================================

# Find steady-state amplitudes (last 25% of simulation)
amp1 = np.max(np.abs(x1[-len(x1)//4:]))
amp2 = np.max(np.abs(x2[-len(x2)//4:]))
amp3 = np.max(np.abs(x3[-len(x3)//4:]))

print("\nSteady-state amplitudes:")
print(f"   Below resonance: {amp1:.4f} m")
print(f"   At resonance: {amp2:.4f} m ({amp2/amp1:.1f}x larger!)")
print(f"   Above resonance: {amp3:.4f} m")

# Plot amplitude vs frequency
omega_d_values = [omega_d1, omega_d2, omega_d3]
amplitudes = [amp1, amp2, amp3]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(omega_d_values, amplitudes, 'bo-', linewidth=2, markersize=10)
ax.axvline(omega0, color='r', linestyle='--', label=f'Natural frequency omega_0 = {omega0:.2f}')
ax.set_xlabel('Driving Frequency (rad/s)', fontsize=12)
ax.set_ylabel('Steady-State Amplitude (m)', fontsize=12)
ax.set_title('Resonance Curve', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('09_resonance_curve.png', dpi=150)
print("[OK] Saved: 09_resonance_curve.png")

# ============================================================================
# Key insights
# ============================================================================

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. RESONANCE: Maximum amplitude when omega_d = omega_0")
print("2. Below resonance: Response in phase with driving force")
print("3. Above resonance: Response out of phase (180 deg shift)")
print("4. At resonance: Amplitude grows linearly with time (no damping)")
print("5. Use \force{} command for time-dependent forces")
print("6. Time-dependent forces use 't' as the time variable")
print("="*60)

plt.show()


