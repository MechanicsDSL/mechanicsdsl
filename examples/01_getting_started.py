import numpy as np
import matplotlib.pyplot as plt
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# Step 1: Create a compiler instance
# ============================================================================

compiler = PhysicsCompiler()

# ============================================================================
# Step 2: Write DSL code
# ============================================================================

# DSL (Domain-Specific Language) uses LaTeX-inspired syntax
# Let's define a free particle (no forces) moving in 1D

dsl_code = r"""
\system{free_particle}

\defvar{x}{Position}{m}

\parameter{m}{1.0}{kg}

\lagrangian{\frac{1}{2} * m * \dot{x}^2}

\initial{x=0.0, x_dot=1.0}
"""

# ============================================================================
# Step 3: Compile the DSL code
# ============================================================================

print("Compiling DSL code...")
result = compiler.compile_dsl(dsl_code)

if result['success']:
    print("✅ Compilation successful!")
    print(f"   Compilation time: {result.get('compilation_time', 0):.4f} seconds")
else:
    print("❌ Compilation failed!")
    print(f"   Error: {result.get('error', 'Unknown error')}")
    # Removed exit(1)

# ============================================================================
# Step 4: Run a simulation
# ============================================================================

print("\nRunning simulation...")
solution = compiler.simulate(t_span=(0, 10), num_points=100)

if solution['success']:
    print("✅ Simulation successful!")
    print(f"   Time points: {len(solution['t'])}")
    print(f"   Function evaluations: {solution.get('nfev', 'N/A')}")
else:
    print("❌ Simulation failed!")
    # Removed exit(1)

# ============================================================================
# Step 5: Extract and plot results
# ============================================================================

# Extract time and position
if solution['success'] and len(solution['y']) > 0:
    t = solution['t']
    x = solution['y'][0]  # First coordinate is position
    x_dot = solution['y'][1]  # Second coordinate is velocity

    # Create a simple plot
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, x, 'b-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Free Particle: Position vs Time')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(t, x_dot, 'r-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Free Particle: Velocity vs Time')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('01_free_particle.png', dpi=150)
    print("\n✅ Plot saved as '01_free_particle.png'")
    print("\nSince this is a free particle (no forces), velocity is constant!")
    print(f"   Initial velocity: {x_dot[0]:.2f} m/s")
    print(f"   Final velocity: {x_dot[-1]:.2f} m/s")
else:
    print("No data to plot for the free particle due to simulation failure.")

# ============================================================================
# What we learned:
# ============================================================================

print("\n" + "="*60)
print("WHAT WE LEARNED:")
print("="*60)
print("1. DSL syntax uses backslashes: \\system{}, \\defvar{}, \\lagrangian{}")
print("2. Variables are defined with \\defvar{name}{description}{units}")
print("3. Parameters are defined with \\parameter{name}{value}{units}")
print("4. Lagrangians use LaTeX math: \\frac{1}{2}, \\dot{x} for derivatives")
print("5. Initial conditions use \\initial{var=value, var_dot=value}")
print("6. compile_dsl() compiles the system")
print("7. simulate() runs the numerical integration")
print("8. Results are in solution['t'] (time) and solution['y'] (state)")
print("="*60)

plt.show()
