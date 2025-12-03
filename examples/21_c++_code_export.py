"""
Tutorial 21: C++ Code Export

This tutorial demonstrates how to generate high-performance C++ code
from your DSL simulation. This is useful for:
1. Systems that are too slow in Python
2. Embedded applications
3. Integration with other C++ projects

We'll use a double pendulum as an example of a system that benefits from speed.
"""

import os
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# Define a complex system (Double Pendulum)
# ============================================================================

dsl_code = r"""
\system{cpp_double_pendulum}

\defvar{theta1}{Angle 1}{rad}
\defvar{theta2}{Angle 2}{rad}

\parameter{m1}{1.0}{kg}
\parameter{m2}{1.0}{kg}
\parameter{l1}{1.0}{m}
\parameter{l2}{1.0}{m}
\parameter{g}{9.81}{m/s^2}

\lagrangian{
    \frac{1}{2} * (m1 + m2) * l1^2 * \dot{theta1}^2 
    + \frac{1}{2} * m2 * l2^2 * \dot{theta2}^2 
    + m2 * l1 * l2 * \dot{theta1} * \dot{theta2} * \cos{theta1 - theta2}
    + (m1 + m2) * g * l1 * \cos{theta1}
    + m2 * g * l2 * \cos{theta2}
}

\initial{theta1=2.0, theta1_dot=0.0, theta2=1.0, theta2_dot=0.0}
"""

print("Initializing compiler...")
compiler = PhysicsCompiler()

print("Compiling DSL to equations...")
result = compiler.compile_dsl(dsl_code)

if not result['success']:
    print(f"Python compilation failed: {result.get('error')}")
    exit(1)

# ============================================================================
# Generate C++ Code
# ============================================================================

print("\nGenerating C++ simulation code...")
# This will generate 'cpp_double_pendulum.cpp' and compile it to 'cpp_double_pendulum' (or .exe)
success = compiler.compile_to_cpp("cpp_double_pendulum.cpp", compile_binary=True)

if success:
    print("✅ C++ code generated and compiled successfully!")
    print("   Source file: cpp_double_pendulum.cpp")
    
    # Check for binary
    binary_name = "./cpp_double_pendulum"
    if os.name == 'nt':
        binary_name = "cpp_double_pendulum.exe"
        
    if os.path.exists(binary_name) or os.path.exists(binary_name[2:]):
        print(f"   Binary executable created: {binary_name}")
        print("\nTo run the simulation, execute:")
        print(f"   {binary_name}")
        print("   (This will generate 'cpp_double_pendulum_results.csv')")
    else:
        print("   ⚠️ Binary not found. Compilation might have failed silently or g++ is missing.")
else:
    print("❌ C++ generation failed.")
