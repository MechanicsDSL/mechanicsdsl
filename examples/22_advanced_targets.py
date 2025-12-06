"""
Tutorial 22: Advanced C++ Compilation Targets

This tutorial demonstrates the "Physics Compiler" capabilities:
1. Standard C++ (Benchmarking)
2. OpenMP (Parallel Parameter Sweep)
3. Raylib (Interactive Visualization)
4. Arduino (Embedded Code)
"""

import os
from mechanics_dsl import PhysicsCompiler

# Define a Double Pendulum (Good candidate for all targets)
dsl_code = r"""
\system{multi_target_pendulum}
\defvar{t1}{Angle 1}{rad} \defvar{t2}{Angle 2}{rad}
\parameter{m}{1.0}{kg} \parameter{l}{1.0}{m} \parameter{g}{9.81}{m/s^2}

\lagrangian{
    0.5*m*l^2*(2*\dot{t1}^2 + \dot{t2}^2 + 2*\dot{t1}*\dot{t2}*\cos{t1-t2}) 
    + m*g*l*(2*\cos{t1} + \cos{t2})
}
\initial{t1=2.0, t1_dot=0.0, t2=1.0, t2_dot=0.0}
"""

print("Initializing Compiler...")
compiler = PhysicsCompiler()
if not compiler.compile_dsl(dsl_code)['success']: exit(1)

# --- 1. Standard C++ ---
print("\n--- Target 1: Standard C++ ---")
compiler.compile_to_cpp("sim_standard.cpp", target="standard")

# --- 2. OpenMP Sweep ---
print("\n--- Target 2: OpenMP Parallel Sweep ---")
# This generates a program that sweeps initial t1 from 0 to 3.14
compiler.compile_to_cpp("sim_openmp.cpp", target="openmp")

# --- 3. Raylib Visualization ---
print("\n--- Target 3: Raylib Interactive Game ---")
# Note: Requires raylib installed (sudo apt install libraylib-dev)
# If not installed, it will generate the .cpp but fail compilation gracefully
compiler.compile_to_cpp("sim_game.cpp", target="raylib")

# --- 4. Arduino Sketch ---
print("\n--- Target 4: Arduino Embedded ---")
compiler.compile_to_cpp("sim_arduino.ino", target="arduino")

print("\n\nDONE! Artifacts generated:")
print("1. ./sim_standard (Executable)")
print("2. ./sim_openmp   (Executable - Run with './sim_openmp')")
print("3. ./sim_game     (Executable - Realtime visualization)")
print("4. ./sim_arduino.ino (Upload to ESP32/Uno)")


