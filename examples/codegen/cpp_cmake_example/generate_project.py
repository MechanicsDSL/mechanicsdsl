"""
C++ CMake Example Project
=========================

This example demonstrates generating a complete C++ project with CMake
from a MechanicsDSL physics simulation.

Usage:
    python generate_project.py
    cd output
    mkdir build && cd build
    cmake .. && make -j$(nproc)
    ./double_pendulum
"""

import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from mechanics_dsl import PhysicsCompiler


def main():
    """Generate a complete C++ project for a double pendulum."""
    
    # Define a double pendulum system
    double_pendulum_code = r"""
    \system{double_pendulum}
    
    \defvar{theta1}{Angle}{rad}
    \defvar{theta2}{Angle}{rad}
    
    \parameter{m1}{1.0}{kg}
    \parameter{m2}{1.0}{kg}
    \parameter{l1}{1.0}{m}
    \parameter{l2}{1.0}{m}
    \parameter{g}{9.81}{m/s^2}
    
    # Double pendulum Lagrangian
    \lagrangian{
        0.5*(m1+m2)*l1^2*\dot{theta1}^2 
        + 0.5*m2*l2^2*\dot{theta2}^2 
        + m2*l1*l2*\dot{theta1}*\dot{theta2}*\cos{theta1 - theta2}
        + (m1+m2)*g*l1*\cos{theta1} 
        + m2*g*l2*\cos{theta2}
    }
    
    \initial{theta1=2.0, theta1_dot=0, theta2=2.5, theta2_dot=0}
    """
    
    print("=" * 60)
    print("MechanicsDSL C++ CMake Example Generator")
    print("=" * 60)
    
    # Compile the DSL
    print("\n[1/3] Compiling DSL...")
    compiler = PhysicsCompiler()
    compiler.compile_dsl(double_pendulum_code)
    print("      ✓ Compiled successfully")
    
    # Generate C++ project
    print("\n[2/3] Generating C++ project...")
    from mechanics_dsl.codegen.cpp import CppGenerator
    
    generator = CppGenerator(
        system_name="double_pendulum",
        coordinates=list(compiler.simulator.coordinates.keys()) if hasattr(compiler.simulator.coordinates, 'keys') else compiler.simulator.coordinates,
        parameters=compiler.simulator.parameters,
        initial_conditions=compiler.simulator.initial_conditions,
        equations=compiler.simulator.equations
    )
    
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    files = generator.generate_project(output_dir)
    
    for name, path in files.items():
        print(f"      ✓ Generated {name}: {os.path.basename(path)}")
    
    # Print build instructions
    print("\n[3/3] Build Instructions:")
    print("-" * 60)
    print(f"""
    cd {output_dir}
    mkdir build && cd build
    cmake ..
    make -j$(nproc)
    ./double_pendulum
    
    # For Raspberry Pi cross-compilation:
    cmake -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \\
          -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ ..
    make -j4
    """)
    
    print("=" * 60)
    print("Project generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
