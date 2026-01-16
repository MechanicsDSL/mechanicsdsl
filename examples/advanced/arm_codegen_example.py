"""
ARM Code Generation Example
===========================

Complete example demonstrating ARM code generation with:
- CMake project generation
- NEON SIMD optimizations
- Cross-compilation for Raspberry Pi
- Embedded/bare-metal option

Usage:
    python arm_codegen_example.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mechanics_dsl import PhysicsCompiler


def example_gyroscope():
    """Generate ARM code for a gyroscope simulation."""
    
    # 3D Gyroscope DSL code
    gyro_code = r"""
    \system{gyroscope}
    
    \defvar{phi}{Angle}{rad}
    \defvar{theta}{Angle}{rad}
    \defvar{psi}{Angle}{rad}
    
    \parameter{I1}{0.01}{kg*m^2}   # Moment of inertia around x
    \parameter{I2}{0.01}{kg*m^2}   # Moment of inertia around y
    \parameter{I3}{0.005}{kg*m^2}  # Moment of inertia around z (spin axis)
    \parameter{omega_s}{100}{rad/s}  # Spin angular velocity
    
    # Euler's equations for rigid body rotation
    \lagrangian{
        0.5 * I1 * \dot{phi}^2 
        + 0.5 * I2 * \dot{theta}^2 
        + 0.5 * I3 * (\dot{psi} + omega_s)^2
    }
    
    \initial{phi=0.1, phi_dot=0, theta=0.1, theta_dot=0, psi=0, psi_dot=0}
    """
    
    print("=" * 60)
    print("MechanicsDSL ARM Code Generation Example")
    print("=" * 60)
    
    # Compile DSL
    print("\n[1/4] Compiling gyroscope system...")
    compiler = PhysicsCompiler()
    compiler.compile_dsl(gyro_code)
    print("      ✓ Compiled successfully")
    
    # Generate ARM project for Raspberry Pi
    print("\n[2/4] Generating ARM project for Raspberry Pi...")
    from mechanics_dsl.codegen.arm import ARMGenerator
    
    gen = ARMGenerator(
        system_name="gyroscope",
        coordinates=list(compiler.simulator.coordinates.keys()) if hasattr(compiler.simulator.coordinates, 'keys') else compiler.simulator.coordinates,
        parameters=compiler.simulator.parameters,
        initial_conditions=compiler.simulator.initial_conditions,
        equations=compiler.simulator.equations,
        target="raspberry_pi",
        use_neon=True,
        embedded=False
    )
    
    output_dir = os.path.join(os.path.dirname(__file__), "arm_gyroscope")
    files = gen.generate_project(output_dir)
    
    for name, path in files.items():
        print(f"      ✓ {name}: {os.path.basename(path)}")
    
    # Generate embedded version for Cortex-M
    print("\n[3/4] Generating embedded version for Cortex-M...")
    
    gen_embedded = ARMGenerator(
        system_name="gyroscope_embedded",
        coordinates=list(compiler.simulator.coordinates.keys()) if hasattr(compiler.simulator.coordinates, 'keys') else compiler.simulator.coordinates,
        parameters=compiler.simulator.parameters,
        initial_conditions=compiler.simulator.initial_conditions,
        equations=compiler.simulator.equations,
        target="cortex_m",
        use_neon=False,
        embedded=True
    )
    
    embedded_dir = os.path.join(os.path.dirname(__file__), "arm_gyroscope_embedded")
    embedded_files = gen_embedded.generate_project(embedded_dir)
    
    for name, path in embedded_files.items():
        print(f"      ✓ {name}: {os.path.basename(path)}")
    
    # Print build instructions
    print("\n[4/4] Build Instructions:")
    print("-" * 60)
    print(f"""
    # Build on Raspberry Pi (native)
    cd {output_dir}
    make native
    ./gyroscope
    
    # Cross-compile from x86
    cd {output_dir}
    make cross
    scp gyroscope_cross pi@raspberrypi:~/
    
    # Build for Cortex-M (bare metal)
    cd {embedded_dir}
    arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -Os \\
        -o gyroscope_embedded gyroscope_embedded_arm.c
    """)
    
    print("=" * 60)
    print("ARM code generation complete!")
    print("=" * 60)


def example_pendulum_with_benchmarks():
    """Generate ARM code with performance benchmarks."""
    
    pendulum_code = r"""
    \system{bench_pendulum}
    
    \defvar{theta}{Angle}{rad}
    
    \parameter{m}{1.0}{kg}
    \parameter{l}{1.0}{m}
    \parameter{g}{9.81}{m/s^2}
    \parameter{b}{0.1}{1/s}  # Damping
    
    \lagrangian{0.5 * m * l^2 * \dot{theta}^2 - m * g * l * (1 - \cos{theta})}
    \dissipation{0.5 * b * l^2 * \dot{theta}^2}
    
    \initial{theta=2.5, theta_dot=0.0}
    """
    
    print("\n" + "=" * 60)
    print("ARM Performance Benchmark Generation")
    print("=" * 60)
    
    compiler = PhysicsCompiler()
    compiler.compile_dsl(pendulum_code)
    
    from mechanics_dsl.codegen.arm import ARMGenerator
    import tempfile
    import time
    
    # Benchmark code generation
    targets = ["raspberry_pi", "jetson", "cortex_m"]
    
    print("\nCode generation benchmarks:")
    print("-" * 40)
    
    for target in targets:
        with tempfile.TemporaryDirectory() as tmpdir:
            start = time.perf_counter()
            
            gen = ARMGenerator(
                system_name=f"bench_{target}",
                coordinates=list(compiler.simulator.coordinates.keys()) if hasattr(compiler.simulator.coordinates, 'keys') else compiler.simulator.coordinates,
                parameters=compiler.simulator.parameters,
                initial_conditions=compiler.simulator.initial_conditions,
                equations=compiler.simulator.equations,
                target=target,
                embedded=(target == "cortex_m")
            )
            gen.generate_project(tmpdir)
            
            elapsed = (time.perf_counter() - start) * 1000
            print(f"  {target:15} : {elapsed:6.2f} ms")
    
    print("-" * 40)


if __name__ == "__main__":
    example_gyroscope()
    example_pendulum_with_benchmarks()
