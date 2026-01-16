"""
Rust Cargo Example Project
==========================

This example demonstrates generating a complete Rust project with Cargo
from a MechanicsDSL physics simulation.

Usage:
    python generate_project.py
    cd output
    cargo run --release
"""

import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from mechanics_dsl import PhysicsCompiler


def main():
    """Generate a complete Rust Cargo project for a harmonic oscillator."""
    
    # Define a damped harmonic oscillator
    oscillator_code = r"""
    \system{harmonic_oscillator}
    
    \defvar{x}{Position}{m}
    
    \parameter{m}{1.0}{kg}
    \parameter{k}{10.0}{N/m}
    \parameter{b}{0.5}{Ns/m}
    
    # Damped harmonic oscillator Lagrangian
    \lagrangian{0.5 * m * \dot{x}^2 - 0.5 * k * x^2}
    
    # Rayleigh dissipation function for damping
    \dissipation{0.5 * b * \dot{x}^2}
    
    \initial{x=1.0, x_dot=0.0}
    """
    
    print("=" * 60)
    print("MechanicsDSL Rust Cargo Example Generator")
    print("=" * 60)
    
    # Compile the DSL
    print("\n[1/4] Compiling DSL...")
    compiler = PhysicsCompiler()
    compiler.compile_dsl(oscillator_code)
    print("      ✓ Compiled successfully")
    
    # Generate standard Rust project
    print("\n[2/4] Generating standard Rust project...")
    from mechanics_dsl.codegen.rust import RustGenerator
    
    generator = RustGenerator(
        system_name="harmonic_oscillator",
        coordinates=list(compiler.simulator.coordinates.keys()) if hasattr(compiler.simulator.coordinates, 'keys') else compiler.simulator.coordinates,
        parameters=compiler.simulator.parameters,
        initial_conditions=compiler.simulator.initial_conditions,
        equations=compiler.simulator.equations
    )
    
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    files = generator.generate_project(output_dir, embedded=False)
    
    for name, path in files.items():
        print(f"      ✓ Generated {name}: {os.path.basename(path)}")
    
    # Generate embedded/no_std Rust project
    print("\n[3/4] Generating embedded (no_std) Rust project...")
    embedded_dir = os.path.join(os.path.dirname(__file__), "output_embedded")
    embedded_files = generator.generate_project(embedded_dir, embedded=True)
    
    for name, path in embedded_files.items():
        print(f"      ✓ Generated {name}: {os.path.basename(path)}")
    
    # Print build instructions
    print("\n[4/4] Build Instructions:")
    print("-" * 60)
    print(f"""
    # Standard build (desktop/server)
    cd {output_dir}
    cargo run --release
    
    # Cross-compile for Raspberry Pi
    rustup target add aarch64-unknown-linux-gnu
    cargo build --release --target aarch64-unknown-linux-gnu
    
    # Embedded build (Cortex-M)
    cd {embedded_dir}
    rustup target add thumbv7em-none-eabihf
    cargo build --release --target thumbv7em-none-eabihf
    """)
    
    print("=" * 60)
    print("Projects generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
