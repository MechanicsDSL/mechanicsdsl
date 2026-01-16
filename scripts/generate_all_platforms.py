"""
Cross-Platform Code Generation Script
======================================

Generates code for all supported platforms from a single DSL file.

Usage:
    python generate_all_platforms.py pendulum.dsl output/
"""

import os
import sys
import argparse
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def generate_all_platforms(dsl_code: str, output_dir: str, system_name: str = "simulation"):
    """Generate code for all platforms from DSL code."""
    from mechanics_dsl import PhysicsCompiler
    from mechanics_dsl.codegen import (
        CppGenerator, RustGenerator, ARMGenerator, CudaGenerator,
        PythonGenerator, JuliaGenerator
    )
    
    print("=" * 60)
    print("Cross-Platform Code Generation")
    print("=" * 60)
    
    # Compile DSL
    print("\n[1] Compiling DSL...")
    compiler = PhysicsCompiler()
    compiler.compile_dsl(dsl_code)
    print("    ✓ Compiled successfully")
    
    # Common parameters
    coords = list(compiler.simulator.coordinates.keys()) if hasattr(compiler.simulator.coordinates, 'keys') else compiler.simulator.coordinates
    params = {
        'system_name': system_name,
        'coordinates': coords,
        'parameters': compiler.simulator.parameters,
        'initial_conditions': compiler.simulator.initial_conditions,
        'equations': compiler.simulator.equations
    }
    
    # Generate for each platform
    platforms = [
        ("C++ (CMake)", CppGenerator, "cpp", {}),
        ("Rust (Cargo)", RustGenerator, "rust", {}),
        ("ARM (Raspberry Pi)", ARMGenerator, "arm_pi", {'target': 'raspberry_pi', 'use_neon': True}),
        ("ARM (Cortex-M)", ARMGenerator, "arm_cortex", {'target': 'cortex_m', 'embedded': True}),
        ("CUDA (GPU)", CudaGenerator, "cuda", {'use_cublas': True}),
        ("Python", PythonGenerator, "python", {}),
        ("Julia", JuliaGenerator, "julia", {}),
    ]
    
    print(f"\n[2] Generating for {len(platforms)} platforms...")
    
    generated = {}
    for name, Generator, subdir, extra_params in platforms:
        try:
            platform_dir = os.path.join(output_dir, subdir)
            os.makedirs(platform_dir, exist_ok=True)
            
            gen = Generator(**params, **extra_params)
            
            if hasattr(gen, 'generate_project'):
                files = gen.generate_project(platform_dir)
                generated[name] = list(files.values())
            else:
                ext = gen.file_extension
                output = gen.generate(os.path.join(platform_dir, f"{system_name}{ext}"))
                generated[name] = [output]
            
            print(f"    ✓ {name}")
        except Exception as e:
            print(f"    ✗ {name}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("GENERATED FILES")
    print("=" * 60)
    
    total_files = 0
    for platform, files in generated.items():
        print(f"\n{platform}:")
        for f in files:
            print(f"  - {os.path.relpath(f, output_dir)}")
            total_files += 1
    
    print(f"\nTotal: {total_files} files generated in {output_dir}")
    
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate code for all platforms")
    parser.add_argument("--output", "-o", default="generated", help="Output directory")
    parser.add_argument("--name", "-n", default="simulation", help="System name")
    args = parser.parse_args()
    
    # Example DSL code
    example_code = r"""
    \system{example}
    
    \defvar{theta}{Angle}{rad}
    
    \parameter{m}{1.0}{kg}
    \parameter{l}{1.0}{m}
    \parameter{g}{9.81}{m/s^2}
    
    \lagrangian{0.5 * m * l^2 * \dot{theta}^2 - m * g * l * (1 - \cos{theta})}
    
    \initial{theta=0.5, theta_dot=0.0}
    """
    
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    generate_all_platforms(example_code, output_dir, args.name)


if __name__ == "__main__":
    main()
