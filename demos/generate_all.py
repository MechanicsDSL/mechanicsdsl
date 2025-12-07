"""
Backend Validation Demo: Generate Simple Pendulum in All Backends

This script generates code for all 9 backends from the same DSL definition.
Use this to validate that all backends produce equivalent simulation results.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import sympy as sp
from mechanics_dsl.codegen import (
    CppGenerator,
    PythonGenerator,
    JuliaGenerator,
    RustGenerator,
    MatlabGenerator,
    FortranGenerator,
    JavaScriptGenerator,
    CudaGenerator
)


def create_pendulum_system():
    """Define a simple pendulum system."""
    # Symbolic variables
    theta = sp.Symbol('theta', real=True)
    theta_dot = sp.Symbol('theta_dot', real=True)
    g = sp.Symbol('g', positive=True)
    l = sp.Symbol('l', positive=True)
    
    # Equation of motion: θ'' = -(g/l) * sin(θ)
    equations = {
        'theta_ddot': -g/l * sp.sin(theta)
    }
    
    return {
        'system_name': 'simple_pendulum',
        'coordinates': ['theta'],
        'parameters': {'g': 9.81, 'l': 1.0},
        'initial_conditions': {'theta': 0.3, 'theta_dot': 0.0},
        'equations': equations
    }


def main():
    """Generate code for all backends."""
    pendulum = create_pendulum_system()
    demos_dir = os.path.dirname(__file__)
    
    # C++ Backend
    print("Generating C++ code...")
    cpp_gen = CppGenerator(**pendulum)
    cpp_gen.generate(os.path.join(demos_dir, 'cpp_pendulum', 'pendulum.cpp'))
    
    # Python Backend
    print("Generating Python code...")
    py_gen = PythonGenerator(**pendulum)
    py_gen.generate(os.path.join(demos_dir, 'python_pendulum', 'pendulum.py'))
    
    # Julia Backend
    print("Generating Julia code...")
    julia_gen = JuliaGenerator(**pendulum)
    julia_gen.generate(os.path.join(demos_dir, 'julia_pendulum', 'pendulum.jl'))
    
    # Rust Backend
    print("Generating Rust code...")
    rust_gen = RustGenerator(**pendulum)
    rust_gen.generate(os.path.join(demos_dir, 'rust_pendulum', 'pendulum.rs'))
    
    # MATLAB Backend
    print("Generating MATLAB code...")
    matlab_gen = MatlabGenerator(**pendulum)
    matlab_gen.generate(os.path.join(demos_dir, 'matlab_pendulum', 'pendulum.m'))
    
    # Fortran Backend
    print("Generating Fortran code...")
    fortran_gen = FortranGenerator(**pendulum)
    fortran_gen.generate(os.path.join(demos_dir, 'fortran_pendulum', 'pendulum.f90'))
    
    # JavaScript Backend
    print("Generating JavaScript code...")
    js_gen = JavaScriptGenerator(**pendulum)
    js_gen.generate(os.path.join(demos_dir, 'javascript_pendulum', 'pendulum.js'))
    
    # CUDA Backend
    print("Generating CUDA code...")
    cuda_gen = CudaGenerator(**pendulum, generate_cpu_fallback=True)
    cuda_gen.generate(os.path.join(demos_dir, 'cuda_pendulum'))
    
    print("\n=== Generation Complete ===")
    print("Generated code for all 9 backends:")
    print("  - C++:        demos/cpp_pendulum/pendulum.cpp")
    print("  - Python:     demos/python_pendulum/pendulum.py")
    print("  - Julia:      demos/julia_pendulum/pendulum.jl")
    print("  - Rust:       demos/rust_pendulum/pendulum.rs")
    print("  - MATLAB:     demos/matlab_pendulum/pendulum.m")
    print("  - Fortran:    demos/fortran_pendulum/pendulum.f90")
    print("  - JavaScript: demos/javascript_pendulum/pendulum.js")
    print("  - CUDA:       demos/cuda_pendulum/simple_pendulum.cu")


if __name__ == '__main__':
    main()
