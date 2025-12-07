#!/usr/bin/env python3
"""
OpenMP Demo: Generate and compile parallel pendulum simulation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import sympy as sp
from mechanics_dsl.codegen import OpenMPGenerator


def main():
    """Generate OpenMP pendulum simulation."""
    theta = sp.Symbol('theta', real=True)
    g = sp.Symbol('g', positive=True)
    l = sp.Symbol('l', positive=True)
    
    gen = OpenMPGenerator(
        system_name="pendulum",
        coordinates=['theta'],
        parameters={'g': 9.81, 'l': 1.0},
        initial_conditions={'theta': 0.3, 'theta_dot': 0.0},
        equations={'theta_ddot': -g/l * sp.sin(theta)},
        num_threads=0  # Auto-detect
    )
    
    output_dir = os.path.dirname(__file__)
    output_file = os.path.join(output_dir, 'pendulum_openmp.cpp')
    gen.generate(output_file)
    
    print(f"Generated: {output_file}")
    print("\nTo compile:")
    print("  g++ -fopenmp -O3 -o pendulum_openmp pendulum_openmp.cpp")
    print("\nTo run:")
    print("  ./pendulum_openmp")


if __name__ == '__main__':
    main()
