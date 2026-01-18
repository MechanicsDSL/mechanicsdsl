#!/usr/bin/env python3
"""
Basic MechanicsDSL Simulation

A simple starter script demonstrating MechanicsDSL usage.
"""

from mechanics_dsl import PhysicsCompiler


def main():
    # Read the DSL file
    with open('system.mdsl', 'r') as f:
        dsl_source = f.read()
    
    # Create compiler and compile
    compiler = PhysicsCompiler()
    result = compiler.compile_dsl(dsl_source)
    
    if not result.get('success'):
        print(f"Compilation failed: {result.get('error')}")
        return 1
    
    print(f"Compiled system: {result.get('system_name')}")
    print(f"Coordinates: {result.get('coordinates')}")
    
    # Run simulation
    solution = compiler.simulate(t_span=(0, 10), num_points=1000)
    
    print(f"Simulation complete: {len(solution['t'])} points")
    
    # Create animation
    compiler.animate(solution, show=True)
    
    return 0


if __name__ == '__main__':
    exit(main())
