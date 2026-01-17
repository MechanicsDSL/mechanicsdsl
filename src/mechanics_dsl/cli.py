#!/usr/bin/env python3
"""
MechanicsDSL Command-Line Interface

A CLI for compiling, running, and exporting physics simulations.

Usage:
    mechanicsdsl compile <input> --target <target> [--output <path>]
    mechanicsdsl run <input> [--t-span <start,end>] [--animate] [--output <path>]
    mechanicsdsl export <input> --format <format> [--output <path>]
    mechanicsdsl info
    mechanicsdsl validate <input>

Examples:
    mechanicsdsl compile pendulum.mdsl --target cpp --output build/
    mechanicsdsl run pendulum.mdsl --t-span 0,10 --animate
    mechanicsdsl export pendulum.mdsl --format json --output results/
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import Optional, Tuple

# Version info
__version__ = "2.0.0"


def parse_t_span(t_span_str: str) -> Tuple[float, float]:
    """Parse t-span argument like '0,10' into (0.0, 10.0)."""
    try:
        parts = t_span_str.split(',')
        if len(parts) != 2:
            raise ValueError("t-span must be two comma-separated numbers")
        return float(parts[0]), float(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid t-span format: {e}")


def cmd_compile(args):
    """Compile DSL file to target language."""
    from mechanics_dsl import PhysicsCompiler
    
    # Read input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        return 1
    
    dsl_code = input_path.read_text(encoding='utf-8')
    
    # Compile
    compiler = PhysicsCompiler()
    result = compiler.compile_dsl(dsl_code)
    
    if not result.get('success', False):
        print(f"Compilation failed: {result.get('error', 'Unknown error')}", file=sys.stderr)
        return 1
    
    # Determine output path
    output_dir = Path(args.output) if args.output else Path('.')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = input_path.stem
    target = args.target.lower()
    
    # Generate code for target
    target_methods = {
        'cpp': ('compile_to_cpp', f'{base_name}.cpp'),
        'c++': ('compile_to_cpp', f'{base_name}.cpp'),
        'cuda': ('compile_to_cuda', f'{base_name}.cu'),
        'rust': ('compile_to_rust', f'{base_name}.rs'),
        'julia': ('compile_to_julia', f'{base_name}.jl'),
        'fortran': ('compile_to_fortran', f'{base_name}.f90'),
        'matlab': ('compile_to_matlab', f'{base_name}.m'),
        'javascript': ('compile_to_javascript', f'{base_name}.js'),
        'js': ('compile_to_javascript', f'{base_name}.js'),
        'wasm': ('compile_to_wasm', f'{base_name}.wat'),
        'webassembly': ('compile_to_wasm', f'{base_name}.wat'),
        'python': ('compile_to_python', f'{base_name}_sim.py'),
        'py': ('compile_to_python', f'{base_name}_sim.py'),
        'arduino': ('compile_to_arduino', f'{base_name}.ino'),
        'openmp': ('compile_to_openmp', f'{base_name}_omp.cpp'),
    }
    
    if target not in target_methods:
        print(f"Error: Unknown target '{target}'", file=sys.stderr)
        print(f"Available targets: {', '.join(sorted(set(t for t in target_methods.keys() if len(t) > 2)))}")
        return 1
    
    method_name, default_filename = target_methods[target]
    output_file = output_dir / default_filename
    
    method = getattr(compiler, method_name)
    method(str(output_file))
    
    print(f"✓ Compiled to {output_file}")
    return 0


def cmd_run(args):
    """Run a simulation from DSL file."""
    from mechanics_dsl import PhysicsCompiler
    import matplotlib.pyplot as plt
    
    # Read input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        return 1
    
    dsl_code = input_path.read_text(encoding='utf-8')
    
    # Compile
    compiler = PhysicsCompiler()
    result = compiler.compile_dsl(dsl_code)
    
    if not result.get('success', False):
        print(f"Compilation failed: {result.get('error', 'Unknown error')}", file=sys.stderr)
        return 1
    
    # Parse t-span
    t_span = parse_t_span(args.t_span) if args.t_span else (0, 10)
    num_points = args.points
    
    print(f"Running simulation: t ∈ [{t_span[0]}, {t_span[1]}], {num_points} points...")
    
    # Simulate
    solution = compiler.simulate(t_span=t_span, num_points=num_points)
    
    print(f"✓ Simulation complete")
    print(f"  Final time: {solution['t'][-1]:.4f}")
    print(f"  States computed: {len(solution['t'])}")
    
    # Save output if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        output_data = {
            't': solution['t'].tolist() if hasattr(solution['t'], 'tolist') else list(solution['t']),
            'y': [yi.tolist() if hasattr(yi, 'tolist') else list(yi) for yi in solution['y']],
            'system': result.get('system_name', 'unknown'),
            'coordinates': result.get('coordinates', []),
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Results saved to {output_path}")
    
    # Animate if requested
    if args.animate:
        print("Starting animation...")
        compiler.animate(solution, show=True)
    
    return 0


def cmd_export(args):
    """Export simulation results to various formats."""
    from mechanics_dsl import PhysicsCompiler
    import numpy as np
    
    # Read input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        return 1
    
    dsl_code = input_path.read_text(encoding='utf-8')
    
    # Compile and simulate
    compiler = PhysicsCompiler()
    result = compiler.compile_dsl(dsl_code)
    
    if not result.get('success', False):
        print(f"Compilation failed: {result.get('error', 'Unknown error')}", file=sys.stderr)
        return 1
    
    t_span = parse_t_span(args.t_span) if args.t_span else (0, 10)
    solution = compiler.simulate(t_span=t_span, num_points=args.points)
    
    # Determine output path
    output_path = Path(args.output) if args.output else Path(f"{input_path.stem}_results")
    
    format_type = args.format.lower()
    
    if format_type == 'json':
        output_file = output_path.with_suffix('.json')
        output_data = {
            't': solution['t'].tolist() if hasattr(solution['t'], 'tolist') else list(solution['t']),
            'y': [yi.tolist() if hasattr(yi, 'tolist') else list(yi) for yi in solution['y']],
            'system': result.get('system_name', 'unknown'),
            'coordinates': result.get('coordinates', []),
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
            
    elif format_type == 'csv':
        output_file = output_path.with_suffix('.csv')
        coords = result.get('coordinates', [f'y{i}' for i in range(len(solution['y']))])
        
        with open(output_file, 'w') as f:
            # Header
            header = ['t'] + coords
            f.write(','.join(header) + '\n')
            
            # Data
            for i, t in enumerate(solution['t']):
                row = [str(t)] + [str(solution['y'][j][i]) for j in range(len(solution['y']))]
                f.write(','.join(row) + '\n')
                
    elif format_type == 'numpy' or format_type == 'npz':
        output_file = output_path.with_suffix('.npz')
        np.savez(output_file, t=solution['t'], y=np.array(solution['y']))
        
    else:
        print(f"Error: Unknown format '{format_type}'", file=sys.stderr)
        print("Available formats: json, csv, numpy")
        return 1
    
    print(f"✓ Exported to {output_file}")
    return 0


def cmd_validate(args):
    """Validate a DSL file without running simulation."""
    from mechanics_dsl import PhysicsCompiler
    
    # Read input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        return 1
    
    dsl_code = input_path.read_text(encoding='utf-8')
    
    # Compile
    compiler = PhysicsCompiler()
    result = compiler.compile_dsl(dsl_code)
    
    if result.get('success', False):
        print(f"✓ Valid DSL file: {args.input}")
        print(f"  System: {result.get('system_name', 'unknown')}")
        print(f"  Coordinates: {', '.join(result.get('coordinates', []))}")
        print(f"  Parameters: {len(result.get('parameters', {}))}")
        
        # Check for conservation laws
        if 'conserved_quantities' in result:
            print(f"  Conserved quantities: {len(result['conserved_quantities'])}")
            for cq in result['conserved_quantities']:
                print(f"    - {cq}")
        
        return 0
    else:
        print(f"✗ Invalid DSL file: {args.input}", file=sys.stderr)
        print(f"  Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
        
        # Print location if available
        if 'line' in result:
            print(f"  Line: {result['line']}", file=sys.stderr)
        
        return 1


def cmd_info(args):
    """Show version and system information."""
    print(f"MechanicsDSL v{__version__}")
    print()
    print("Physics Domains:")
    print("  ✓ Classical Mechanics (17 modules)")
    print("  ✓ Quantum Mechanics")
    print("  ✓ Electromagnetism")
    print("  ✓ Special Relativity")
    print("  ✓ General Relativity")
    print("  ✓ Fluid Dynamics (SPH)")
    print("  ✓ Statistical Mechanics")
    print("  ✓ Thermodynamics")
    print("  ✓ Orbital Mechanics")
    print()
    print("Code Generation Targets:")
    print("  C++, CUDA, Rust, Julia, Fortran, MATLAB,")
    print("  JavaScript, WebAssembly, Python, Arduino, OpenMP")
    print()
    print("Documentation: https://mechanicsdsl.readthedocs.io")
    print("GitHub: https://github.com/MechanicsDSL/mechanicsdsl")
    print()
    
    # Check optional dependencies
    print("Optional Features:")
    
    try:
        import jax
        print(f"  ✓ JAX backend available (v{jax.__version__})")
    except ImportError:
        print("  ✗ JAX backend not installed (pip install mechanicsdsl-core[jax])")
    
    try:
        import numba
        print(f"  ✓ Numba JIT available (v{numba.__version__})")
    except ImportError:
        print("  ✗ Numba not installed")
    
    try:
        import fastapi
        print(f"  ✓ FastAPI server available")
    except ImportError:
        print("  ✗ FastAPI server not installed (pip install mechanicsdsl-core[server])")
    
    return 0


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog='mechanicsdsl',
        description='MechanicsDSL - A domain-specific language for computational physics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mechanicsdsl compile pendulum.mdsl --target cpp
  mechanicsdsl run pendulum.mdsl --t-span 0,20 --animate
  mechanicsdsl export system.mdsl --format csv --output results.csv
  mechanicsdsl validate mystery.mdsl
  mechanicsdsl info
        """
    )
    
    parser.add_argument('--version', '-v', action='version', version=f'MechanicsDSL {__version__}')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # compile command
    compile_parser = subparsers.add_parser('compile', help='Compile DSL to target language')
    compile_parser.add_argument('input', help='Input DSL file (.mdsl)')
    compile_parser.add_argument('--target', '-t', required=True, 
                                help='Target language (cpp, cuda, rust, julia, etc.)')
    compile_parser.add_argument('--output', '-o', help='Output directory or file')
    
    # run command
    run_parser = subparsers.add_parser('run', help='Run a simulation')
    run_parser.add_argument('input', help='Input DSL file (.mdsl)')
    run_parser.add_argument('--t-span', help='Time span as start,end (default: 0,10)')
    run_parser.add_argument('--points', '-n', type=int, default=1000, 
                           help='Number of time points (default: 1000)')
    run_parser.add_argument('--animate', '-a', action='store_true', help='Show animation')
    run_parser.add_argument('--output', '-o', help='Save results to JSON file')
    
    # export command
    export_parser = subparsers.add_parser('export', help='Export simulation results')
    export_parser.add_argument('input', help='Input DSL file (.mdsl)')
    export_parser.add_argument('--format', '-f', required=True, 
                               help='Output format (json, csv, numpy)')
    export_parser.add_argument('--output', '-o', help='Output file path')
    export_parser.add_argument('--t-span', help='Time span as start,end (default: 0,10)')
    export_parser.add_argument('--points', '-n', type=int, default=1000,
                               help='Number of time points (default: 1000)')
    
    # validate command
    validate_parser = subparsers.add_parser('validate', help='Validate a DSL file')
    validate_parser.add_argument('input', help='Input DSL file (.mdsl)')
    
    # info command
    info_parser = subparsers.add_parser('info', help='Show version and system info')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # Dispatch to appropriate command handler
    commands = {
        'compile': cmd_compile,
        'run': cmd_run,
        'export': cmd_export,
        'validate': cmd_validate,
        'info': cmd_info,
    }
    
    handler = commands.get(args.command)
    if handler:
        try:
            return handler(args)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            if os.environ.get('MECHANICSDSL_DEBUG'):
                raise
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
