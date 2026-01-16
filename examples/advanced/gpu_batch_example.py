"""
GPU Batch Simulation Example
============================

Demonstrates CUDA batch simulation for:
- Parameter sweeps
- Monte Carlo analysis
- Sensitivity studies

Generates code that runs 1000+ parallel simulations on GPU.

Usage:
    python gpu_batch_example.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mechanics_dsl import PhysicsCompiler


def monte_carlo_pendulum():
    """Generate CUDA code for Monte Carlo analysis of pendulum."""
    
    pendulum_code = r"""
    \system{monte_carlo_pendulum}
    
    \defvar{theta}{Angle}{rad}
    
    \parameter{m}{1.0}{kg}
    \parameter{l}{1.0}{m}
    \parameter{g}{9.81}{m/s^2}
    
    \lagrangian{0.5 * m * l^2 * \dot{theta}^2 - m * g * l * (1 - \cos{theta})}
    
    \initial{theta=0.5, theta_dot=0.0}
    """
    
    print("=" * 60)
    print("MechanicsDSL GPU Batch Simulation Example")
    print("=" * 60)
    
    # Compile
    print("\n[1/3] Compiling DSL...")
    compiler = PhysicsCompiler()
    compiler.compile_dsl(pendulum_code)
    print("      ✓ Compiled successfully")
    
    # Generate CUDA batch simulation
    print("\n[2/3] Generating CUDA batch simulation code...")
    from mechanics_dsl.codegen.cuda import CudaGenerator
    
    gen = CudaGenerator(
        system_name="monte_carlo_pendulum",
        coordinates=list(compiler.simulator.coordinates.keys()) if hasattr(compiler.simulator.coordinates, 'keys') else compiler.simulator.coordinates,
        parameters=compiler.simulator.parameters,
        initial_conditions=compiler.simulator.initial_conditions,
        equations=compiler.simulator.equations,
        use_cublas=True,
        batch_size=10000,  # 10,000 parallel simulations
        compute_capability="70"
    )
    
    output_dir = os.path.join(os.path.dirname(__file__), "cuda_monte_carlo")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate standard CUDA files
    cuda_file = gen.generate(output_dir)
    print(f"      ✓ Standard CUDA: {os.path.basename(cuda_file)}")
    
    # Generate batch simulation
    batch_file = gen.generate_batch_simulation(output_dir)
    print(f"      ✓ Batch CUDA: {os.path.basename(batch_file)}")
    
    # Print instructions
    print("\n[3/3] Build Instructions:")
    print("-" * 60)
    print(f"""
    cd {output_dir}
    
    # Build standard simulation
    nvcc -arch=sm_70 -O3 -o monte_carlo_pendulum monte_carlo_pendulum.cu
    ./monte_carlo_pendulum
    
    # Build batch simulation (10,000 parallel)
    nvcc -arch=sm_70 -O3 -o batch monte_carlo_pendulum_batch.cu
    ./batch
    
    # With cuBLAS (for matrix operations)
    nvcc -arch=sm_70 -O3 -lcublas -DUSE_CUBLAS -o with_cublas monte_carlo_pendulum.cu
    """)
    
    print("=" * 60)
    print("CUDA batch simulation code generated!")
    print("=" * 60)


def parameter_sweep_oscillator():
    """Generate CUDA code for parameter sweep of oscillator."""
    
    oscillator_code = r"""
    \system{param_sweep}
    
    \defvar{x}{Position}{m}
    
    \parameter{m}{1.0}{kg}
    \parameter{k}{10.0}{N/m}
    \parameter{b}{0.5}{Ns/m}
    
    \lagrangian{0.5 * m * \dot{x}^2 - 0.5 * k * x^2}
    \dissipation{0.5 * b * \dot{x}^2}
    
    \initial{x=1.0, x_dot=0.0}
    """
    
    print("\n" + "=" * 60)
    print("Parameter Sweep CUDA Generation")
    print("=" * 60)
    
    compiler = PhysicsCompiler()
    compiler.compile_dsl(oscillator_code)
    
    from mechanics_dsl.codegen.cuda import CudaGenerator
    
    # Generate with different batch sizes
    batch_sizes = [100, 1000, 10000]
    
    for batch in batch_sizes:
        gen = CudaGenerator(
            system_name=f"sweep_{batch}",
            coordinates=list(compiler.simulator.coordinates.keys()) if hasattr(compiler.simulator.coordinates, 'keys') else compiler.simulator.coordinates,
            parameters=compiler.simulator.parameters,
            initial_conditions=compiler.simulator.initial_conditions,
            equations=compiler.simulator.equations,
            batch_size=batch
        )
        
        output = os.path.join(os.path.dirname(__file__), "cuda_sweeps")
        os.makedirs(output, exist_ok=True)
        
        batch_file = gen.generate_batch_simulation(output)
        print(f"  Batch {batch:,}: {os.path.basename(batch_file)}")
    
    print("\nSweep configurations generated!")


if __name__ == "__main__":
    monte_carlo_pendulum()
    parameter_sweep_oscillator()
