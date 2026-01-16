"""
Raspberry Pi Pendulum Example
=============================

A complete example demonstrating MechanicsDSL on Raspberry Pi hardware.
This example simulates a pendulum and optionally reads IMU sensor data.

Hardware Requirements:
- Raspberry Pi (any model with GPIO)
- Optional: MPU6050 IMU sensor for real physics data

Author: MechanicsDSL Team
License: MIT
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for Pi without display
import matplotlib.pyplot as plt

from mechanics_dsl import PhysicsCompiler

# =============================================================================
# Check if running on actual Raspberry Pi
# =============================================================================
def is_raspberry_pi():
    """Detect if we're running on a Raspberry Pi."""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            return 'Raspberry Pi' in f.read()
    except:
        return False

IS_PI = is_raspberry_pi()
print(f"Running on {'Raspberry Pi' if IS_PI else 'standard system'}")

# =============================================================================
# Optional: Hardware IMU Integration
# =============================================================================
class MockIMU:
    """Mock IMU for systems without hardware."""
    def get_angle(self):
        return np.random.normal(0, 0.1)
    
    def get_angular_velocity(self):
        return np.random.normal(0, 0.05)

def get_imu():
    """Get IMU instance (real or mock)."""
    if IS_PI:
        try:
            import smbus2
            # Real MPU6050 implementation would go here
            print("Hardware IMU detected")
            return MockIMU()  # Replace with real MPU6050 class
        except ImportError:
            print("smbus2 not installed, using mock IMU")
            return MockIMU()
    else:
        return MockIMU()

# =============================================================================
# Pendulum Simulation
# =============================================================================

# Define pendulum system using MechanicsDSL
pendulum_code = r"""
\system{rpi_pendulum}

\defvar{theta}{Angle}{rad}
\parameter{m}{1.0}{kg}
\parameter{l}{0.5}{m}
\parameter{g}{9.81}{m/s^2}
\parameter{b}{0.1}{1/s}  # Damping coefficient

# Damped simple pendulum Lagrangian
\lagrangian{
    0.5 * m * l^2 * \dot{theta}^2 - m * g * l * (1 - \cos{theta})
}

# Add damping via Rayleigh dissipation
\dissipation{0.5 * b * l^2 * \dot{theta}^2}

# Initial conditions
\initial{theta=0.5, theta_dot=0.0}
"""

def run_simulation():
    """Run pendulum simulation."""
    print("Compiling pendulum system...")
    compiler = PhysicsCompiler()
    compiler.compile_dsl(pendulum_code)
    
    # Simulation parameters optimized for Pi (fewer points for speed)
    t_span = (0, 10)
    num_points = 500 if IS_PI else 1000
    
    print(f"Running simulation ({num_points} points)...")
    solution = compiler.simulate(t_span=t_span, num_points=num_points)
    
    return solution

def plot_results(solution):
    """Generate and save plot."""
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(solution['t'], solution['theta'], 'b-', linewidth=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title('Raspberry Pi Pendulum Simulation')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(solution['t'], solution['theta_dot'], 'r-', linewidth=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pendulum_results.png', dpi=150)
    print("Plot saved to pendulum_results.png")

def export_for_realtime():
    """Export C++ code for real-time control on Pi."""
    from mechanics_dsl.codegen.cpp import CppGenerator
    
    compiler = PhysicsCompiler()
    compiler.compile_dsl(pendulum_code)
    
    # Create C++ generator
    generator = CppGenerator(
        system_name="rpi_pendulum",
        coordinates=compiler.simulator.coordinates,
        parameters=compiler.simulator.parameters,
        initial_conditions=compiler.simulator.initial_conditions,
        equations=compiler.simulator.equations
    )
    
    # Generate complete project with ARM optimizations
    print("Generating C++ project for Raspberry Pi...")
    files = generator.generate_project("./rpi_pendulum_cpp")
    
    print(f"Generated files:")
    for name, path in files.items():
        print(f"  {name}: {path}")
    
    print("\nTo build on Raspberry Pi:")
    print("  cd rpi_pendulum_cpp")
    print("  mkdir build && cd build")
    print("  cmake .. && make -j4")
    print("  ./rpi_pendulum")

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MechanicsDSL Raspberry Pi Demo")
    parser.add_argument('--simulate', action='store_true', help="Run Python simulation")
    parser.add_argument('--export-cpp', action='store_true', help="Export C++ code")
    parser.add_argument('--use-imu', action='store_true', help="Test IMU reading")
    
    args = parser.parse_args()
    
    if args.use_imu:
        imu = get_imu()
        print(f"IMU angle: {imu.get_angle():.4f} rad")
        print(f"IMU velocity: {imu.get_angular_velocity():.4f} rad/s")
    
    if args.export_cpp:
        export_for_realtime()
    
    if args.simulate or (not args.export_cpp and not args.use_imu):
        solution = run_simulation()
        plot_results(solution)
        print(f"Final angle: {solution['theta'][-1]:.4f} rad")
