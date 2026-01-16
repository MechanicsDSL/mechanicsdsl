"""
Embedded Quadcopter Attitude Control
=====================================

Simulates quadcopter attitude dynamics using quaternions and
generates real-time control code for embedded processors.

This demonstrates:
- Quaternion-based attitude representation
- Angular velocity estimation from gyroscope
- PD attitude controller
- Code generation for ARM Cortex-M

Author: MechanicsDSL Team
"""

import numpy as np
from typing import Tuple
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


class Quaternion:
    """Quaternion for attitude representation."""
    
    def __init__(self, w: float = 1.0, x: float = 0.0, 
                 y: float = 0.0, z: float = 0.0):
        self.q = np.array([w, x, y, z])
    
    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float) -> 'Quaternion':
        """Create quaternion from Euler angles (radians)."""
        cr = np.cos(roll / 2)
        sr = np.sin(roll / 2)
        cp = np.cos(pitch / 2)
        sp = np.sin(pitch / 2)
        cy = np.cos(yaw / 2)
        sy = np.sin(yaw / 2)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return cls(w, x, y, z)
    
    def to_euler(self) -> Tuple[float, float, float]:
        """Convert to Euler angles (roll, pitch, yaw)."""
        w, x, y, z = self.q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def normalize(self):
        """Normalize quaternion."""
        norm = np.linalg.norm(self.q)
        if norm > 0:
            self.q /= norm
    
    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """Quaternion multiplication."""
        w1, x1, y1, z1 = self.q
        w2, x2, y2, z2 = other.q
        
        return Quaternion(
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        )
    
    def conjugate(self) -> 'Quaternion':
        """Return conjugate."""
        return Quaternion(self.q[0], -self.q[1], -self.q[2], -self.q[3])


class QuadcopterDynamics:
    """
    Quadcopter rigid body dynamics.
    
    State: [quaternion(4), angular_velocity(3)]
    Input: [motor_torques(3)]
    """
    
    def __init__(self, Ixx=0.01, Iyy=0.01, Izz=0.02):
        """
        Initialize with inertia tensor (diagonal).
        
        Args:
            Ixx: Moment of inertia around x (kg*m^2)
            Iyy: Moment of inertia around y (kg*m^2)
            Izz: Moment of inertia around z (kg*m^2)
        """
        self.I = np.diag([Ixx, Iyy, Izz])
        self.I_inv = np.linalg.inv(self.I)
        
        # State
        self.q = Quaternion()  # Identity quaternion
        self.omega = np.zeros(3)  # Angular velocity
    
    def step(self, torque: np.ndarray, dt: float):
        """Step simulation forward."""
        # Euler's equations for rigid body
        domega = self.I_inv @ (torque - np.cross(self.omega, self.I @ self.omega))
        self.omega += domega * dt
        
        # Quaternion derivative
        omega_q = Quaternion(0, self.omega[0], self.omega[1], self.omega[2])
        q_dot = self.q * omega_q
        
        self.q.q += 0.5 * q_dot.q * dt
        self.q.normalize()
    
    @property
    def euler(self) -> Tuple[float, float, float]:
        """Get Euler angles (roll, pitch, yaw)."""
        return self.q.to_euler()


class AttitudeController:
    """PD attitude controller for quadcopter."""
    
    def __init__(self, Kp_roll=10.0, Kp_pitch=10.0, Kp_yaw=5.0,
                 Kd_roll=2.0, Kd_pitch=2.0, Kd_yaw=1.0):
        self.Kp = np.array([Kp_roll, Kp_pitch, Kp_yaw])
        self.Kd = np.array([Kd_roll, Kd_pitch, Kd_yaw])
    
    def compute_torque(self, current: Tuple[float, float, float],
                       target: Tuple[float, float, float],
                       omega: np.ndarray) -> np.ndarray:
        """Compute control torques."""
        error = np.array(target) - np.array(current)
        
        # Wrap yaw error
        error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
        
        # PD control
        torque = self.Kp * error - self.Kd * omega
        
        return torque


def generate_embedded_code():
    """Generate ARM code for embedded attitude controller."""
    from mechanics_dsl import PhysicsCompiler
    from mechanics_dsl.codegen.arm import ARMGenerator
    
    # Define attitude dynamics in DSL
    attitude_dsl = r"""
    \system{quadcopter_attitude}
    
    \defvar{phi}{Roll}{rad}
    \defvar{theta}{Pitch}{rad}
    \defvar{psi}{Yaw}{rad}
    
    \parameter{Ixx}{0.01}{kg*m^2}
    \parameter{Iyy}{0.01}{kg*m^2}
    \parameter{Izz}{0.02}{kg*m^2}
    
    # Simplified Euler angle dynamics (small angle)
    \lagrangian{
        0.5 * Ixx * \dot{phi}^2 
        + 0.5 * Iyy * \dot{theta}^2 
        + 0.5 * Izz * \dot{psi}^2
    }
    
    \initial{phi=0.1, phi_dot=0, theta=0.1, theta_dot=0, psi=0, psi_dot=0}
    """
    
    print("Generating ARM code for attitude controller...")
    
    compiler = PhysicsCompiler()
    compiler.compile_dsl(attitude_dsl)
    
    gen = ARMGenerator(
        system_name="attitude_controller",
        coordinates=list(compiler.simulator.coordinates.keys()) if hasattr(compiler.simulator.coordinates, 'keys') else compiler.simulator.coordinates,
        parameters=compiler.simulator.parameters,
        initial_conditions=compiler.simulator.initial_conditions,
        equations=compiler.simulator.equations,
        target="cortex_m",
        embedded=True
    )
    
    output_dir = os.path.join(os.path.dirname(__file__), "attitude_controller")
    files = gen.generate_project(output_dir)
    
    print(f"Generated: {list(files.keys())}")
    return files


def simulate_attitude():
    """Simulate attitude control."""
    print("=" * 60)
    print("Quadcopter Attitude Control Simulation")
    print("=" * 60)
    
    quad = QuadcopterDynamics()
    controller = AttitudeController()
    
    # Start with some initial attitude error
    quad.q = Quaternion.from_euler(0.2, 0.3, 0.0)
    
    # Target: level hover
    target = (0.0, 0.0, 0.0)
    
    dt = 0.01
    t = 0.0
    t_end = 5.0
    
    print("\nSimulating stabilization...")
    print(f"{'Time':>6} {'Roll':>10} {'Pitch':>10} {'Yaw':>10}")
    print("-" * 40)
    
    while t < t_end:
        current = quad.euler
        torque = controller.compute_torque(current, target, quad.omega)
        quad.step(torque, dt)
        
        if int(t * 10) % 5 == 0:
            print(f"{t:6.2f} {np.degrees(current[0]):10.2f}° "
                  f"{np.degrees(current[1]):10.2f}° "
                  f"{np.degrees(current[2]):10.2f}°")
        
        t += dt
    
    final = quad.euler
    print("-" * 40)
    print(f"Final: Roll={np.degrees(final[0]):.2f}°, "
          f"Pitch={np.degrees(final[1]):.2f}°, "
          f"Yaw={np.degrees(final[2]):.2f}°")


def main():
    simulate_attitude()
    
    print("\n")
    
    try:
        generate_embedded_code()
    except Exception as e:
        print(f"Code generation skipped: {e}")


if __name__ == "__main__":
    main()
