#!/usr/bin/env python3
"""
Example 20: Quaternion-Based Rigid Body Dynamics

Demonstrates the singularity-free quaternion formulation for rigid body
dynamics. Compares quaternion and Euler angle formulations for a spinning top.

Key advantages of quaternions:
- No gimbal lock singularities
- Numerical stability
- Compact representation (4 numbers vs 9 in rotation matrix)

Author: MechanicsDSL
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mechanics_dsl.domains.classical.rigidbody import (
    RigidBodyDynamics,
    Quaternion,
    EulerAngles,
    SymmetricTop
)


def demo_quaternion_basics():
    """Demonstrate basic quaternion operations."""
    print("=" * 60)
    print("QUATERNION BASICS")
    print("=" * 60)
    
    # Identity quaternion
    q_identity = Quaternion(1.0, 0.0, 0.0, 0.0)
    print(f"\nIdentity quaternion: {q_identity}")
    print(f"As array: {q_identity.to_array()}")
    
    # Create quaternion from Euler angles
    euler = EulerAngles(phi=0.5, theta=0.3, psi=0.7)
    q_from_euler = Quaternion.from_euler_angles(euler)
    print(f"\nEuler angles: φ={euler.phi:.3f}, θ={euler.theta:.3f}, ψ={euler.psi:.3f}")
    print(f"Equivalent quaternion: q0={q_from_euler.q0:.4f}, q1={q_from_euler.q1:.4f}, "
          f"q2={q_from_euler.q2:.4f}, q3={q_from_euler.q3:.4f}")
    
    # Rotation matrix
    R = q_from_euler.to_rotation_matrix()
    print(f"\nRotation matrix:\n{R}")
    
    # Rotate a vector
    v = np.array([1.0, 0.0, 0.0])
    v_rotated = q_from_euler.rotate_vector(v)
    print(f"\nOriginal vector: {v}")
    print(f"Rotated vector:  {v_rotated}")


def demo_quaternion_rigid_body():
    """Demonstrate quaternion formulation for rigid body dynamics."""
    print("\n" + "=" * 60)
    print("QUATERNION RIGID BODY DYNAMICS")
    print("=" * 60)
    
    # Create a symmetric top using quaternions
    body = RigidBodyDynamics("spacecraft", use_quaternions=True)
    body.set_inertia_principal(I1=1.0, I2=1.0, I3=0.5)
    
    print(f"\nSystem: {body.name}")
    print(f"Coordinates: {body.coordinates}")
    print(f"Principal moments: I1={body._I1}, I2={body._I2}, I3={body._I3}")
    
    # Compute kinetic energy
    T = body._rotational_kinetic_energy()
    print(f"\nKinetic energy expression (symbolic):")
    print(f"T = {T}")
    
    # Get the quaternion constraint
    constraint = body.quaternion_constraint()
    print(f"\nUnit quaternion constraint:")
    print(f"g(q) = {constraint} = 0")
    
    # Derive equations of motion
    print("\nDeriving equations of motion...")
    eom = body.derive_equations_of_motion()
    print(f"Derived {len(eom)} acceleration equations")
    for key in eom:
        print(f"  {key}")


def demo_gimbal_lock_avoidance():
    """Demonstrate that quaternions avoid gimbal lock."""
    print("\n" + "=" * 60)
    print("GIMBAL LOCK AVOIDANCE")
    print("=" * 60)
    
    # At θ = 0 (gimbal lock orientation for Euler angles)
    euler_gimbal = EulerAngles(phi=0.5, theta=0.0, psi=0.3)
    q_gimbal = Quaternion.from_euler_angles(euler_gimbal)
    
    print(f"\nAt gimbal lock (θ=0):")
    print(f"Euler angles: φ={euler_gimbal.phi:.3f}, θ={euler_gimbal.theta:.3f}, ψ={euler_gimbal.psi:.3f}")
    print(f"Quaternion: {q_gimbal}")
    
    # The quaternion is well-defined even at gimbal lock
    norm = np.sqrt(q_gimbal.q0**2 + q_gimbal.q1**2 + 
                   q_gimbal.q2**2 + q_gimbal.q3**2)
    print(f"Quaternion norm: {norm:.10f} (should be 1.0)")
    
    # Rotation matrix is still valid
    R = q_gimbal.to_rotation_matrix()
    det = np.linalg.det(R)
    print(f"Rotation matrix determinant: {det:.10f} (should be 1.0)")


def demo_spacecraft_attitude():
    """Simulate spacecraft attitude dynamics using quaternions."""
    print("\n" + "=" * 60)
    print("SPACECRAFT ATTITUDE SIMULATION")
    print("=" * 60)
    
    # Initial orientation (small tilt from vertical)
    q0 = Quaternion.from_euler_angles(EulerAngles(phi=0.0, theta=0.1, psi=0.0))
    
    # Initial angular velocity (body frame)
    omega = np.array([0.1, 0.0, 1.0])  # Mainly spinning about z-axis
    
    # Inertia tensor for cylindrical spacecraft
    I1, I2, I3 = 100.0, 100.0, 50.0  # kg*m²
    
    print(f"\nInitial quaternion: {q0}")
    print(f"Initial ω (body): {omega} rad/s")
    print(f"Inertia: I1={I1}, I2={I2}, I3={I3} kg*m²")
    
    # Simple Euler integration for demo (not for production!)
    dt = 0.01
    t_final = 10.0
    steps = int(t_final / dt)
    
    # Storage
    times = np.zeros(steps)
    quaternions = np.zeros((steps, 4))
    euler_angles = np.zeros((steps, 3))
    norms = np.zeros(steps)
    
    q = np.array([q0.q0, q0.q1, q0.q2, q0.q3])
    
    for i in range(steps):
        times[i] = i * dt
        quaternions[i] = q
        norms[i] = np.linalg.norm(q)
        
        # Convert to Euler angles for plotting
        q_obj = Quaternion(q[0], q[1], q[2], q[3])
        euler = q_obj.to_euler_angles()
        euler_angles[i] = [euler.phi, euler.theta, euler.psi]
        
        # Quaternion derivative: q̇ = (1/2) * Ω(ω) * q
        # where Ω is the quaternion omega matrix
        omega_quat = np.array([0, omega[0], omega[1], omega[2]])
        q_dot = 0.5 * quaternion_multiply(omega_quat, q)
        
        # Euler step
        q = q + dt * q_dot
        
        # Renormalize (constraint projection)
        q = q / np.linalg.norm(q)
        
        # Update angular velocity (torque-free, so use Euler's equations)
        omega_dot = np.array([
            (I2 - I3) * omega[1] * omega[2] / I1,
            (I3 - I1) * omega[0] * omega[2] / I2,
            (I1 - I2) * omega[0] * omega[1] / I3
        ])
        omega = omega + dt * omega_dot
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Quaternion components
    axes[0, 0].plot(times, quaternions[:, 0], label='q0')
    axes[0, 0].plot(times, quaternions[:, 1], label='q1')
    axes[0, 0].plot(times, quaternions[:, 2], label='q2')
    axes[0, 0].plot(times, quaternions[:, 3], label='q3')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Quaternion components')
    axes[0, 0].set_title('Quaternion Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Euler angles
    axes[0, 1].plot(times, np.degrees(euler_angles[:, 0]), label='φ (precession)')
    axes[0, 1].plot(times, np.degrees(euler_angles[:, 1]), label='θ (nutation)')
    axes[0, 1].plot(times, np.degrees(euler_angles[:, 2]), label='ψ (spin)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Angle (degrees)')
    axes[0, 1].set_title('Euler Angles from Quaternion')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Quaternion norm (should stay 1.0)
    axes[1, 0].plot(times, norms - 1.0)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('|q| - 1')
    axes[1, 0].set_title('Quaternion Norm Deviation')
    axes[1, 0].grid(True)
    
    # 3D trajectory of body z-axis
    ax3d = fig.add_subplot(2, 2, 4, projection='3d')
    z_body = np.zeros((steps, 3))
    for i in range(steps):
        q_obj = Quaternion(quaternions[i, 0], quaternions[i, 1], 
                          quaternions[i, 2], quaternions[i, 3])
        z_body[i] = q_obj.rotate_vector(np.array([0, 0, 1]))
    
    ax3d.plot(z_body[:, 0], z_body[:, 1], z_body[:, 2])
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title('Body Z-axis Trajectory')
    
    plt.tight_layout()
    plt.savefig('quaternion_spacecraft.png', dpi=150)
    print(f"\nPlot saved to quaternion_spacecraft.png")
    plt.show()


def quaternion_multiply(q1, q2):
    """Quaternion multiplication q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


if __name__ == '__main__':
    demo_quaternion_basics()
    demo_quaternion_rigid_body()
    demo_gimbal_lock_avoidance()
    demo_spacecraft_attitude()
