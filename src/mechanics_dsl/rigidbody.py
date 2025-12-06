"""
Rigid Body Dynamics Module for MechanicsDSL
============================================

Handles 3D rotations, inertia tensors, quaternions, and Euler's equations.
"""

import numpy as np
import sympy as sp
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from .utils import logger


@dataclass
class InertiaTensor:
    """Moment of inertia tensor for rigid body"""
    I_xx: float
    I_yy: float
    I_zz: float
    I_xy: float = 0.0
    I_xz: float = 0.0
    I_yz: float = 0.0
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 matrix"""
        return np.array([
            [self.I_xx, -self.I_xy, -self.I_xz],
            [-self.I_xy, self.I_yy, -self.I_yz],
            [-self.I_xz, -self.I_yz, self.I_zz]
        ])
    
    def principal_axes(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find principal axes and principal moments
        Returns: (eigenvalues, eigenvectors)
        """
        I_matrix = self.to_matrix()
        eigenvalues, eigenvectors = np.linalg.eigh(I_matrix)
        return eigenvalues, eigenvectors


class RigidBody3D:
    """
    3D rigid body dynamics with Euler angles and quaternions
    
    Supports:
    - Euler angle representation (ZYZ convention)
    - Quaternion representation (singularity-free)
    - Moment of inertia tensor calculations
    - Euler's equations of motion
    - Torque-free precession
    """
    
    @staticmethod
    def euler_to_rotation_matrix(theta: float, phi: float, psi: float) -> np.ndarray:
        """
        Convert ZYZ Euler angles to rotation matrix
        
        Args:
            theta: Nutation angle (0 to π)
            phi: Precession angle (0 to 2π)
            psi: Spin angle (0 to 2π)
            
        Returns:
            3x3 rotation matrix
        """
        # ZYZ convention
        R_z_phi = np.array([
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi), 0],
            [0, 0, 1]
        ])
        
        R_y_theta = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        
        R_z_psi = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        
        return R_z_phi @ R_y_theta @ R_z_psi
    
    @staticmethod
    def quaternion_to_rotation_matrix(q0: float, q1: float, q2: float, q3: float) -> np.ndarray:
        """
        Convert unit quaternion to rotation matrix
        
        Quaternion: q = q0 + q1*i + q2*j + q3*k
        Constraint: q0² + q1² + q2² + q3² = 1
        
        Args:
            q0, q1, q2, q3: Quaternion components
            
        Returns:
            3x3 rotation matrix
        """
        # Normalize quaternion
        norm = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
        q0, q1, q2, q3 = q0/norm, q1/norm, q2/norm, q3/norm
        
        R = np.array([
            [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
        ])
        
        return R
    
    @staticmethod
    def rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Convert rotation matrix to quaternion
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            (q0, q1, q2, q3) quaternion components
        """
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            q0 = 0.25 / s
            q1 = (R[2, 1] - R[1, 2]) * s
            q2 = (R[0, 2] - R[2, 0]) * s
            q3 = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q0 = (R[2, 1] - R[1, 2]) / s
            q1 = 0.25 * s
            q2 = (R[0, 1] + R[1, 0]) / s
            q3 = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q0 = (R[0, 2] - R[2, 0]) / s
            q1 = (R[0, 1] + R[1, 0]) / s
            q2 = 0.25 * s
            q3 = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q0 = (R[1, 0] - R[0, 1]) / s
            q1 = (R[0, 2] + R[2, 0]) / s
            q2 = (R[1, 2] + R[2, 1]) / s
            q3 = 0.25 * s
        
        return q0, q1, q2, q3
    
    @staticmethod
    def compute_inertia_sphere(mass: float, radius: float) -> InertiaTensor:
        """
        Calculate inertia tensor for solid sphere
        
        I = (2/5) * m * r²
        """
        I_val = 0.4 * mass * radius**2
        return InertiaTensor(I_val, I_val, I_val)
    
    @staticmethod
    def compute_inertia_cylinder(mass: float, radius: float, height: float, 
                                 axis: str = 'z') -> InertiaTensor:
        """
        Calculate inertia tensor for solid cylinder
        
        Args:
            mass: Total mass
            radius: Cylinder radius
            height: Cylinder height
            axis: Axis of symmetry ('x', 'y', or 'z')
        """
        I_axis = 0.5 * mass * radius**2
        I_perp = (1/12) * mass * (3*radius**2 + height**2)
        
        if axis == 'z':
            return InertiaTensor(I_perp, I_perp, I_axis)
        elif axis == 'y':
            return InertiaTensor(I_axis, I_perp, I_perp)
        else:  # 'x'
            return InertiaTensor(I_perp, I_axis, I_perp)
    
    @staticmethod
    def compute_inertia_box(mass: float, length: float, width: float, height: float) -> InertiaTensor:
        """
        Calculate inertia tensor for rectangular box
        
        I_xx = (1/12) * m * (h² + w²)
        I_yy = (1/12) * m * (l² + h²)
        I_zz = (1/12) * m * (l² + w²)
        """
        I_xx = (1/12) * mass * (height**2 + width**2)
        I_yy = (1/12) * mass * (length**2 + height**2)
        I_zz = (1/12) * mass * (length**2 + width**2)
        return InertiaTensor(I_xx, I_yy, I_zz)
    
    @staticmethod
    def compute_inertia_rod(mass: float, length: float, axis: str = 'z') -> InertiaTensor:
        """
        Calculate inertia tensor for thin rod
        
        About center, perpendicular to rod: I = (1/12) * m * L²
        About axis along rod: I = 0
        """
        I_perp = (1/12) * mass * length**2
        
        if axis == 'z':
            return InertiaTensor(I_perp, I_perp, 0.0)
        elif axis == 'y':
            return InertiaTensor(I_perp, 0.0, I_perp)
        else:  # 'x'
            return InertiaTensor(0.0, I_perp, I_perp)
    
    @staticmethod
    def euler_equations_torque_free(omega: np.ndarray, I: InertiaTensor) -> np.ndarray:
        """
        Euler's equations for torque-free rigid body rotation
        
        I₁ω̇₁ = (I₂ - I₃)ω₂ω₃
        I₂ω̇₂ = (I₃ - I₁)ω₃ω₁
        I₃ω̇₃ = (I₁ - I₂)ω₁ω₂
        
        Args:
            omega: Angular velocity vector [ω₁, ω₂, ω₃]
            I: Inertia tensor
            
        Returns:
            Angular acceleration [ω̇₁, ω̇₂, ω̇₃]
        """
        I1 = I.I_xx
        I2 = I.I_yy
        I3 = I.I_zz
        
        omega1, omega2, omega3 = omega
        
        omega_dot1 = (I2 - I3) / I1 * omega2 * omega3
        omega_dot2 = (I3 - I1) / I2 * omega3 * omega1
        omega_dot3 = (I1 - I2) / I3 * omega1 * omega2
        
        return np.array([omega_dot1, omega_dot2, omega_dot3])
    
    @staticmethod
    def euler_equations_with_torque(omega: np.ndarray, I: InertiaTensor, 
                                    torque: np.ndarray) -> np.ndarray:
        """
        Euler's equations with external torque
        
        I·ω̇ + ω × (I·ω) = τ
        
        Args:
            omega: Angular velocity vector
            I: Inertia tensor
            torque: External torque vector
            
        Returns:
            Angular acceleration
        """
        I_matrix = I.to_matrix()
        I_omega = I_matrix @ omega
        omega_cross_I_omega = np.cross(omega, I_omega)
        
        omega_dot = np.linalg.solve(I_matrix, torque - omega_cross_I_omega)
        return omega_dot
    
    @staticmethod
    def quaternion_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """
        Quaternion kinematic equation
        
        dq/dt = 0.5 * Ω(ω) * q
        
        Where Ω(ω) is the skew-symmetric matrix:
        Ω = [   0   -ω₁  -ω₂  -ω₃]
            [  ω₁    0    ω₃  -ω₂]
            [  ω₂  -ω₃    0    ω₁]
            [  ω₃   ω₂  -ω₁    0 ]
        
        Args:
            q: Quaternion [q0, q1, q2, q3]
            omega: Angular velocity in body frame [ω₁, ω₂, ω₃]
            
        Returns:
            Quaternion derivative [q̇0, q̇1, q̇2, q̇3]
        """
        q0, q1, q2, q3 = q
        w1, w2, w3 = omega
        
        Omega = np.array([
            [0, -w1, -w2, -w3],
            [w1, 0, w3, -w2],
            [w2, -w3, 0, w1],
            [w3, w2, -w1, 0]
        ])
        
        q_dot = 0.5 * Omega @ q
        return q_dot
    
    @staticmethod
    def angular_momentum(omega: np.ndarray, I: InertiaTensor) -> np.ndarray:
        """
        Calculate angular momentum L = I·ω
        
        Args:
            omega: Angular velocity
            I: Inertia tensor
            
        Returns:
            Angular momentum vector
        """
        I_matrix = I.to_matrix()
        return I_matrix @ omega
    
    @staticmethod
    def rotational_kinetic_energy(omega: np.ndarray, I: InertiaTensor) -> float:
        """
        Calculate rotational kinetic energy
        
        T_rot = 0.5 * ω^T · I · ω
        
        Args:
            omega: Angular velocity
            I: Inertia tensor
            
        Returns:
            Kinetic energy (scalar)
        """
        I_matrix = I.to_matrix()
        return 0.5 * omega @ I_matrix @ omega


class SpinningTop:
    """
    Special case: Symmetric spinning top
    
    Lagrangian formulation for top with I₁ = I₂ ≠ I₃
    """
    
    @staticmethod
    def lagrangian_symmetric_top(theta: float, phi: float, psi: float,
                                 theta_dot: float, phi_dot: float, psi_dot: float,
                                 I1: float, I3: float, m: float, g: float, l: float) -> float:
        """
        Lagrangian for symmetric top
        
        T = 0.5*I1*(θ̇² + sin²θ φ̇²) + 0.5*I3*(ψ̇ + cosθ φ̇)²
        V = mgl cosθ
        
        Args:
            theta, phi, psi: Euler angles
            theta_dot, phi_dot, psi_dot: Angular velocities
            I1, I3: Moments of inertia
            m, g, l: Mass, gravity, distance to CM
            
        Returns:
            Lagrangian L = T - V
        """
        T = (0.5 * I1 * (theta_dot**2 + np.sin(theta)**2 * phi_dot**2) +
             0.5 * I3 * (psi_dot + np.cos(theta) * phi_dot)**2)
        V = m * g * l * np.cos(theta)
        return T - V
    
    @staticmethod
    def conserved_quantities(theta: float, phi_dot: float, psi_dot: float,
                            I1: float, I3: float) -> Tuple[float, float]:
        """
        Calculate conserved quantities for symmetric top
        
        Returns:
            (L_z, L_psi) - conserved angular momenta
        """
        # L_z = ∂L/∂φ̇ (constant due to azimuthal symmetry)
        L_z = I1 * np.sin(theta)**2 * phi_dot + I3 * np.cos(theta) * (psi_dot + np.cos(theta) * phi_dot)
        
        # L_ψ = ∂L/∂ψ̇ (constant due to spin symmetry)
        L_psi = I3 * (psi_dot + np.cos(theta) * phi_dot)
        
        return L_z, L_psi


# Example: Dzhanibekov Effect (Tennis Racket Theorem)
class DzhanibekovEffect:
    """
    Demonstrates instability of rotation about intermediate axis
    (Tennis Racket Theorem)
    """
    
    @staticmethod
    def is_stable_axis(I1: float, I2: float, I3: float, axis: int) -> bool:
        """
        Check if rotation about given axis is stable
        
        Args:
            I1, I2, I3: Principal moments (sorted I1 < I2 < I3)
            axis: 1, 2, or 3
            
        Returns:
            True if stable, False if unstable
        """
        if axis == 1:  # Minimum moment - stable
            return True
        elif axis == 2:  # Intermediate moment - unstable!
            return False
        elif axis == 3:  # Maximum moment - stable
            return True
        else:
            raise ValueError("axis must be 1, 2, or 3")


logger.info("Rigid body dynamics module loaded")
