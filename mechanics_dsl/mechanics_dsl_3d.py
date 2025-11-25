"""
3D Motion and Advanced Physics Extensions for MechanicsDSL v0.5.0

This module provides:
- True 3D motion with Euler angles and quaternions
- Non-conservative forces (friction, damping, air drag)
- Non-holonomic constraints
"""

import numpy as np
import sympy as sp
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

# ============================================================================
# 3D ROTATION REPRESENTATIONS
# ============================================================================

class EulerAngles:
    """
    Euler angles representation for 3D rotations.
    
    Uses ZYX convention (yaw-pitch-roll):
    - phi (φ): rotation about z-axis (yaw)
    - theta (θ): rotation about y-axis (pitch)  
    - psi (ψ): rotation about x-axis (roll)
    """
    
    @staticmethod
    def rotation_matrix(phi: sp.Symbol, theta: sp.Symbol, psi: sp.Symbol) -> sp.Matrix:
        """
        Compute rotation matrix from Euler angles (ZYX convention).
        
        Args:
            phi: Yaw angle (rotation about z)
            theta: Pitch angle (rotation about y)
            psi: Roll angle (rotation about x)
            
        Returns:
            3x3 rotation matrix
        """
        # Rotation about z-axis (yaw)
        Rz = sp.Matrix([
            [sp.cos(phi), -sp.sin(phi), 0],
            [sp.sin(phi), sp.cos(phi), 0],
            [0, 0, 1]
        ])
        
        # Rotation about y-axis (pitch)
        Ry = sp.Matrix([
            [sp.cos(theta), 0, sp.sin(theta)],
            [0, 1, 0],
            [-sp.sin(theta), 0, sp.cos(theta)]
        ])
        
        # Rotation about x-axis (roll)
        Rx = sp.Matrix([
            [1, 0, 0],
            [0, sp.cos(psi), -sp.sin(psi)],
            [0, sp.sin(psi), sp.cos(psi)]
        ])
        
        # Combined rotation: R = Rz * Ry * Rx
        return Rz * Ry * Rx
    
    @staticmethod
    def angular_velocity(phi: sp.Symbol, theta: sp.Symbol, psi: sp.Symbol,
                        phi_dot: sp.Symbol, theta_dot: sp.Symbol, psi_dot: sp.Symbol) -> sp.Matrix:
        """
        Compute angular velocity vector in body frame from Euler angles.
        
        Args:
            phi, theta, psi: Euler angles
            phi_dot, theta_dot, psi_dot: Time derivatives
            
        Returns:
            3x1 angular velocity vector [ωx, ωy, ωz]
        """
        # Angular velocity in body frame
        omega_x = phi_dot * sp.sin(theta) * sp.sin(psi) + theta_dot * sp.cos(psi)
        omega_y = phi_dot * sp.sin(theta) * sp.cos(psi) - theta_dot * sp.sin(psi)
        omega_z = phi_dot * sp.cos(theta) + psi_dot
        
        return sp.Matrix([omega_x, omega_y, omega_z])


class Quaternion:
    """
    Quaternion representation for 3D rotations.
    
    Quaternion: q = w + xi + yj + zk
    where w is the scalar part and (x, y, z) is the vector part.
    """
    
    @staticmethod
    def from_euler(phi: float, theta: float, psi: float) -> np.ndarray:
        """
        Convert Euler angles to quaternion.
        
        Args:
            phi: Yaw angle
            theta: Pitch angle
            psi: Roll angle
            
        Returns:
            Quaternion [w, x, y, z]
        """
        cy = np.cos(phi * 0.5)
        sy = np.sin(phi * 0.5)
        cp = np.cos(theta * 0.5)
        sp = np.sin(theta * 0.5)
        cr = np.cos(psi * 0.5)
        sr = np.sin(psi * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])
    
    @staticmethod
    def rotation_matrix(q: np.ndarray) -> np.ndarray:
        """
        Compute rotation matrix from quaternion.
        
        Args:
            q: Quaternion [w, x, y, z]
            
        Returns:
            3x3 rotation matrix
        """
        w, x, y, z = q
        
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
        ])


# ============================================================================
# NON-CONSERVATIVE FORCES
# ============================================================================

class NonConservativeForces:
    """Handle non-conservative forces (friction, damping, air drag)"""
    
    @staticmethod
    def friction_force(normal_force: sp.Expr, mu: sp.Symbol, 
                      velocity: sp.Expr) -> sp.Expr:
        """
        Compute friction force.
        
        Args:
            normal_force: Normal force magnitude
            mu: Coefficient of friction
            velocity: Velocity vector
            
        Returns:
            Friction force (opposes motion)
        """
        # Friction opposes velocity direction
        v_mag = sp.sqrt(velocity.dot(velocity))
        if v_mag == 0:
            return sp.Matrix([0, 0, 0])
        
        # Direction opposite to velocity
        v_hat = velocity / v_mag
        return -mu * normal_force * v_hat
    
    @staticmethod
    def damping_force(velocity: sp.Expr, damping_coeff: sp.Symbol) -> sp.Expr:
        """
        Compute linear damping force.
        
        Args:
            velocity: Velocity vector
            damping_coeff: Damping coefficient
            
        Returns:
            Damping force F = -c * v
        """
        return -damping_coeff * velocity
    
    @staticmethod
    def air_drag(velocity: sp.Expr, drag_coeff: sp.Symbol, 
                rho: sp.Symbol = None, area: sp.Symbol = None) -> sp.Expr:
        """
        Compute air drag force (quadratic in velocity).
        
        Args:
            velocity: Velocity vector
            drag_coeff: Drag coefficient (or Cd * rho * A / 2)
            rho: Air density (optional)
            area: Cross-sectional area (optional)
            
        Returns:
            Drag force F = -0.5 * Cd * rho * A * |v| * v
        """
        v_mag = sp.sqrt(velocity.dot(velocity))
        if v_mag == 0:
            return sp.Matrix([0, 0, 0])
        
        # Drag opposes velocity
        if rho is not None and area is not None:
            # Full form: F = -0.5 * Cd * rho * A * |v| * v
            drag = -0.5 * drag_coeff * rho * area * v_mag * velocity
        else:
            # Simplified: F = -C * |v| * v (where C includes all constants)
            drag = -drag_coeff * v_mag * velocity
        
        return drag


# ============================================================================
# NON-HOLONOMIC CONSTRAINTS
# ============================================================================

class NonHolonomicConstraints:
    """
    Handle non-holonomic constraints (velocity-dependent).
    
    Non-holonomic constraints have the form:
    Σ a_i(q) * q̇_i + b(q) = 0
    
    Common examples:
    - Rolling without slipping
    - Skidding constraints
    """
    
    @staticmethod
    def rolling_without_slipping(radius: sp.Symbol, 
                                 position: sp.Expr,
                                 angular_velocity: sp.Expr) -> sp.Expr:
        """
        Constraint for rolling without slipping.
        
        For a wheel/ball: v = r * ω
        
        Args:
            radius: Radius of rolling object
            position: Position vector
            angular_velocity: Angular velocity vector
            
        Returns:
            Constraint expression: v - r*ω = 0
        """
        # Linear velocity from position derivative
        # This is a simplified version - full implementation would
        # compute velocity from position derivatives
        return position - radius * angular_velocity
    
    @staticmethod
    def add_to_lagrangian(lagrangian: sp.Expr, 
                         constraints: List[sp.Expr],
                         multipliers: List[sp.Symbol]) -> sp.Expr:
        """
        Add non-holonomic constraints to Lagrangian using multipliers.
        
        For non-holonomic constraints, we use the method of Lagrange
        multipliers but with velocity-dependent constraint equations.
        
        Args:
            lagrangian: Original Lagrangian
            constraints: List of constraint expressions
            multipliers: List of Lagrange multipliers
            
        Returns:
            Augmented Lagrangian
        """
        augmented = lagrangian
        for constraint, multiplier in zip(constraints, multipliers):
            augmented += multiplier * constraint
        return augmented

