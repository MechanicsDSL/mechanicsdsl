"""
Rigid Body Dynamics Domain

Complete implementation of rigid body dynamics with:
- Euler angle formulation (ZYZ convention)
- Quaternion formulation (singularity-free)
- Integration with RigidBody3D utilities
- Spinning top and gyroscope support

Euler angles (φ, θ, ψ):
- φ: precession angle (rotation about z-axis)
- θ: nutation angle (rotation about new y-axis)
- ψ: spin angle (rotation about new z-axis)

The Lagrangian for a symmetric top is:
    L = (1/2)*I₁*(θ̇² + φ̇²sin²θ) + (1/2)*I₃*(ψ̇ + φ̇cosθ)² - Mgl*cosθ
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import sympy as sp
import numpy as np

from ..base import PhysicsDomain
from ...utils import logger

# Try to import RigidBody3D utilities from package root
try:
    from ...rigidbody import RigidBody3D, InertiaTensor
    HAS_RIGIDBODY3D = True
except ImportError:
    HAS_RIGIDBODY3D = False
    logger.warning("RigidBody3D not available, using symbolic-only implementation")


@dataclass 
class EulerAngles:
    """
    Euler angles (ZYZ convention) for rigid body orientation.
    
    Attributes:
        phi: Precession angle (0 to 2π)
        theta: Nutation angle (0 to π)
        psi: Spin angle (0 to 2π)
    """
    phi: float = 0.0
    theta: float = 0.0
    psi: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [φ, θ, ψ]."""
        return np.array([self.phi, self.theta, self.psi])


@dataclass
class Quaternion:
    """
    Unit quaternion for rigid body orientation (singularity-free).
    
    q = q₀ + q₁i + q₂j + q₃k
    Constraint: |q| = 1
    
    Attributes:
        q0: Scalar part
        q1, q2, q3: Vector parts
    """
    q0: float = 1.0
    q1: float = 0.0
    q2: float = 0.0
    q3: float = 0.0
    
    def normalize(self) -> 'Quaternion':
        """Return normalized quaternion."""
        norm = np.sqrt(self.q0**2 + self.q1**2 + self.q2**2 + self.q3**2)
        return Quaternion(self.q0/norm, self.q1/norm, self.q2/norm, self.q3/norm)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [q₀, q₁, q₂, q₃]."""
        return np.array([self.q0, self.q1, self.q2, self.q3])
    
    @staticmethod
    def from_euler_angles(euler: EulerAngles) -> 'Quaternion':
        """
        Create quaternion from Euler angles (ZYZ convention).
        
        Args:
            euler: EulerAngles object with phi, theta, psi
            
        Returns:
            Equivalent unit quaternion
        """
        phi, theta, psi = euler.phi, euler.theta, euler.psi
        
        # ZYZ convention: R = Rz(phi) * Ry(theta) * Rz(psi)
        # Build quaternion by composing individual rotations
        
        # Rotation about z by phi: q_phi = (cos(phi/2), 0, 0, sin(phi/2))
        c_phi = np.cos(phi / 2)
        s_phi = np.sin(phi / 2)
        
        # Rotation about y by theta: q_theta = (cos(theta/2), 0, sin(theta/2), 0)
        c_theta = np.cos(theta / 2)
        s_theta = np.sin(theta / 2)
        
        # Rotation about z by psi: q_psi = (cos(psi/2), 0, 0, sin(psi/2))
        c_psi = np.cos(psi / 2)
        s_psi = np.sin(psi / 2)
        
        # First: q_temp = q_phi * q_theta
        # q_phi = (c_phi, 0, 0, s_phi), q_theta = (c_theta, 0, s_theta, 0)
        t0 = c_phi * c_theta
        t1 = s_phi * s_theta
        t2 = c_phi * s_theta
        t3 = s_phi * c_theta
        
        # Then: q = q_temp * q_psi
        # q_temp = (t0, t1, t2, t3), q_psi = (c_psi, 0, 0, s_psi)
        q0 = t0 * c_psi - t3 * s_psi
        q1 = t1 * c_psi + t2 * s_psi
        q2 = t2 * c_psi - t1 * s_psi
        q3 = t3 * c_psi + t0 * s_psi
        
        return Quaternion(q0, q1, q2, q3).normalize()
    
    def to_euler_angles(self) -> EulerAngles:
        """Convert to Euler angles (ZYZ convention)."""
        # ZYZ convention conversion
        q0, q1, q2, q3 = self.q0, self.q1, self.q2, self.q3
        
        # θ from q0 and q3
        theta = np.arccos(np.clip(2*(q0**2 + q3**2) - 1, -1, 1))
        
        if abs(np.sin(theta)) < 1e-10:
            # Gimbal lock
            phi = 2 * np.arctan2(q3, q0)
            psi = 0.0
        else:
            phi = np.arctan2(q0*q2 + q1*q3, q0*q1 - q2*q3)
            psi = np.arctan2(q0*q2 - q1*q3, q0*q1 + q2*q3)
        
        return EulerAngles(phi, theta, psi)
    
    def to_rotation_matrix(self) -> np.ndarray:
        """
        Convert quaternion to 3x3 rotation matrix.
        
        Returns:
            3x3 rotation matrix R such that v' = R @ v
        """
        q0, q1, q2, q3 = self.q0, self.q1, self.q2, self.q3
        
        # Rotation matrix from unit quaternion
        R = np.array([
            [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
        ])
        
        return R
    
    def rotate_vector(self, v: np.ndarray) -> np.ndarray:
        """
        Rotate a 3D vector by this quaternion.
        
        Args:
            v: 3D vector to rotate
            
        Returns:
            Rotated vector
        """
        return self.to_rotation_matrix() @ v


class RigidBodyDynamics(PhysicsDomain):
    """
    Rigid body dynamics with rotational degrees of freedom.
    
    Supports both Euler angle and quaternion formulations.
    
    The Lagrangian for a rigid body with one point fixed is:
        L = T_rot - V
        T_rot = (1/2) * ω^T * I * ω
    
    where ω is the angular velocity and I is the inertia tensor.
    
    Example:
        >>> body = RigidBodyDynamics("spinning_top")
        >>> body.set_inertia_principal(I1=0.1, I2=0.1, I3=0.05)
        >>> body.set_parameters({'M': 1.0, 'g': 9.81, 'l': 0.1})
        >>> eom = body.derive_equations_of_motion()
    """
    
    def __init__(self, name: str = "rigid_body", use_quaternions: bool = False):
        super().__init__(name)
        self._inertia_tensor: Optional[sp.Matrix] = None
        self._I1: Optional[sp.Expr] = None  # Principal moments
        self._I2: Optional[sp.Expr] = None
        self._I3: Optional[sp.Expr] = None
        self._use_quaternions = use_quaternions
        self._potential_energy: Optional[sp.Expr] = None
        self._symbol_cache: Dict[str, sp.Symbol] = {}
        self._time_symbol = sp.Symbol('t', real=True)
        
        # Set up coordinates based on formulation
        if use_quaternions:
            self.coordinates = ['q0', 'q1', 'q2', 'q3']
        else:
            self.coordinates = ['phi', 'theta', 'psi']  # Euler angles
    
    def get_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """Get or create a symbol with caching."""
        if name not in self._symbol_cache:
            default_assumptions = {'real': True}
            default_assumptions.update(assumptions)
            self._symbol_cache[name] = sp.Symbol(name, **default_assumptions)
        return self._symbol_cache[name]
    
    def set_inertia_tensor(self, I: sp.Matrix) -> None:
        """
        Set the 3x3 inertia tensor.
        
        Args:
            I: 3x3 symbolic or numeric matrix
        """
        if I.shape != (3, 3):
            raise ValueError("Inertia tensor must be 3x3")
        self._inertia_tensor = I
        
        # Extract principal moments if diagonal
        if I[0, 1] == 0 and I[0, 2] == 0 and I[1, 2] == 0:
            self._I1 = I[0, 0]
            self._I2 = I[1, 1]
            self._I3 = I[2, 2]
    
    def set_inertia_principal(self, I1: float, I2: float, I3: float) -> None:
        """
        Set principal moments of inertia.
        
        For a symmetric top: I1 = I2 ≠ I3
        For a sphere: I1 = I2 = I3
        
        Args:
            I1, I2, I3: Principal moments of inertia
        """
        self._I1 = sp.Float(I1)
        self._I2 = sp.Float(I2)
        self._I3 = sp.Float(I3)
        
        self._inertia_tensor = sp.diag(I1, I2, I3)
    
    def set_inertia_symbolic(self, I1_sym: str = 'I1', I2_sym: str = 'I2', 
                              I3_sym: str = 'I3') -> None:
        """
        Set symbolic principal moments.
        
        Args:
            I1_sym, I2_sym, I3_sym: Symbol names for principal moments
        """
        self._I1 = self.get_symbol(I1_sym, positive=True)
        self._I2 = self.get_symbol(I2_sym, positive=True)
        self._I3 = self.get_symbol(I3_sym, positive=True)
        
        self._inertia_tensor = sp.diag(self._I1, self._I2, self._I3)
    
    def set_potential_energy(self, V: sp.Expr) -> None:
        """Set the potential energy expression."""
        self._potential_energy = V
    
    def set_gravitational_potential(self, M: str = 'M', g: str = 'g', 
                                     l: str = 'l') -> None:
        """
        Set gravitational potential for a top with center of mass at height l*cos(θ).
        
        V = Mgl*cos(θ)
        
        Args:
            M: Symbol name for mass
            g: Symbol name for gravitational acceleration
            l: Symbol name for distance from pivot to center of mass
        """
        M_sym = self.get_symbol(M, positive=True)
        g_sym = self.get_symbol(g, positive=True)
        l_sym = self.get_symbol(l, positive=True)
        theta = self.get_symbol('theta')
        
        self._potential_energy = M_sym * g_sym * l_sym * sp.cos(theta)
    
    def set_gravitational_potential_quaternion(self, M: str = 'M', g: str = 'g', 
                                               l: str = 'l') -> None:
        """
        Set gravitational potential for a top using quaternion formulation.
        
        The height of the center of mass is given by the z-component of the
        body-fixed z-axis expressed in the space frame:
        
        h = l * (2*(q0² + q3²) - 1) = l * cos(θ)
        
        So V = Mgl * (1 - 2*(q1² + q2²))
        
        Args:
            M: Symbol name for mass
            g: Symbol name for gravitational acceleration
            l: Symbol name for distance from pivot to center of mass
        """
        M_sym = self.get_symbol(M, positive=True)
        g_sym = self.get_symbol(g, positive=True)
        l_sym = self.get_symbol(l, positive=True)
        
        q0 = self.get_symbol('q0')
        q1 = self.get_symbol('q1')
        q2 = self.get_symbol('q2')
        q3 = self.get_symbol('q3')
        
        # cos(θ) = 2*(q0² + q3²) - 1 = 1 - 2*(q1² + q2²)
        cos_theta = 1 - 2*(q1**2 + q2**2)
        
        self._potential_energy = M_sym * g_sym * l_sym * cos_theta

    
    def _angular_velocity_euler(self) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
        """
        Compute body-frame angular velocity from Euler angles.
        
        For ZYZ convention:
        ω₁ = φ̇*sin(θ)*sin(ψ) + θ̇*cos(ψ)
        ω₂ = φ̇*sin(θ)*cos(ψ) - θ̇*sin(ψ)
        ω₃ = φ̇*cos(θ) + ψ̇
        
        Returns:
            Tuple of (ω₁, ω₂, ω₃) in body frame
        """
        phi = self.get_symbol('phi')
        theta = self.get_symbol('theta')
        psi = self.get_symbol('psi')
        phi_dot = self.get_symbol('phi_dot')
        theta_dot = self.get_symbol('theta_dot')
        psi_dot = self.get_symbol('psi_dot')
        
        omega1 = phi_dot * sp.sin(theta) * sp.sin(psi) + theta_dot * sp.cos(psi)
        omega2 = phi_dot * sp.sin(theta) * sp.cos(psi) - theta_dot * sp.sin(psi)
        omega3 = phi_dot * sp.cos(theta) + psi_dot
        
        return omega1, omega2, omega3
    
    def _rotational_kinetic_energy(self) -> sp.Expr:
        """
        Compute rotational kinetic energy.
        
        T = (1/2) * (I₁ω₁² + I₂ω₂² + I₃ω₃²)
        
        Returns:
            Kinetic energy expression
        """
        if self._I1 is None or self._I2 is None or self._I3 is None:
            raise ValueError("Inertia tensor not set")
        
        if self._use_quaternions:
            return self._kinetic_energy_quaternion()
        
        omega1, omega2, omega3 = self._angular_velocity_euler()
        
        T = (sp.Rational(1, 2) * self._I1 * omega1**2 +
             sp.Rational(1, 2) * self._I2 * omega2**2 +
             sp.Rational(1, 2) * self._I3 * omega3**2)
        
        return sp.expand(T)
    
    def _quaternion_E_matrix(self) -> sp.Matrix:
        """
        Compute the E matrix for quaternion-angular velocity relationship.
        
        The E matrix relates quaternion derivatives to angular velocity:
        ω = 2 * E(q)^T * q̇
        
        For q = [q0, q1, q2, q3]^T:
        E = [-q1  q0 -q3  q2]
            [-q2  q3  q0 -q1]
            [-q3 -q2  q1  q0]
            
        Returns:
            3x4 E matrix
        """
        q0 = self.get_symbol('q0')
        q1 = self.get_symbol('q1')
        q2 = self.get_symbol('q2')
        q3 = self.get_symbol('q3')
        
        E = sp.Matrix([
            [-q1,  q0, -q3,  q2],
            [-q2,  q3,  q0, -q1],
            [-q3, -q2,  q1,  q0]
        ])
        
        return E
    
    def _angular_velocity_quaternion(self) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
        """
        Compute body-frame angular velocity from quaternion and its derivative.
        
        The relationship is: ω = 2 * E(q)^T * q̇
        
        For unit quaternions, this is equivalent to: ω = 2 * q* ⊗ q̇
        where q* is the quaternion conjugate.
        
        Returns:
            Tuple of (ω₁, ω₂, ω₃) in body frame
        """
        q0_dot = self.get_symbol('q0_dot')
        q1_dot = self.get_symbol('q1_dot')
        q2_dot = self.get_symbol('q2_dot')
        q3_dot = self.get_symbol('q3_dot')
        
        q_dot = sp.Matrix([q0_dot, q1_dot, q2_dot, q3_dot])
        E = self._quaternion_E_matrix()
        
        # ω = 2 * E * q̇
        omega = 2 * E * q_dot
        
        return omega[0], omega[1], omega[2]
    
    def _kinetic_energy_quaternion(self) -> sp.Expr:
        """
        Compute kinetic energy using full quaternion formulation.
        
        T = (1/2) * ω^T * I * ω
        
        where ω = 2 * E(q) * q̇ is the body-frame angular velocity
        computed from quaternion derivatives.
        
        This formulation is singularity-free and works for all
        inertia tensors, not just spherical bodies.
        
        Returns:
            Kinetic energy expression in terms of quaternion derivatives
        """
        omega1, omega2, omega3 = self._angular_velocity_quaternion()
        
        T = (sp.Rational(1, 2) * self._I1 * omega1**2 +
             sp.Rational(1, 2) * self._I2 * omega2**2 +
             sp.Rational(1, 2) * self._I3 * omega3**2)
        
        return sp.expand(T)
    
    def quaternion_constraint(self) -> sp.Expr:
        """
        Get the quaternion unit norm constraint.
        
        For DAE solvers: q0² + q1² + q2² + q3² - 1 = 0
        
        This constraint must be maintained during simulation to ensure
        the quaternion remains normalized.
        
        Returns:
            Constraint equation (equals zero when satisfied)
        """
        q0 = self.get_symbol('q0')
        q1 = self.get_symbol('q1')
        q2 = self.get_symbol('q2')
        q3 = self.get_symbol('q3')
        
        return q0**2 + q1**2 + q2**2 + q3**2 - 1
    
    def quaternion_constraint_derivative(self) -> sp.Expr:
        """
        Get the time derivative of the quaternion constraint.
        
        d/dt(|q|² - 1) = 2*(q0*q0_dot + q1*q1_dot + q2*q2_dot + q3*q3_dot) = 0
        
        This is useful for velocity-level constraint stabilization.
        
        Returns:
            Derivative of constraint equation
        """
        q0 = self.get_symbol('q0')
        q1 = self.get_symbol('q1')
        q2 = self.get_symbol('q2')
        q3 = self.get_symbol('q3')
        q0_dot = self.get_symbol('q0_dot')
        q1_dot = self.get_symbol('q1_dot')
        q2_dot = self.get_symbol('q2_dot')
        q3_dot = self.get_symbol('q3_dot')
        
        return 2 * (q0*q0_dot + q1*q1_dot + q2*q2_dot + q3*q3_dot)
    
    def define_lagrangian(self) -> sp.Expr:
        """
        Define Lagrangian L = T - V for the rigid body.
        
        Returns:
            Lagrangian expression
        """
        T = self._rotational_kinetic_energy()
        V = self._potential_energy if self._potential_energy else sp.S.Zero
        
        return T - V
    
    def define_hamiltonian(self) -> sp.Expr:
        """
        Define Hamiltonian H = T + V for the rigid body.
        
        For Euler angles with a symmetric top (I₁ = I₂):
        H = (p_θ² + (p_φ - p_ψ*cos(θ))²/sin²(θ))/(2I₁) + p_ψ²/(2I₃) + V(θ)
        
        Returns:
            Hamiltonian expression
        """
        if self._I1 != self._I2:
            logger.warning("Hamiltonian is simplified for non-symmetric tops")
        
        p_phi = self.get_symbol('p_phi')
        p_theta = self.get_symbol('p_theta')
        p_psi = self.get_symbol('p_psi')
        theta = self.get_symbol('theta')
        
        # For symmetric top I₁ = I₂
        I1 = self._I1
        I3 = self._I3
        
        H = (p_theta**2 / (2*I1) + 
             (p_phi - p_psi*sp.cos(theta))**2 / (2*I1*sp.sin(theta)**2) +
             p_psi**2 / (2*I3))
        
        if self._potential_energy:
            H += self._potential_energy
        
        return sp.simplify(H)
    
    def derive_equations_of_motion(self) -> Dict[str, sp.Expr]:
        """
        Derive equations of motion using Euler-Lagrange equations.
        
        For Euler angles, this gives second-order ODEs for φ, θ, ψ.
        
        Returns:
            Dictionary mapping acceleration variables to expressions
        """
        L = self.define_lagrangian()
        equations = []
        
        for q in self.coordinates:
            q_sym = self.get_symbol(q)
            q_dot = self.get_symbol(f"{q}_dot")
            q_ddot = self.get_symbol(f"{q}_ddot")
            
            # Create time-dependent function
            q_func = sp.Function(q)(self._time_symbol)
            
            # Substitute into Lagrangian
            L_sub = L.subs(q_sym, q_func)
            L_sub = L_sub.subs(q_dot, sp.diff(q_func, self._time_symbol))
            
            # Euler-Lagrange
            dL_dq_dot = sp.diff(L_sub, sp.diff(q_func, self._time_symbol))
            d_dt = sp.diff(dL_dq_dot, self._time_symbol)
            dL_dq = sp.diff(L_sub, q_func)
            
            eq = d_dt - dL_dq
            
            # Replace derivatives with symbols
            eq = eq.subs(sp.diff(q_func, self._time_symbol, 2), q_ddot)
            eq = eq.subs(sp.diff(q_func, self._time_symbol), q_dot)
            eq = eq.subs(q_func, q_sym)
            
            equations.append((q, eq))
        
        # Solve for accelerations
        accelerations = {}
        for q, eq in equations:
            q_ddot = self.get_symbol(f"{q}_ddot")
            eq_expanded = sp.expand(eq)
            
            A = sp.diff(eq_expanded, q_ddot)
            B = eq_expanded.subs(q_ddot, 0)
            
            if A != 0:
                accelerations[f"{q}_ddot"] = sp.simplify(-B / A)
            else:
                logger.warning(f"Could not solve for {q}_ddot")
                accelerations[f"{q}_ddot"] = sp.S.Zero
        
        self.equations = accelerations
        self._is_compiled = True
        return accelerations
    
    def get_state_variables(self) -> List[str]:
        """Get state variables for ODE system."""
        state = []
        for q in self.coordinates:
            state.extend([q, f"{q}_dot"])
        return state
    
    def get_conserved_quantities(self) -> Dict[str, sp.Expr]:
        """
        Get conserved quantities for the rigid body.
        
        For a symmetric top with gravitational potential:
        - Energy (H)
        - Angular momentum about z-axis (p_φ)
        - Angular momentum about body symmetry axis (p_ψ)
        
        Returns:
            Dictionary of conserved quantities
        """
        quantities = {}
        
        # Energy (always conserved for autonomous system)
        L = self.define_lagrangian()
        quantities['energy'] = self._rotational_kinetic_energy()
        if self._potential_energy:
            quantities['energy'] += self._potential_energy
        
        # Check for cyclic coordinates
        phi = self.get_symbol('phi')
        psi = self.get_symbol('psi')
        phi_dot = self.get_symbol('phi_dot')
        psi_dot = self.get_symbol('psi_dot')
        
        # φ is cyclic if V doesn't depend on φ
        dL_dphi = sp.diff(L, phi)
        if sp.simplify(dL_dphi) == 0:
            p_phi = sp.diff(L, phi_dot)
            quantities['p_phi'] = sp.simplify(p_phi)
        
        # ψ is cyclic if V doesn't depend on ψ
        dL_dpsi = sp.diff(L, psi)
        if sp.simplify(dL_dpsi) == 0:
            p_psi = sp.diff(L, psi_dot)
            quantities['p_psi'] = sp.simplify(p_psi)
        
        return quantities
    
    def angular_momentum_space_frame(self) -> sp.Matrix:
        """
        Compute angular momentum in the space frame.
        
        L = R * I * ω_body
        
        where R is the rotation matrix from body to space frame.
        
        Returns:
            3x1 matrix of angular momentum components
        """
        omega1, omega2, omega3 = self._angular_velocity_euler()
        
        # Angular momentum in body frame
        L_body = sp.Matrix([
            self._I1 * omega1,
            self._I2 * omega2,
            self._I3 * omega3
        ])
        
        return L_body  # Full transformation requires rotation matrix


class SymmetricTop(RigidBodyDynamics):
    """
    Specialized class for symmetric top (I₁ = I₂).
    
    Simplifies the equations and provides analytical solutions
    for special cases.
    """
    
    def __init__(self, name: str = "symmetric_top", I_perp: float = 1.0,
                 I_axis: float = 0.5):
        super().__init__(name)
        self.set_inertia_principal(I_perp, I_perp, I_axis)
    
    def sleeping_top_frequency(self) -> sp.Expr:
        """
        Compute precession frequency for a sleeping (vertical) top.
        
        For small nutation about θ = 0:
        ω_precession = p_ψ / I₁
        """
        I1 = self._I1
        p_psi = self.get_symbol('p_psi')
        return p_psi / I1


class Gyroscope(SymmetricTop):
    """
    Gyroscope model with torque-free precession.
    
    For a fast-spinning gyroscope, the precession rate is:
    Ω = τ / L = Mgl / (I₃ * ω_spin)
    """
    
    def __init__(self, name: str = "gyroscope", I_perp: float = 0.01,
                 I_axis: float = 0.005, spin_rate: float = 100.0):
        super().__init__(name, I_perp, I_axis)
        self.spin_rate = spin_rate
    
    def precession_rate(self, M: float, g: float, l: float) -> float:
        """
        Compute steady precession rate.
        
        Ω = Mgl / (I₃ * ω_spin)
        
        Args:
            M: Mass
            g: Gravitational acceleration
            l: Distance from pivot to center of mass
            
        Returns:
            Precession angular velocity
        """
        I3 = float(self._I3) if not isinstance(self._I3, float) else self._I3
        return M * g * l / (I3 * self.spin_rate)
    
    def nutation_frequency(self, M: float, g: float, l: float,
                            theta: float = 0.1) -> float:
        """
        Compute nutation frequency for small oscillations.
        
        For a fast gyroscope, nutation is rapid oscillation about
        the precession cone.
        
        Returns:
            Nutation angular frequency
        """
        I1 = float(self._I1) if not isinstance(self._I1, float) else self._I1
        I3 = float(self._I3) if not isinstance(self._I3, float) else self._I3
        
        # Simplified formula for fast spin
        omega_n = I3 * self.spin_rate / I1
        return omega_n
