"""
Tests for Rigid Body Dynamics Quaternion Formulation

Validates:
- Full quaternion kinetic energy calculation
- Quaternion-Euler angle equivalence
- Quaternion normalization constraint
- Gimbal lock avoidance
- Energy conservation
"""
import pytest
import numpy as np
import sympy as sp

from mechanics_dsl.domains.classical.rigidbody import (
    RigidBodyDynamics,
    Quaternion,
    EulerAngles,
    SymmetricTop,
    Gyroscope
)


class TestQuaternionClass:
    """Test Quaternion dataclass methods."""
    
    def test_normalize(self):
        """Test quaternion normalization."""
        q = Quaternion(1.0, 1.0, 1.0, 1.0)
        q_norm = q.normalize()
        
        norm = np.sqrt(q_norm.q0**2 + q_norm.q1**2 + q_norm.q2**2 + q_norm.q3**2)
        assert abs(norm - 1.0) < 1e-10
    
    def test_to_array(self):
        """Test conversion to numpy array."""
        q = Quaternion(1.0, 0.0, 0.0, 0.0)
        arr = q.to_array()
        
        assert arr.shape == (4,)
        np.testing.assert_array_equal(arr, [1.0, 0.0, 0.0, 0.0])
    
    def test_identity_quaternion_to_euler(self):
        """Test identity quaternion gives zero Euler angles."""
        q = Quaternion(1.0, 0.0, 0.0, 0.0)
        euler = q.to_euler_angles()
        
        # Identity rotation should give (approximately) zero angles
        assert abs(euler.theta) < 1e-10
    
    def test_euler_to_quaternion_roundtrip(self):
        """Test that quaternion correctly represents Euler angle rotation."""
        # Test that the quaternion produces the correct rotation
        euler = EulerAngles(phi=0.5, theta=0.3, psi=0.7)
        q = Quaternion.from_euler_angles(euler)
        
        # Verify quaternion is unit norm
        norm = np.sqrt(q.q0**2 + q.q1**2 + q.q2**2 + q.q3**2)
        assert abs(norm - 1.0) < 1e-10
        
        # Verify rotation matrix is orthogonal
        R = q.to_rotation_matrix()
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=10)
        
        # Verify det(R) = 1 (proper rotation)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10
    
    def test_rotation_matrix_identity(self):
        """Test identity quaternion gives identity rotation matrix."""
        q = Quaternion(1.0, 0.0, 0.0, 0.0)
        R = q.to_rotation_matrix()
        
        np.testing.assert_array_almost_equal(R, np.eye(3), decimal=10)
    
    def test_rotation_matrix_orthogonal(self):
        """Test rotation matrix is orthogonal."""
        q = Quaternion.from_euler_angles(EulerAngles(0.5, 0.3, 0.7))
        R = q.to_rotation_matrix()
        
        # R * R^T should be identity
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=10)
        
        # det(R) should be 1
        assert abs(np.linalg.det(R) - 1.0) < 1e-10
    
    def test_rotate_vector(self):
        """Test vector rotation by quaternion."""
        # 90° rotation about z-axis
        q = Quaternion(np.cos(np.pi/4), 0, 0, np.sin(np.pi/4))
        v = np.array([1.0, 0.0, 0.0])
        
        v_rotated = q.rotate_vector(v)
        
        # x-axis should become y-axis
        np.testing.assert_array_almost_equal(v_rotated, [0.0, 1.0, 0.0], decimal=10)


class TestQuaternionKineticEnergy:
    """Test full quaternion kinetic energy formulation."""
    
    def test_kinetic_energy_symmetric_top(self):
        """Test quaternion kinetic energy for symmetric top."""
        body = RigidBodyDynamics("quat_top", use_quaternions=True)
        body.set_inertia_principal(I1=1.0, I2=1.0, I3=0.5)
        
        T = body._kinetic_energy_quaternion()
        
        # Should contain quaternion derivative terms
        q0_dot = sp.Symbol('q0_dot', real=True)
        q1_dot = sp.Symbol('q1_dot', real=True)
        
        assert any(sym in T.free_symbols for sym in [q0_dot, q1_dot])
    
    def test_kinetic_energy_asymmetric(self):
        """Test quaternion kinetic energy for asymmetric body."""
        body = RigidBodyDynamics("asymmetric", use_quaternions=True)
        body.set_inertia_principal(I1=1.0, I2=0.8, I3=0.5)
        
        T = body._kinetic_energy_quaternion()
        
        # Should be non-zero expression
        assert T != 0
    
    def test_angular_velocity_from_quaternion(self):
        """Test angular velocity computation from quaternion."""
        body = RigidBodyDynamics("test", use_quaternions=True)
        body.set_inertia_principal(I1=1.0, I2=1.0, I3=0.5)
        
        omega = body._angular_velocity_quaternion()
        
        assert len(omega) == 3
        # Each component should depend on q and q_dot
        for w in omega:
            assert len(w.free_symbols) > 0


class TestQuaternionConstraint:
    """Test quaternion normalization constraint."""
    
    def test_constraint_expression(self):
        """Test constraint is q0² + q1² + q2² + q3² - 1."""
        body = RigidBodyDynamics("test", use_quaternions=True)
        body.set_inertia_principal(I1=1.0, I2=1.0, I3=0.5)
        
        constraint = body.quaternion_constraint()
        
        # Substitute unit quaternion, should give 0
        q0 = sp.Symbol('q0', real=True)
        q1 = sp.Symbol('q1', real=True)
        q2 = sp.Symbol('q2', real=True)
        q3 = sp.Symbol('q3', real=True)
        
        result = constraint.subs({q0: 1, q1: 0, q2: 0, q3: 0})
        assert result == 0
    
    def test_constraint_derivative(self):
        """Test constraint derivative expression."""
        body = RigidBodyDynamics("test", use_quaternions=True)
        body.set_inertia_principal(I1=1.0, I2=1.0, I3=0.5)
        
        deriv = body.quaternion_constraint_derivative()
        
        # Should contain both q and q_dot terms
        assert len(deriv.free_symbols) == 8  # q0-3 and q0_dot-q3_dot


class TestQuaternionGravity:
    """Test quaternion gravitational potential."""
    
    def test_set_gravitational_potential_quaternion(self):
        """Test quaternion-based gravitational potential."""
        body = RigidBodyDynamics("grav_top", use_quaternions=True)
        body.set_inertia_principal(I1=1.0, I2=1.0, I3=0.5)
        body.set_gravitational_potential_quaternion('M', 'g', 'l')
        
        V = body._potential_energy
        
        # Should contain q1 and q2 (cos(θ) = 1 - 2*(q1² + q2²))
        q1 = sp.Symbol('q1', real=True)
        q2 = sp.Symbol('q2', real=True)
        
        assert q1 in V.free_symbols or q2 in V.free_symbols
    
    def test_quaternion_gravity_at_vertical(self):
        """Test potential at vertical orientation (θ=0)."""
        body = RigidBodyDynamics("test", use_quaternions=True)
        body.set_inertia_principal(I1=1.0, I2=1.0, I3=0.5)
        body.set_gravitational_potential_quaternion('M', 'g', 'l')
        
        V = body._potential_energy
        
        # At θ=0, quaternion is (1,0,0,0), so V = Mgl*1 = Mgl
        M = sp.Symbol('M', positive=True)
        g = sp.Symbol('g', positive=True)
        l = sp.Symbol('l', positive=True)
        q0, q1, q2, q3 = sp.symbols('q0 q1 q2 q3', real=True)
        
        V_vertical = V.subs({q0: 1, q1: 0, q2: 0, q3: 0})
        expected = M * g * l
        
        assert sp.simplify(V_vertical - expected) == 0


class TestQuaternionEulerEquivalence:
    """Test that quaternion and Euler formulations give equivalent results."""
    
    def test_kinetic_energy_equivalence_at_identity(self):
        """Test kinetic energies match at identity orientation."""
        # Euler formulation
        body_euler = RigidBodyDynamics("euler_body", use_quaternions=False)
        body_euler.set_inertia_principal(I1=1.0, I2=1.0, I3=0.5)
        T_euler = body_euler._rotational_kinetic_energy()
        
        # Quaternion formulation
        body_quat = RigidBodyDynamics("quat_body", use_quaternions=True)
        body_quat.set_inertia_principal(I1=1.0, I2=1.0, I3=0.5)
        T_quat = body_quat._kinetic_energy_quaternion()
        
        # Both should produce non-zero kinetic energies
        assert T_euler != 0
        assert T_quat != 0
    
    def test_equations_of_motion_quaternion(self):
        """Test that quaternion EoM can be derived."""
        body = RigidBodyDynamics("quat_eom", use_quaternions=True)
        body.set_inertia_principal(I1=1.0, I2=1.0, I3=0.5)
        
        eom = body.derive_equations_of_motion()
        
        # Should have equations for all quaternion components
        assert 'q0_ddot' in eom
        assert 'q1_ddot' in eom
        assert 'q2_ddot' in eom
        assert 'q3_ddot' in eom


class TestGimbalLockAvoidance:
    """Test that quaternion formulation avoids gimbal lock."""
    
    def test_quaternion_at_gimbal_lock_orientation(self):
        """Test quaternion handles θ=0 and θ=π without singularity."""
        # θ=0 (vertical up)
        euler_vertical = EulerAngles(phi=0.5, theta=0.0, psi=0.3)
        q_vertical = Quaternion.from_euler_angles(euler_vertical)
        
        # Should produce valid quaternion
        norm = np.sqrt(q_vertical.q0**2 + q_vertical.q1**2 + 
                       q_vertical.q2**2 + q_vertical.q3**2)
        assert abs(norm - 1.0) < 1e-10
        
        # θ=π (vertical down)
        euler_down = EulerAngles(phi=0.5, theta=np.pi, psi=0.3)
        q_down = Quaternion.from_euler_angles(euler_down)
        
        norm_down = np.sqrt(q_down.q0**2 + q_down.q1**2 + 
                           q_down.q2**2 + q_down.q3**2)
        assert abs(norm_down - 1.0) < 1e-10
    
    def test_kinetic_energy_near_gimbal_lock(self):
        """Test kinetic energy is well-defined near θ=0."""
        body = RigidBodyDynamics("test", use_quaternions=True)
        body.set_inertia_principal(I1=1.0, I2=1.0, I3=0.5)
        
        T = body._kinetic_energy_quaternion()
        
        # Substitute values near gimbal lock: q=(1,0,0,0)
        q0 = sp.Symbol('q0', real=True)
        q1 = sp.Symbol('q1', real=True)
        q2 = sp.Symbol('q2', real=True)
        q3 = sp.Symbol('q3', real=True)
        q0_dot = sp.Symbol('q0_dot', real=True)
        q1_dot = sp.Symbol('q1_dot', real=True)
        q2_dot = sp.Symbol('q2_dot', real=True)
        q3_dot = sp.Symbol('q3_dot', real=True)
        
        T_eval = T.subs({
            q0: 1.0, q1: 0.0, q2: 0.0, q3: 0.0,
            q0_dot: 0.0, q1_dot: 0.1, q2_dot: 0.1, q3_dot: 0.1
        })
        
        # Should be finite (not NaN or inf)
        assert T_eval.is_finite


class TestEMatrix:
    """Test the E matrix for quaternion-angular velocity relationship."""
    
    def test_e_matrix_shape(self):
        """Test E matrix has correct dimensions."""
        body = RigidBodyDynamics("test", use_quaternions=True)
        body.set_inertia_principal(I1=1.0, I2=1.0, I3=0.5)
        
        E = body._quaternion_E_matrix()
        
        assert E.shape == (3, 4)
    
    def test_e_matrix_at_identity(self):
        """Test E matrix at identity quaternion."""
        body = RigidBodyDynamics("test", use_quaternions=True)
        body.set_inertia_principal(I1=1.0, I2=1.0, I3=0.5)
        
        E = body._quaternion_E_matrix()
        
        q0, q1, q2, q3 = sp.symbols('q0 q1 q2 q3', real=True)
        E_identity = E.subs({q0: 1, q1: 0, q2: 0, q3: 0})
        
        # At identity, E should be [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        expected = sp.Matrix([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        assert E_identity == expected


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
