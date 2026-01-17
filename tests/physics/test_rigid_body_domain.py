"""
Tests for Rigid Body Dynamics Domain Module

Validates:
- Euler angle formulation
- Inertia tensor handling
- Lagrangian/Hamiltonian derivation
- Conserved quantities (angular momentum)
"""

import pytest
import sympy as sp

from mechanics_dsl.domains.classical import Gyroscope, RigidBodyDynamics, SymmetricTop


class TestRigidBodyDynamics:
    """Test basic rigid body dynamics."""

    def test_inertia_principal(self):
        """Test setting principal moments of inertia."""
        body = RigidBodyDynamics("test_body")
        body.set_inertia_principal(I1=1.0, I2=1.0, I3=0.5)

        assert body._I1 == 1.0
        assert body._I2 == 1.0
        assert body._I3 == 0.5

        # Inertia tensor should be diagonal
        I = body._inertia_tensor
        assert I[0, 0] == 1.0
        assert I[1, 1] == 1.0
        assert I[2, 2] == 0.5
        assert I[0, 1] == 0

    def test_symbolic_inertia(self):
        """Test symbolic principal moments."""
        body = RigidBodyDynamics("test_body")
        body.set_inertia_symbolic("I1", "I2", "I3")

        I1 = sp.Symbol("I1", positive=True)
        I2 = sp.Symbol("I2", positive=True)
        I3 = sp.Symbol("I3", positive=True)

        assert body._I1 == I1
        assert body._I2 == I2
        assert body._I3 == I3

    def test_euler_angle_coordinates(self):
        """Test Euler angle coordinate setup."""
        body = RigidBodyDynamics("test_body", use_quaternions=False)

        coords = body.coordinates

        assert "phi" in coords
        assert "theta" in coords
        assert "psi" in coords

    def test_quaternion_coordinates(self):
        """Test quaternion coordinate setup."""
        body = RigidBodyDynamics("test_body", use_quaternions=True)

        coords = body.coordinates

        assert "q0" in coords
        assert "q1" in coords
        assert "q2" in coords
        assert "q3" in coords


class TestLagrangianFormulation:
    """Test Lagrangian formulation for rigid body."""

    def test_kinetic_energy_symmetric_top(self):
        """Test rotational kinetic energy for symmetric top."""
        body = RigidBodyDynamics("symmetric_top")
        body.set_inertia_symbolic("I1", "I1", "I3")  # I1 = I2

        T = body._rotational_kinetic_energy()

        # T should contain angular velocity terms
        phi_dot = sp.Symbol("phi_dot", real=True)
        theta_dot = sp.Symbol("theta_dot", real=True)
        psi_dot = sp.Symbol("psi_dot", real=True)

        # Should have velocity terms
        assert any(sym in T.free_symbols for sym in [phi_dot, theta_dot, psi_dot])

    def test_lagrangian_with_gravity(self):
        """Test Lagrangian with gravitational potential."""
        body = RigidBodyDynamics("gravitational_top")
        body.set_inertia_principal(I1=0.1, I2=0.1, I3=0.05)
        body.set_gravitational_potential("M", "g", "l")

        L = body.define_lagrangian()

        # L should be non-trivial
        assert L != 0

        # Should contain gravitational parameters
        theta = sp.Symbol("theta", real=True)
        assert theta in L.free_symbols or sp.cos(theta) in L.atoms(sp.cos)

    def test_equations_of_motion(self):
        """Test derivation of equations of motion."""
        body = RigidBodyDynamics("test_eom")
        body.set_inertia_principal(I1=1.0, I2=1.0, I3=0.5)

        eom = body.derive_equations_of_motion()

        # Should have equations for all Euler angles
        assert "phi_ddot" in eom
        assert "theta_ddot" in eom
        assert "psi_ddot" in eom


class TestSymmetricTop:
    """Test symmetric top specialization."""

    def test_symmetric_inertia(self):
        """Test that I1 = I2 for symmetric top."""
        top = SymmetricTop("my_top", I_perp=0.1, I_axis=0.05)

        assert top._I1 == top._I2
        assert top._I1 != top._I3

    def test_sleeping_top_frequency(self):
        """Test sleeping top precession formula."""
        top = SymmetricTop("test", I_perp=1.0, I_axis=0.5)

        omega_prec = top.sleeping_top_frequency()

        # Should depend on p_psi and I1
        p_psi = sp.Symbol("p_psi", real=True)
        I1 = sp.Float(1.0)

        expected = p_psi / I1
        assert sp.simplify(omega_prec - expected) == 0


class TestGyroscope:
    """Test gyroscope model."""

    def test_precession_rate(self):
        """Test gyroscope precession rate formula."""
        gyro = Gyroscope(I_perp=0.01, I_axis=0.005, spin_rate=100.0)

        # Ω = Mgl / (I₃ * ω_spin)
        M, g, l = 1.0, 10.0, 0.1

        omega_prec = gyro.precession_rate(M, g, l)

        expected = M * g * l / (0.005 * 100.0)

        assert abs(omega_prec - expected) < 1e-10

    def test_nutation_frequency(self):
        """Test gyroscope nutation frequency."""
        gyro = Gyroscope(I_perp=0.01, I_axis=0.005, spin_rate=100.0)

        M, g, l = 1.0, 10.0, 0.1
        omega_n = gyro.nutation_frequency(M, g, l)

        # Should be approximately I₃*ω_spin/I₁
        expected = 0.005 * 100.0 / 0.01

        assert abs(omega_n - expected) < 1e-10


class TestConservedQuantities:
    """Test conserved quantities in rigid body dynamics."""

    def test_free_top_conserved(self):
        """Test conserved quantities for torque-free top."""
        body = RigidBodyDynamics("free_top")
        body.set_inertia_principal(I1=1.0, I2=1.0, I3=0.5)
        # No potential = torque-free

        quantities = body.get_conserved_quantities()

        # Should have energy
        assert "energy" in quantities

        # For symmetric top, should have p_phi and p_psi
        assert "p_psi" in quantities  # ψ is cyclic

    def test_gravitational_top_conserved(self):
        """Test conserved quantities with gravity."""
        body = RigidBodyDynamics("gravitational_top")
        body.set_inertia_principal(I1=1.0, I2=1.0, I3=0.5)
        body.set_gravitational_potential("M", "g", "l")

        quantities = body.get_conserved_quantities()

        # Energy should be conserved
        assert "energy" in quantities

        # p_phi and p_psi should still be conserved
        # (gravity V = Mgl*cos(θ) doesn't depend on φ or ψ)
        assert "p_phi" in quantities
        assert "p_psi" in quantities


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
