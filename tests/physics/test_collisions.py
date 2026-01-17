"""
Tests for Collision Dynamics Module
"""

import numpy as np
import pytest

from mechanics_dsl.domains.classical import (
    CollisionSolver,
    Particle,
    SymbolicCollisionSolver,
    elastic_collision_1d,
    inelastic_collision_1d,
    perfectly_inelastic_1d,
)


class TestParticle:
    """Test Particle class."""

    def test_create_particle(self):
        """Test creating a particle."""
        p = Particle(mass=1.0, position=np.array([0, 0, 0]), velocity=np.array([1, 0, 0]))

        assert p.mass == 1.0
        assert np.allclose(p.velocity, [1, 0, 0])

    def test_momentum(self):
        """Test momentum calculation."""
        p = Particle(mass=2.0, position=np.zeros(3), velocity=np.array([3, 0, 0]))

        assert np.allclose(p.momentum, [6, 0, 0])

    def test_kinetic_energy(self):
        """Test kinetic energy calculation."""
        p = Particle(mass=2.0, position=np.zeros(3), velocity=np.array([3, 0, 0]))

        # KE = 0.5 * 2 * 9 = 9
        assert np.isclose(p.kinetic_energy, 9.0)


class TestCollisionSolver:
    """Test CollisionSolver class."""

    def test_elastic_collision_equal_masses(self):
        """Test elastic collision of equal masses - velocities exchange."""
        solver = CollisionSolver()

        # Use solve_1d for reliable 1D collision
        v1f, v2f = solver.solve_1d(1.0, 1.0, 1.0, 0.0, e=1.0)

        # Equal masses: velocities should exchange
        assert np.isclose(v1f, 0.0, atol=1e-10)
        assert np.isclose(v2f, 1.0, atol=1e-10)

    def test_perfectly_inelastic(self):
        """Test perfectly inelastic collision."""
        solver = CollisionSolver()

        # Use solve_1d for reliable 1D collision
        v1f, v2f = solver.solve_1d(1.0, 2.0, 1.0, 0.0, e=0.0)

        # Both should have same velocity (momentum conserved)
        assert np.isclose(v1f, v2f)
        assert np.isclose(v1f, 1.0)  # (1*2 + 1*0)/(1+1) = 1

    def test_solve_1d(self):
        """Test 1D collision solver."""
        solver = CollisionSolver()

        v1f, v2f = solver.solve_1d(1.0, 2.0, 1.0, 0.0, e=1.0)

        # Equal masses, elastic: velocities exchange
        assert np.isclose(v1f, 0.0)
        assert np.isclose(v2f, 2.0)

    def test_center_of_mass_frame(self):
        """Test CM frame transformation."""
        solver = CollisionSolver()

        p1 = Particle(mass=1.0, position=np.zeros(3), velocity=np.array([2.0, 0.0, 0.0]))
        p2 = Particle(mass=1.0, position=np.ones(3), velocity=np.array([0.0, 0.0, 0.0]))

        v1_cm, v2_cm = solver.center_of_mass_frame(p1, p2)

        # In CM frame, momenta are equal and opposite
        assert np.allclose(p1.mass * v1_cm + p2.mass * v2_cm, 0.0)

    def test_reduced_mass(self):
        """Test reduced mass calculation."""
        solver = CollisionSolver()

        mu = solver.reduced_mass(1.0, 1.0)
        assert np.isclose(mu, 0.5)

        mu = solver.reduced_mass(2.0, 2.0)
        assert np.isclose(mu, 1.0)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_elastic_collision_1d(self):
        """Test elastic_collision_1d function."""
        v1f, v2f = elastic_collision_1d(1.0, 1.0, 1.0, 0.0)

        assert np.isclose(v1f, 0.0)
        assert np.isclose(v2f, 1.0)

    def test_inelastic_collision_1d(self):
        """Test inelastic_collision_1d function."""
        v1f, v2f = inelastic_collision_1d(1.0, 2.0, 1.0, 0.0, e=0.5)

        # Momentum conserved: m1*v1 + m2*v2 = m1*v1f + m2*v2f
        p_initial = 1.0 * 2.0 + 1.0 * 0.0
        p_final = 1.0 * v1f + 1.0 * v2f
        assert np.isclose(p_initial, p_final)

    def test_perfectly_inelastic_1d(self):
        """Test perfectly_inelastic_1d function."""
        vf = perfectly_inelastic_1d(1.0, 2.0, 3.0, 0.0)

        # (1*2 + 3*0) / (1+3) = 0.5
        assert np.isclose(vf, 0.5)


class TestSymbolicCollisionSolver:
    """Test symbolic collision solver."""

    def test_solve_1d_elastic(self):
        """Test symbolic elastic collision formulas."""
        solver = SymbolicCollisionSolver()

        result = solver.solve_1d_elastic()

        assert "v1_final" in result
        assert "v2_final" in result

    def test_energy_loss_formula(self):
        """Test symbolic energy loss formula."""
        solver = SymbolicCollisionSolver()

        delta_KE = solver.energy_loss()

        # Should be proportional to (1 - e^2)
        import sympy as sp

        e = sp.Symbol("e", positive=True)

        assert delta_KE.has(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
