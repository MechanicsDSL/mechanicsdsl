"""
Tests for relativistic mechanics domain.
"""

import numpy as np
import pytest


class TestRelativisticParticle:
    """Test RelativisticParticle dynamics."""

    def test_creation(self):
        """Test basic creation."""
        from mechanics_dsl.domains.relativistic import RelativisticParticle

        particle = RelativisticParticle(mass=1.0)
        assert particle.rest_mass == 1.0
        assert particle.coordinates == ["x", "y", "z"]

    def test_lorentz_factor_zero_velocity(self):
        """Test γ = 1 at rest."""
        from mechanics_dsl.domains.relativistic import RelativisticParticle

        particle = RelativisticParticle()
        particle.set_parameter("c", 1.0)

        gamma = particle.lorentz_factor(0.0)
        assert gamma == 1.0

    def test_lorentz_factor_half_c(self):
        """Test γ at v = 0.5c."""
        from mechanics_dsl.domains.relativistic import RelativisticParticle

        particle = RelativisticParticle()
        particle.set_parameter("c", 1.0)

        gamma = particle.lorentz_factor(0.5)
        expected = 1.0 / np.sqrt(1 - 0.25)  # 1/√0.75 ≈ 1.1547
        assert np.isclose(gamma, expected)

    def test_lorentz_factor_exceeds_c(self):
        """Test that v >= c raises error."""
        from mechanics_dsl.domains.relativistic import RelativisticParticle

        particle = RelativisticParticle()
        particle.set_parameter("c", 1.0)

        with pytest.raises(ValueError):
            particle.lorentz_factor(1.0)

        with pytest.raises(ValueError):
            particle.lorentz_factor(1.5)

    def test_rest_energy(self):
        """Test E₀ = mc²."""
        from mechanics_dsl.domains.relativistic import RelativisticParticle

        particle = RelativisticParticle(mass=2.0)
        particle.set_parameter("c", 3.0)

        E0 = particle.rest_energy()
        assert E0 == 2.0 * 9.0  # mc² = 2 * 9 = 18

    def test_relativistic_momentum(self):
        """Test p = γmv."""
        from mechanics_dsl.domains.relativistic import RelativisticParticle

        particle = RelativisticParticle(mass=1.0)
        particle.set_parameter("c", 1.0)

        p = particle.relativistic_momentum(0.6)
        gamma = 1.0 / np.sqrt(1 - 0.36)  # 1.25
        expected = gamma * 1.0 * 0.6
        assert np.isclose(p, expected)

    def test_kinetic_energy(self):
        """Test T = (γ-1)mc²."""
        from mechanics_dsl.domains.relativistic import RelativisticParticle

        particle = RelativisticParticle(mass=1.0)
        particle.set_parameter("c", 1.0)

        T = particle.kinetic_energy(0.8)
        gamma = 1.0 / np.sqrt(1 - 0.64)  # 5/3 ≈ 1.667
        expected = (gamma - 1) * 1.0 * 1.0
        assert np.isclose(T, expected)


class TestFourVector:
    """Test FourVector operations."""

    def test_invariant_timelike(self):
        """Test timelike interval."""
        from mechanics_dsl.domains.relativistic import FourVector

        # Event at (ct=5, x=3, y=0, z=0)
        fv = FourVector(ct=5.0, x=3.0, y=0.0, z=0.0)
        s2 = fv.invariant()

        # s² = 25 - 9 = 16 > 0 (timelike)
        assert s2 == 16.0
        assert fv.is_timelike()

    def test_invariant_spacelike(self):
        """Test spacelike interval."""
        from mechanics_dsl.domains.relativistic import FourVector

        fv = FourVector(ct=3.0, x=5.0, y=0.0, z=0.0)
        s2 = fv.invariant()

        # s² = 9 - 25 = -16 < 0 (spacelike)
        assert s2 == -16.0
        assert fv.is_spacelike()

    def test_invariant_lightlike(self):
        """Test lightlike (null) interval."""
        from mechanics_dsl.domains.relativistic import FourVector

        fv = FourVector(ct=5.0, x=3.0, y=4.0, z=0.0)
        s2 = fv.invariant()

        # s² = 25 - 9 - 16 = 0 (lightlike)
        assert np.isclose(s2, 0.0)
        assert fv.is_lightlike()


class TestLorentzTransform:
    """Test Lorentz transformations."""

    def test_boost_at_rest(self):
        """Test that v=0 boost is identity."""
        from mechanics_dsl.domains.relativistic import FourVector, LorentzTransform

        fv = FourVector(ct=10.0, x=5.0, y=3.0, z=1.0)
        boosted = LorentzTransform.boost_x(fv, v=0.0, c=1.0)

        assert np.isclose(boosted.ct, 10.0)
        assert np.isclose(boosted.x, 5.0)

    def test_velocity_addition(self):
        """Test relativistic velocity addition."""
        from mechanics_dsl.domains.relativistic import LorentzTransform

        # Two velocities 0.5c each
        u = LorentzTransform.velocity_addition(0.5, 0.5, c=1.0)

        # Relativistic: (0.5 + 0.5)/(1 + 0.25) = 0.8
        assert np.isclose(u, 0.8)

        # Not Galilean: 0.5 + 0.5 = 1.0
        assert u < 1.0

    def test_velocity_addition_light_speed(self):
        """Test that c + anything = c."""
        from mechanics_dsl.domains.relativistic import LorentzTransform

        u = LorentzTransform.velocity_addition(1.0, 0.5, c=1.0)
        assert np.isclose(u, 1.0)

    def test_time_dilation(self):
        """Test time dilation."""
        from mechanics_dsl.domains.relativistic import LorentzTransform

        # Proper time 1s, v = 0.6c, γ = 1.25
        delta_t = LorentzTransform.time_dilation(1.0, 0.6, c=1.0)
        gamma = 1.0 / np.sqrt(1 - 0.36)
        assert np.isclose(delta_t, gamma)

    def test_length_contraction(self):
        """Test length contraction."""
        from mechanics_dsl.domains.relativistic import LorentzTransform

        # Proper length 1m, v = 0.8c, γ = 5/3
        L = LorentzTransform.length_contraction(1.0, 0.8, c=1.0)
        gamma = 1.0 / np.sqrt(1 - 0.64)
        assert np.isclose(L, 1.0 / gamma)


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_gamma_function(self):
        """Test standalone gamma function."""
        from mechanics_dsl.domains.relativistic import gamma

        g = gamma(0.8, c=1.0)
        expected = 1.0 / np.sqrt(1 - 0.64)
        assert np.isclose(g, expected)

    def test_beta_function(self):
        """Test beta function."""
        from mechanics_dsl.domains.relativistic import beta

        b = beta(0.6, c=1.0)
        assert b == 0.6

    def test_rapidity(self):
        """Test rapidity calculation."""
        from mechanics_dsl.domains.relativistic import rapidity

        eta = rapidity(0.6, c=1.0)
        expected = np.arctanh(0.6)
        assert np.isclose(eta, expected)
