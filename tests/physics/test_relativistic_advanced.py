"""
Comprehensive tests for enhanced relativistic domain.
"""

import numpy as np


class TestSynchrotronRadiation:
    """Tests for SynchrotronRadiation class."""

    def test_radiated_power_positive(self):
        """Radiated power must be positive."""
        from mechanics_dsl.domains.relativistic import SynchrotronRadiation

        synch = SynchrotronRadiation(charge=1.6e-19, mass=9.1e-31)
        P = synch.radiated_power(gamma_factor=1000, radius=1.0)

        assert P > 0

    def test_power_scaling_gamma4(self):
        """P ∝ γ⁴ for highly relativistic particles."""
        from mechanics_dsl.domains.relativistic import SynchrotronRadiation

        synch = SynchrotronRadiation(charge=1.6e-19, mass=9.1e-31)

        P1 = synch.radiated_power(gamma_factor=100, radius=1.0)
        P2 = synch.radiated_power(gamma_factor=200, radius=1.0)

        # P2/P1 should be ≈ (200/100)⁴ = 16
        ratio = P2 / P1
        assert np.isclose(ratio, 16, rtol=0.1)

    def test_beaming_angle(self):
        """Beaming angle θ ≈ 1/γ."""
        from mechanics_dsl.domains.relativistic import SynchrotronRadiation

        synch = SynchrotronRadiation(charge=1.6e-19, mass=9.1e-31)
        theta = synch.characteristic_angle(gamma_factor=1000)

        assert np.isclose(theta, 0.001, rtol=0.01)


class TestThomasPrecession:
    """Tests for ThomasPrecession class."""

    def test_nonrelativistic_limit(self):
        """At v << c, Thomas precession is small."""
        from mechanics_dsl.domains.relativistic import ThomasPrecession

        thomas = ThomasPrecession()

        omega_T = thomas.precession_frequency(v=1000, a=10, theta_av=np.pi / 2)

        # Should be much smaller than orbital frequency
        assert omega_T < 1e-6

    def test_circular_motion_formula(self):
        """Ω_T = (γ-1) ω for circular motion."""
        from mechanics_dsl.domains.relativistic import ThomasPrecession

        thomas = ThomasPrecession(c=1.0)  # Natural units

        # At v = 0.6c, γ = 1.25
        omega_orbit = 10.0
        omega_T = thomas.circular_motion_precession(omega_orbit, v=0.6)

        gamma = 1.25
        expected = (gamma - 1) * omega_orbit

        assert np.isclose(omega_T, expected, rtol=0.01)


class TestTwinParadox:
    """Tests for TwinParadox class."""

    def test_traveler_ages_less(self):
        """Traveling twin ages less than stationary twin."""
        from mechanics_dsl.domains.relativistic import TwinParadox

        paradox = TwinParadox(c=1.0)  # Natural units
        result = paradox.constant_velocity(distance=4.0, speed=0.8)

        assert result["proper_time"] < result["earth_time"]
        assert result["age_difference"] > 0

    def test_gamma_factor(self):
        """Verify correct gamma calculation."""
        from mechanics_dsl.domains.relativistic import TwinParadox

        paradox = TwinParadox(c=1.0)
        result = paradox.constant_velocity(distance=4.0, speed=0.8)

        expected_gamma = 1.0 / np.sqrt(1 - 0.8**2)

        assert np.isclose(result["gamma"], expected_gamma)

    def test_constant_acceleration_ages_less(self):
        """Accelerated twin also ages less."""
        from mechanics_dsl.domains.relativistic import TwinParadox

        paradox = TwinParadox(c=1.0)
        result = paradox.constant_acceleration(distance=1.0, acceleration=1.0)

        assert result["proper_time"] < result["earth_time"]


class TestRelativisticFunctions:
    """Tests for relativistic convenience functions."""

    def test_proper_acceleration(self):
        """a_proper = γ³ × a_coord."""
        from mechanics_dsl.domains.relativistic import proper_acceleration

        # At v = 0.6c, γ = 1.25, γ³ ≈ 1.95
        a_coord = 10.0
        c = 1.0

        a_proper = proper_acceleration(a_coord, v=0.6, c=c)

        gamma = 1.0 / np.sqrt(1 - 0.6**2)
        expected = gamma**3 * a_coord

        assert np.isclose(a_proper, expected)

    def test_gravitational_redshift(self):
        """z = Δφ/c²."""
        from mechanics_dsl.domains.relativistic import gravitational_redshift

        # GPS satellite Δφ ≈ 5.3e5 m²/s²
        c = 299792458.0
        delta_phi = 5.3e5

        z = gravitational_redshift(delta_phi, c=c)

        # Should be ~5.9e-10
        expected = delta_phi / c**2
        assert np.isclose(z, expected)

    def test_kinetic_energy_consistency(self):
        """T = (γ-1)mc² should be consistent with v from T."""
        from mechanics_dsl.domains.relativistic import (
            relativistic_kinetic_energy,
            velocity_from_kinetic_energy,
        )

        mass = 1.0
        v = 0.8 * 299792458.0
        c = 299792458.0

        T = relativistic_kinetic_energy(mass, v, c)
        v_recovered = velocity_from_kinetic_energy(T, mass, c)

        assert np.isclose(v, v_recovered, rtol=1e-6)

    def test_energy_momentum_relation(self):
        """E² = (pc)² + (mc²)²."""
        from mechanics_dsl.domains.relativistic import momentum_from_energy

        mass = 1.0
        c = 1.0
        E = 2.0  # Total energy

        p = momentum_from_energy(E, mass, c)

        # Verify: E² = p² + m²
        E_check = np.sqrt(p**2 + mass**2)
        assert np.isclose(E, E_check)

    def test_mandelstam_s(self):
        """Mandelstam s = (p1 + p2)²."""
        from mechanics_dsl.domains.relativistic import FourVector, mandelstam_s

        # Two particles at rest with mass 1
        p1 = FourVector(ct=1.0, x=0.0, y=0.0, z=0.0)
        p2 = FourVector(ct=1.0, x=0.0, y=0.0, z=0.0)

        s = mandelstam_s(p1, p2)

        # s = (2mc)² = 4
        assert np.isclose(s, 4.0)


class TestDopplerEffect:
    """Tests for Doppler effect calculations."""

    def test_approaching_blueshift(self):
        """Approaching source is blueshifted."""
        from mechanics_dsl.domains.relativistic import DopplerEffect

        f_observed = DopplerEffect.longitudinal_frequency(
            f_source=1.0, v=0.5 * 299792458.0, approaching=True
        )

        assert f_observed > 1.0

    def test_receding_redshift(self):
        """Receding source is redshifted."""
        from mechanics_dsl.domains.relativistic import DopplerEffect

        f_observed = DopplerEffect.longitudinal_frequency(
            f_source=1.0, v=0.5 * 299792458.0, approaching=False
        )

        assert f_observed < 1.0
