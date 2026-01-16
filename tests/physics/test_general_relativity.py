"""
Comprehensive tests for General Relativity domain.
"""

import numpy as np
import pytest


class TestSchwarzschildMetric:
    """Tests for Schwarzschild black hole."""

    def test_schwarzschild_radius(self):
        """Event horizon rs = 2GM/c²."""
        from mechanics_dsl.domains.general_relativity import SOLAR_MASS, SchwarzschildMetric

        bh = SchwarzschildMetric(mass=SOLAR_MASS)
        rs = bh.schwarzschild_radius()

        # Solar mass: rs ≈ 3 km
        assert np.isclose(rs, 2954, rtol=0.01)

    def test_time_dilation(self):
        """Time runs slower near black hole."""
        from mechanics_dsl.domains.general_relativity import SchwarzschildMetric

        bh = SchwarzschildMetric(mass=1e30)
        rs = bh.schwarzschild_radius()

        # Far from horizon
        factor_far = bh.proper_time_factor(r=100 * rs)
        assert factor_far > 0.99

        # Close to horizon
        factor_near = bh.proper_time_factor(r=2 * rs)
        assert factor_near < factor_far

    def test_isco_is_3rs(self):
        """ISCO = 3 × Schwarzschild radius."""
        from mechanics_dsl.domains.general_relativity import SchwarzschildMetric

        bh = SchwarzschildMetric(mass=1e30)
        rs = bh.schwarzschild_radius()
        r_isco = bh.isco_radius()

        assert np.isclose(r_isco, 3 * rs)

    def test_photon_sphere(self):
        """Photon sphere at 1.5 × rs."""
        from mechanics_dsl.domains.general_relativity import SchwarzschildMetric

        bh = SchwarzschildMetric(mass=1e30)
        rs = bh.schwarzschild_radius()
        r_photon = bh.photon_sphere_radius()

        assert np.isclose(r_photon, 1.5 * rs)

    def test_hawking_temperature(self):
        """Hawking temperature inversely proportional to mass."""
        from mechanics_dsl.domains.general_relativity import SOLAR_MASS, SchwarzschildMetric

        bh1 = SchwarzschildMetric(mass=SOLAR_MASS)
        bh2 = SchwarzschildMetric(mass=2 * SOLAR_MASS)

        T1 = bh1.hawking_temperature()
        T2 = bh2.hawking_temperature()

        # T ∝ 1/M
        assert np.isclose(T1 / T2, 2.0, rtol=0.01)


class TestKerrMetric:
    """Tests for rotating black holes."""

    def test_schwarzschild_limit(self):
        """a=0 should give Schwarzschild horizon."""
        from mechanics_dsl.domains.general_relativity import KerrMetric, SchwarzschildMetric

        mass = 1e30
        kerr = KerrMetric(mass=mass, spin_parameter=0.0)
        schwarz = SchwarzschildMetric(mass=mass)

        r_kerr = kerr.outer_horizon()
        r_schwarz = schwarz.schwarzschild_radius()

        assert np.isclose(r_kerr, r_schwarz, rtol=0.01)

    def test_horizons_exist(self):
        """Outer horizon > inner horizon for a < M."""
        from mechanics_dsl.domains.general_relativity import KerrMetric

        kerr = KerrMetric(mass=1e30, spin_parameter=0.5)

        r_outer = kerr.outer_horizon()
        r_inner = kerr.inner_horizon()

        assert r_outer > r_inner > 0

    def test_prograde_isco_smaller(self):
        """Prograde ISCO < retrograde ISCO."""
        from mechanics_dsl.domains.general_relativity import KerrMetric

        kerr = KerrMetric(mass=1e30, spin_parameter=0.9)

        r_pro = kerr.isco_radius(prograde=True)
        r_retro = kerr.isco_radius(prograde=False)

        assert r_pro < r_retro


class TestGravitationalLensing:
    """Tests for gravitational lensing."""

    def test_sun_deflection(self):
        """Sun deflects light by 1.75 arcseconds."""
        from mechanics_dsl.domains.general_relativity import SOLAR_MASS, GravitationalLensing

        lens = GravitationalLensing(mass=SOLAR_MASS)
        R_sun = 6.96e8  # Solar radius

        alpha = lens.deflection_angle(impact_parameter=R_sun)
        alpha_arcsec = alpha * 206265  # Convert to arcseconds

        assert np.isclose(alpha_arcsec, 1.75, rtol=0.01)

    def test_magnification_at_einstein_radius(self):
        """At Einstein radius (u=1), magnification ≈ 1.34."""
        from mechanics_dsl.domains.general_relativity import GravitationalLensing

        lens = GravitationalLensing(mass=1e30)
        mu = lens.magnification(u=1.0)

        assert np.isclose(mu, 1.34, rtol=0.01)


class TestFLRWCosmology:
    """Tests for cosmological model."""

    def test_hubble_time(self):
        """Hubble time ≈ 14 Gyr for H0 = 70 km/s/Mpc."""
        from mechanics_dsl.domains.general_relativity import FLRWCosmology

        cosmos = FLRWCosmology(H0=70)
        t_H = cosmos.hubble_time()

        # Convert to Gyr
        t_Gyr = t_H / (365.25 * 24 * 3600 * 1e9)

        assert np.isclose(t_Gyr, 14, rtol=0.05)

    def test_flat_universe(self):
        """Ω_total = 1 for flat universe."""
        from mechanics_dsl.domains.general_relativity import FLRWCosmology

        cosmos = FLRWCosmology(Omega_m=0.3, Omega_Lambda=0.7)

        assert np.isclose(cosmos.Omega_k, 0.0)

    def test_accelerating_expansion(self):
        """q < 0 for ΛCDM universe."""
        from mechanics_dsl.domains.general_relativity import FLRWCosmology

        cosmos = FLRWCosmology(Omega_m=0.3, Omega_Lambda=0.7)
        q = cosmos.deceleration_parameter()

        assert q < 0  # Accelerating
