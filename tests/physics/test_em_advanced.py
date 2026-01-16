"""
Comprehensive tests for enhanced electromagnetic domain.
"""

import numpy as np
import pytest


class TestElectromagneticWave:
    """Tests for ElectromagneticWave class."""

    def test_vacuum_speed_of_light(self):
        """Wave in vacuum travels at c."""
        from mechanics_dsl.domains.electromagnetic import ElectromagneticWave

        wave = ElectromagneticWave(frequency=1e9)
        v = wave.phase_velocity()

        assert np.isclose(v, 299792458.0, rtol=1e-6)

    def test_wavelength_frequency_relation(self):
        """c = f × λ."""
        from mechanics_dsl.domains.electromagnetic import ElectromagneticWave

        wave = ElectromagneticWave(frequency=1e9)  # 1 GHz
        wavelength = wave.wavelength()

        # Should be ~30 cm
        assert np.isclose(wavelength, 0.3, rtol=0.01)

    def test_refractive_index(self):
        """Refractive index n = √(εᵣμᵣ)."""
        from mechanics_dsl.domains.electromagnetic import ElectromagneticWave

        # Wave in glass (n ≈ 1.5, so εᵣ ≈ 2.25)
        wave = ElectromagneticWave(frequency=1e14, medium_epsilon_r=2.25)
        n = wave.refractive_index()

        assert np.isclose(n, 1.5)

    def test_impedance_free_space(self):
        """Free space impedance ≈ 377 Ω."""
        from mechanics_dsl.domains.electromagnetic import ElectromagneticWave

        wave = ElectromagneticWave(frequency=1e9)
        eta = wave.impedance()

        assert np.isclose(eta, 377, rtol=0.01)

    def test_intensity_positive(self):
        """Intensity must be positive."""
        from mechanics_dsl.domains.electromagnetic import ElectromagneticWave

        wave = ElectromagneticWave(frequency=1e9, amplitude_E=100.0)
        I = wave.intensity()

        assert I > 0


class TestAntenna:
    """Tests for Antenna class."""

    def test_half_wave_dipole_resistance(self):
        """Half-wave dipole R_rad ≈ 73 Ω."""
        from mechanics_dsl.domains.electromagnetic import Antenna

        antenna = Antenna.half_wave_dipole(frequency=1e9)

        assert np.isclose(antenna.R_rad, 73.1, rtol=0.01)

    def test_hertzian_dipole_scaling(self):
        """Radiation resistance ∝ (l/λ)²."""
        from mechanics_dsl.domains.electromagnetic import Antenna

        # Double the frequency = half the wavelength = 4x the resistance
        a1 = Antenna.hertzian_dipole(length=0.01, frequency=1e9)
        a2 = Antenna.hertzian_dipole(length=0.01, frequency=2e9)

        ratio = a2.R_rad / a1.R_rad
        assert np.isclose(ratio, 4.0, rtol=0.1)

    def test_radiated_power(self):
        """P_rad = (1/2) I² R_rad."""
        from mechanics_dsl.domains.electromagnetic import Antenna

        antenna = Antenna.half_wave_dipole(frequency=1e9)
        P = antenna.radiated_power(current=1.0)

        expected = 0.5 * 1.0**2 * 73.1
        assert np.isclose(P, expected, rtol=0.01)


class TestWaveguide:
    """Tests for Waveguide class."""

    def test_te10_cutoff(self):
        """TE₁₀ cutoff: f_c = c/(2a)."""
        from mechanics_dsl.domains.electromagnetic import Waveguide

        wg = Waveguide(a=0.0229, b=0.0102)  # WR-90
        fc = wg.cutoff_frequency(m=1, n=0)

        # Should be ~6.56 GHz for WR-90
        assert 6e9 < fc < 7e9

    def test_group_phase_velocity_product(self):
        """v_g × v_p = c² (for lossless waveguide)."""
        from mechanics_dsl.domains.electromagnetic import Waveguide

        wg = Waveguide(a=0.023, b=0.010)
        f_op = 10e9  # Operating frequency

        v_g = wg.group_velocity(f_op, m=1, n=0)
        v_p = wg.phase_velocity(f_op, m=1, n=0)
        c = 299792458.0

        assert np.isclose(v_g * v_p, c**2, rtol=0.01)

    def test_below_cutoff_no_propagation(self):
        """Below cutoff: v_g = 0."""
        from mechanics_dsl.domains.electromagnetic import Waveguide

        wg = Waveguide(a=0.023, b=0.010)
        fc = wg.cutoff_frequency(1, 0)

        v_g = wg.group_velocity(fc * 0.5, m=1, n=0)

        assert v_g == 0.0


class TestSkinEffect:
    """Tests for SkinEffect class."""

    def test_copper_skin_depth(self):
        """Copper at 1 MHz: δ ≈ 66 μm."""
        from mechanics_dsl.domains.electromagnetic import SkinEffect

        copper = SkinEffect(conductivity=5.8e7)
        delta = copper.skin_depth(frequency=1e6)

        # Should be approximately 66 μm
        assert np.isclose(delta, 66e-6, rtol=0.1)

    def test_skin_depth_frequency_scaling(self):
        """δ ∝ 1/√f."""
        from mechanics_dsl.domains.electromagnetic import SkinEffect

        skin = SkinEffect(conductivity=5.8e7)

        delta_1 = skin.skin_depth(1e6)
        delta_4 = skin.skin_depth(4e6)

        # 4x frequency = 1/2 skin depth
        ratio = delta_1 / delta_4
        assert np.isclose(ratio, 2.0, rtol=0.01)


class TestEMConvenienceFunctions:
    """Tests for EM convenience functions."""

    def test_plasma_frequency(self):
        """Plasma frequency scales with density."""
        from mechanics_dsl.domains.electromagnetic import plasma_frequency

        # Ionosphere: n_e ~ 10^12 m⁻³ → f_p ~ 10 MHz
        f_p = plasma_frequency(n_e=1e12)

        assert 1e6 < f_p < 1e8

    def test_debye_length(self):
        """Debye length calculation."""
        from mechanics_dsl.domains.electromagnetic import debye_length

        # Typical plasma parameters
        lambda_D = debye_length(n_e=1e19, T_e=10000)

        assert lambda_D > 0
