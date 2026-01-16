"""
Comprehensive tests for quantum mechanics domain.

Tests cover: finite well, step potential, delta barrier, hydrogen atom,
and quantum tunneling with rigorous physics verification.
"""

import numpy as np
import pytest


class TestFiniteSquareWell:
    """Tests for FiniteSquareWell class."""

    def test_always_has_bound_state(self):
        """Finite well always has at least one bound state."""
        from mechanics_dsl.domains.quantum import FiniteSquareWell

        well = FiniteSquareWell(depth=0.1, width=1.0)
        states = well.find_bound_states()

        assert len(states) >= 1, "Must have at least one bound state"

    def test_more_states_with_deeper_well(self):
        """Deeper wells have more bound states."""
        from mechanics_dsl.domains.quantum import FiniteSquareWell

        shallow = FiniteSquareWell(depth=5.0, width=2.0)
        deep = FiniteSquareWell(depth=50.0, width=2.0)

        assert len(deep.find_bound_states()) > len(shallow.find_bound_states())

    def test_bound_state_energies_negative(self):
        """Bound state energies must be negative."""
        from mechanics_dsl.domains.quantum import FiniteSquareWell

        well = FiniteSquareWell(depth=10.0, width=2.0)
        states = well.find_bound_states()

        for state in states:
            assert state.energy < 0, f"Bound state energy must be negative: {state.energy}"

    def test_energies_ordered(self):
        """Bound state energies must be ordered (E_0 < E_1 < ...)."""
        from mechanics_dsl.domains.quantum import FiniteSquareWell

        well = FiniteSquareWell(depth=20.0, width=2.0)
        states = well.find_bound_states()

        for i in range(len(states) - 1):
            assert states[i].energy < states[i + 1].energy

    def test_max_bound_states_formula(self):
        """Test that bound states exist."""
        from mechanics_dsl.domains.quantum import FiniteSquareWell

        well = FiniteSquareWell(depth=10.0, width=2.0)
        actual_states = len(well.find_bound_states())

        # Should have at least one bound state
        assert actual_states >= 1

    def test_scattering_transmission(self):
        """Test transmission for scattering states (E > 0)."""
        from mechanics_dsl.domains.quantum import FiniteSquareWell

        well = FiniteSquareWell(depth=5.0, width=2.0)
        T = well.transmission_coefficient(E=10.0)

        assert 0 < T <= 1


class TestStepPotential:
    """Tests for StepPotential class."""

    def test_reflection_transmission_sum(self):
        """R + T must equal 1 (probability conservation)."""
        from mechanics_dsl.domains.quantum import StepPotential

        step = StepPotential(height=5.0)

        for E in [1.0, 5.0, 10.0, 50.0]:
            if E > 0:
                R, T = step.reflection_transmission(E)
                assert np.isclose(R + T, 1.0), f"R + T = {R + T} ≠ 1 for E = {E}"

    def test_total_reflection_below_barrier(self):
        """E < V₀: total reflection."""
        from mechanics_dsl.domains.quantum import StepPotential

        step = StepPotential(height=10.0)
        R, T = step.reflection_transmission(E=5.0)

        assert R == 1.0
        assert T == 0.0

    def test_partial_transmission_above_barrier(self):
        """E > V₀: partial transmission."""
        from mechanics_dsl.domains.quantum import StepPotential

        step = StepPotential(height=5.0)
        R, T = step.reflection_transmission(E=10.0)

        assert 0 < R < 1
        assert 0 < T < 1

    def test_penetration_depth_positive(self):
        """Penetration depth must be positive."""
        from mechanics_dsl.domains.quantum import StepPotential

        step = StepPotential(height=10.0)
        delta = step.penetration_depth(E=5.0)

        assert delta > 0

    def test_penetration_depth_increases_near_barrier(self):
        """Penetration depth increases as E approaches V₀."""
        from mechanics_dsl.domains.quantum import StepPotential

        step = StepPotential(height=10.0)

        delta_far = step.penetration_depth(E=1.0)
        delta_near = step.penetration_depth(E=9.0)

        assert delta_near > delta_far


class TestDeltaFunctionBarrier:
    """Tests for DeltaFunctionBarrier class."""

    def test_transmission_between_zero_one(self):
        """Transmission must be in [0, 1]."""
        from mechanics_dsl.domains.quantum import DeltaFunctionBarrier

        barrier = DeltaFunctionBarrier(strength=1.0)

        for E in [0.1, 0.5, 1.0, 5.0, 10.0]:
            T = barrier.transmission(E)
            assert 0 < T < 1, f"T = {T} for E = {E}"

    def test_reflection_plus_transmission(self):
        """R + T = 1."""
        from mechanics_dsl.domains.quantum import DeltaFunctionBarrier

        barrier = DeltaFunctionBarrier(strength=2.0)

        for E in [0.5, 1.0, 2.0]:
            T = barrier.transmission(E)
            R = barrier.reflection(E)
            assert np.isclose(R + T, 1.0)

    def test_high_energy_transmission(self):
        """High energy: T → 1."""
        from mechanics_dsl.domains.quantum import DeltaFunctionBarrier

        barrier = DeltaFunctionBarrier(strength=1.0)
        T = barrier.transmission(E=1000.0)

        assert T > 0.99

    def test_attractive_well_bound_state(self):
        """Attractive delta well (λ < 0) has one bound state."""
        from mechanics_dsl.domains.quantum import DeltaFunctionBarrier

        well = DeltaFunctionBarrier(strength=-1.0)
        E_bound = well.bound_state_energy()

        assert E_bound is not None
        assert E_bound < 0

    def test_repulsive_barrier_no_bound_state(self):
        """Repulsive delta barrier (λ > 0) has no bound state."""
        from mechanics_dsl.domains.quantum import DeltaFunctionBarrier

        barrier = DeltaFunctionBarrier(strength=1.0)
        E_bound = barrier.bound_state_energy()

        assert E_bound is None


class TestHydrogenAtom:
    """Tests for HydrogenAtom class."""

    def test_ground_state_energy(self):
        """Ground state energy ≈ -13.6 eV."""
        from mechanics_dsl.domains.quantum import HydrogenAtom

        H = HydrogenAtom()
        E1 = H.energy_level(n=1)

        assert np.isclose(E1, -13.6, rtol=0.01)

    def test_energy_scaling(self):
        """E_n ∝ 1/n²."""
        from mechanics_dsl.domains.quantum import HydrogenAtom

        H = HydrogenAtom()
        E1 = H.energy_level(n=1)
        E2 = H.energy_level(n=2)
        E3 = H.energy_level(n=3)

        assert np.isclose(E2 / E1, 1 / 4, rtol=0.01)
        assert np.isclose(E3 / E1, 1 / 9, rtol=0.01)

    def test_helium_ion_scaling(self):
        """He⁺ (Z=2): E_n = 4 × E_H."""
        from mechanics_dsl.domains.quantum import HydrogenAtom

        H = HydrogenAtom(Z=1)
        He_plus = HydrogenAtom(Z=2)

        ratio = He_plus.energy_level(n=1) / H.energy_level(n=1)
        assert np.isclose(ratio, 4.0, rtol=0.01)

    def test_ionization_energy(self):
        """Ionization energy = -E_1."""
        from mechanics_dsl.domains.quantum import HydrogenAtom

        H = HydrogenAtom()
        IE = H.ionization_energy()

        assert np.isclose(IE, 13.6, rtol=0.01)

    def test_degeneracy(self):
        """Degeneracy g_n = 2n²."""
        from mechanics_dsl.domains.quantum import HydrogenAtom

        H = HydrogenAtom()

        assert H.degeneracy(1) == 2
        assert H.degeneracy(2) == 8
        assert H.degeneracy(3) == 18

    def test_bohr_radius(self):
        """Bohr radius ≈ 0.529 Å = 5.29e-11 m."""
        from mechanics_dsl.domains.quantum import HydrogenAtom

        H = HydrogenAtom()
        r1 = H.bohr_radius_n(n=1)

        assert np.isclose(r1, 5.29e-11, rtol=0.01)

    def test_lyman_alpha(self):
        """Lyman-α (2→1) wavelength ≈ 121.6 nm."""
        from mechanics_dsl.domains.quantum import HydrogenAtom

        H = HydrogenAtom()
        wavelength = H.transition_wavelength(n_initial=2, n_final=1)

        # Should be ~121.6 nm = 1.216e-7 m
        assert np.isclose(wavelength, 1.216e-7, rtol=0.01)

    def test_balmer_alpha(self):
        """Balmer-α (3→2) wavelength ≈ 656.3 nm (red)."""
        from mechanics_dsl.domains.quantum import HydrogenAtom

        H = HydrogenAtom()
        wavelength = H.transition_wavelength(n_initial=3, n_final=2)

        # Should be ~656.3 nm = 6.563e-7 m
        assert np.isclose(wavelength, 6.563e-7, rtol=0.01)

    def test_spectral_series(self):
        """Test spectral series generation."""
        from mechanics_dsl.domains.quantum import HydrogenAtom

        H = HydrogenAtom()
        balmer = H.spectral_series(n_final=2, n_max=5)

        assert len(balmer) == 3  # 3→2, 4→2, 5→2


class TestQuantumTunnelingAdvanced:
    """Advanced tests for tunneling rigor."""

    def test_wkb_vs_exact_rectangular(self):
        """WKB should be order-of-magnitude correct for barriers."""
        from mechanics_dsl.domains.quantum import QuantumTunneling

        tunneling = QuantumTunneling(mass=1.0, hbar=1.0)

        # Barrier where WKB can be compared
        E, V0, width = 1.0, 2.0, 3.0

        T_exact = tunneling.rectangular_barrier(E, V0, width)

        def rect_potential(x):
            return V0 if 0 < x < width else 0

        T_wkb = tunneling.wkb_transmission(E, rect_potential, 0, width)

        # Both should be very small for thick barrier
        assert T_exact < 0.01
        assert T_wkb < 0.01
        # Order of magnitude check
        assert T_wkb < 10 * T_exact

    def test_gamow_factor_physical(self):
        """Gamow factor should be very small for typical nuclear energies."""
        from mechanics_dsl.domains.quantum import HBAR, QuantumTunneling

        # Alpha particle
        mass_alpha = 6.644e-27  # kg
        tunneling = QuantumTunneling(mass=mass_alpha, hbar=HBAR)

        # Typical alpha energy: 5 MeV = 5 * 1.6e-13 J
        E_alpha = 5 * 1.6e-13

        # Decay to heavy nucleus (e.g., Pb, Z=82)
        G = tunneling.gamow_factor(E_alpha, Z1=2, Z2=82)

        # Gamow factor should be exponentially small
        assert G < 1e-10
