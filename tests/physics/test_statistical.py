"""
Comprehensive tests for Statistical Mechanics domain.
"""

import numpy as np


class TestBoltzmannDistribution:
    """Tests for Boltzmann distribution."""

    def test_equipartition(self):
        """At high T, probability ratio approaches 1."""
        from mechanics_dsl.domains.statistical import BoltzmannDistribution

        # At high T, small energy differences have similar probability
        boltz = BoltzmannDistribution(temperature=300)

        # Energy difference of kT
        E1 = 0.0
        E2 = 1.38e-23 * 300  # kT at 300K

        ratio = boltz.probability_ratio(E1=E1, E2=E2)

        # exp(-1) ≈ 0.37
        assert np.isclose(ratio, np.exp(1), rtol=0.01)

    def test_ground_state_dominates_low_T(self):
        """At low T, ground state dominates."""
        from mechanics_dsl.domains.statistical import BoltzmannDistribution

        boltz = BoltzmannDistribution(temperature=1.0)

        # E=0 vs E=100 (in units of k_B)
        ratio = boltz.probability_ratio(E1=0.0, E2=100 * 1.38e-23)

        assert ratio > 1e10  # Ground state much more probable

    def test_maxwell_speed_normalization(self):
        """Maxwell distribution integrates to 1."""
        from scipy.integrate import quad

        from mechanics_dsl.domains.statistical import BoltzmannDistribution

        boltz = BoltzmannDistribution(temperature=300)
        m = 4.65e-26  # N2 mass

        def f(v):
            return boltz.maxwell_speed_distribution(np.array([v]), m)[0]

        integral, _ = quad(f, 0, 3000)

        assert np.isclose(integral, 1.0, rtol=0.05)

    def test_rms_greater_than_mean(self):
        """v_rms > <v> > v_p."""
        from mechanics_dsl.domains.statistical import BoltzmannDistribution

        boltz = BoltzmannDistribution(temperature=300)
        m = 4.65e-26

        v_p = boltz.most_probable_speed(m)
        v_mean = boltz.mean_speed(m)
        v_rms = boltz.rms_speed(m)

        assert v_rms > v_mean > v_p


class TestIdealGas:
    """Tests for ideal gas."""

    def test_ideal_gas_law(self):
        """PV = nRT."""
        from mechanics_dsl.domains.statistical import IdealGas

        gas = IdealGas(n_moles=1.0, temperature=273.15, volume=0.0224)
        P = gas.pressure()

        # Should be ~1 atm = 101325 Pa
        assert np.isclose(P, 101325, rtol=0.01)

    def test_internal_energy_monoatomic(self):
        """U = (3/2)NkT for monoatomic."""
        from mechanics_dsl.domains.statistical import AVOGADRO_NUMBER, BOLTZMANN_CONSTANT, IdealGas

        gas = IdealGas(n_moles=1.0, temperature=300)
        U = gas.internal_energy(degrees_of_freedom=3)

        expected = 1.5 * AVOGADRO_NUMBER * BOLTZMANN_CONSTANT * 300

        assert np.isclose(U, expected, rtol=0.01)

    def test_cp_minus_cv(self):
        """C_P - C_V = Nk."""
        from mechanics_dsl.domains.statistical import AVOGADRO_NUMBER, BOLTZMANN_CONSTANT, IdealGas

        gas = IdealGas(n_moles=1.0, temperature=300)

        C_V = gas.heat_capacity_V()
        C_P = gas.heat_capacity_P()

        expected_diff = AVOGADRO_NUMBER * BOLTZMANN_CONSTANT

        assert np.isclose(C_P - C_V, expected_diff, rtol=0.01)


class TestIsingModel:
    """Tests for Ising model."""

    def test_ordered_state_energy(self):
        """All spins aligned gives minimum energy."""
        from mechanics_dsl.domains.statistical import IsingModel

        ising = IsingModel(L=10, J=1.0, dimension=2)
        ising.initialize_ordered(up=True)

        E_ordered = ising.energy()

        ising.initialize_random()
        E_random = ising.energy()

        assert E_ordered <= E_random

    def test_magnetization_ordered(self):
        """Ordered state has maximum magnetization."""
        from mechanics_dsl.domains.statistical import IsingModel

        ising = IsingModel(L=10, dimension=2)
        ising.initialize_ordered(up=True)

        M = ising.magnetization_density()

        assert M == 1.0

    def test_critical_temperature_2d(self):
        """2D Ising T_c ≈ 2.269."""
        from mechanics_dsl.domains.statistical import IsingModel

        ising = IsingModel(L=10, J=1.0, dimension=2)
        T_c = ising.critical_temperature_2d()

        assert np.isclose(T_c, 2.269, rtol=0.01)


class TestQuantumStatistics:
    """Tests for Fermi-Dirac and Bose-Einstein."""

    def test_fermi_occupation_at_mu(self):
        """At E = μ, f = 0.5."""
        from mechanics_dsl.domains.statistical import FermiDirac

        fd = FermiDirac(temperature=300, chemical_potential=1e-19)

        f = fd.occupation(energy=1e-19)

        assert np.isclose(f, 0.5)

    def test_bose_condensation(self):
        """BE occupation increases as energy approaches μ."""
        from mechanics_dsl.domains.statistical import BOLTZMANN_CONSTANT, BoseEinstein

        T = 300
        be = BoseEinstein(temperature=T, chemical_potential=0)

        # At energy = kT, occupation n = 1/(e^1 - 1) ≈ 0.58
        E_kT = BOLTZMANN_CONSTANT * T
        n_kT = be.occupation(energy=E_kT)

        # At energy = 0.1*kT, occupation should be higher
        E_low = 0.1 * BOLTZMANN_CONSTANT * T
        n_low = be.occupation(energy=E_low)

        assert n_low > n_kT  # Lower energy = higher occupation
