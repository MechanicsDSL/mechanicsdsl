"""
Property-based tests for physics domains using Hypothesis.

Tests fundamental physics invariants that must hold for all valid inputs.
"""

pytest = __import__("pytest")
hypothesis = pytest.importorskip("hypothesis")

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st


class TestEnergyConservation:
    """Property tests for energy conservation."""

    @given(
        mass=st.floats(min_value=0.1, max_value=100, allow_nan=False),
        v=st.floats(min_value=0.0, max_value=100, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_kinetic_energy_positive(self, mass, v):
        """Kinetic energy is always non-negative."""
        KE = 0.5 * mass * v**2
        assert KE >= 0

    @given(
        m1=st.floats(min_value=0.1, max_value=10, allow_nan=False),
        m2=st.floats(min_value=0.1, max_value=10, allow_nan=False),
        v1=st.floats(min_value=-10, max_value=10, allow_nan=False),
        v2=st.floats(min_value=-10, max_value=10, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_momentum_conservation_elastic(self, m1, m2, v1, v2):
        """Total momentum conserved in elastic collision."""
        p_before = m1 * v1 + m2 * v2

        # Elastic collision formulas
        v1_after = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
        v2_after = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)

        p_after = m1 * v1_after + m2 * v2_after

        assert np.isclose(p_before, p_after, rtol=1e-10)


class TestRelativisticInvariants:
    """Property tests for relativistic physics."""

    @given(v=st.floats(min_value=0.0, max_value=0.99, allow_nan=False))
    @settings(max_examples=50)
    def test_gamma_greater_than_one(self, v):
        """Lorentz factor γ ≥ 1."""
        c = 1.0
        gamma = 1.0 / np.sqrt(1 - v**2 / c**2)
        assert gamma >= 1.0

    @given(v=st.floats(min_value=0.01, max_value=0.99, allow_nan=False))
    @settings(max_examples=50)
    def test_time_dilation(self, v):
        """Moving clocks run slower: Δt' > Δt₀."""
        c = 1.0
        gamma = 1.0 / np.sqrt(1 - v**2 / c**2)
        proper_time = 1.0
        coordinate_time = gamma * proper_time

        assert coordinate_time >= proper_time

    @given(
        E=st.floats(min_value=1, max_value=100, allow_nan=False),
        m=st.floats(min_value=0.1, max_value=10, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_energy_momentum_relation(self, E, m):
        """E² = (pc)² + (mc²)² has real momentum for E > mc²."""
        c = 1.0
        assume(E >= m * c**2)

        p_squared = E**2 - (m * c**2) ** 2
        assert p_squared >= 0


class TestQuantumMechanics:
    """Property tests for quantum mechanics."""

    @given(n=st.integers(min_value=1, max_value=100))
    @settings(max_examples=50)
    def test_infinite_well_energies_increase(self, n):
        """E_n ∝ n² means higher n = higher energy."""
        from mechanics_dsl.domains.quantum import InfiniteSquareWell

        well = InfiniteSquareWell(length=1.0, mass=1.0, hbar=1.0)

        E_n = well.energy_level(n)
        E_n_plus_1 = well.energy_level(n + 1)

        assert E_n_plus_1 > E_n

    @given(
        E=st.floats(min_value=0.1, max_value=10, allow_nan=False),
        V0=st.floats(min_value=0.1, max_value=10, allow_nan=False),
        width=st.floats(min_value=0.1, max_value=5, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_tunneling_probability_bounded(self, E, V0, width):
        """Tunneling probability T ∈ [0, 1]."""
        from mechanics_dsl.domains.quantum import QuantumTunneling

        assume(E > 0 and V0 > 0 and width > 0)

        tunneling = QuantumTunneling(mass=1.0, hbar=1.0)
        T = tunneling.rectangular_barrier(E, V0, width)

        assert 0 <= T <= 1


class TestThermodynamics:
    """Property tests for thermodynamics."""

    @given(
        T_hot=st.floats(min_value=100, max_value=1000, allow_nan=False),
        T_cold=st.floats(min_value=10, max_value=99, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_carnot_efficiency_less_than_one(self, T_hot, T_cold):
        """Carnot efficiency η < 1."""
        from mechanics_dsl.domains.thermodynamics import CarnotEngine

        assume(T_hot > T_cold)

        engine = CarnotEngine(T_hot=T_hot, T_cold=T_cold)
        eta = engine.efficiency()

        assert 0 < eta < 1

    @given(
        T=st.floats(min_value=100, max_value=1000, allow_nan=False),
        V=st.floats(min_value=0.001, max_value=1, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_ideal_gas_pressure_positive(self, T, V):
        """Ideal gas pressure P > 0 for T, V > 0."""
        from mechanics_dsl.domains.statistical import IdealGas

        gas = IdealGas(n_moles=1.0, temperature=T, volume=V)
        P = gas.pressure()

        assert P > 0


class TestGeneralRelativity:
    """Property tests for general relativity."""

    @given(mass=st.floats(min_value=1e20, max_value=1e40, allow_nan=False))
    @settings(max_examples=30)
    def test_schwarzschild_radius_positive(self, mass):
        """Schwarzschild radius is always positive."""
        from mechanics_dsl.domains.general_relativity import SchwarzschildMetric

        bh = SchwarzschildMetric(mass=mass)
        rs = bh.schwarzschild_radius()

        assert rs > 0

    @given(mass=st.floats(min_value=1e28, max_value=1e35, allow_nan=False))
    @settings(max_examples=30)
    def test_isco_greater_than_horizon(self, mass):
        """ISCO radius > event horizon."""
        from mechanics_dsl.domains.general_relativity import SchwarzschildMetric

        bh = SchwarzschildMetric(mass=mass)

        rs = bh.schwarzschild_radius()
        r_isco = bh.isco_radius()

        assert r_isco > rs
