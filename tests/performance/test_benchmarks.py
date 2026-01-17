"""
Performance benchmarks for MechanicsDSL physics domains.

Run with: pytest tests/performance/test_benchmarks.py -v --benchmark-only
Or without pytest-benchmark: pytest tests/performance/test_benchmarks.py -v
"""

import time

import numpy as np


class TestPhysicsBenchmarks:
    """Benchmark tests for physics calculations."""

    def test_benchmark_quantum_tunneling(self):
        """Benchmark quantum tunneling calculations."""
        from mechanics_dsl.domains.quantum import QuantumTunneling

        tunneling = QuantumTunneling(mass=1.0, hbar=1.0)

        start = time.perf_counter()
        for _ in range(1000):
            tunneling.rectangular_barrier(E=0.5, V0=1.0, width=2.0)
        elapsed = time.perf_counter() - start

        print(f"\nTunneling: {1000/elapsed:.0f} calculations/sec")
        assert elapsed < 1.0  # Should complete in < 1 second

    def test_benchmark_hydrogen_atom(self):
        """Benchmark hydrogen atom calculations."""
        from mechanics_dsl.domains.quantum import HydrogenAtom

        H = HydrogenAtom(Z=1)

        start = time.perf_counter()
        for n in range(1, 101):
            for _ in range(10):
                H.energy_level(n)
                H.bohr_radius_n(n)
                H.degeneracy(n)
        elapsed = time.perf_counter() - start

        print(f"\nHydrogen atom: {3000/elapsed:.0f} calculations/sec")
        assert elapsed < 1.0

    def test_benchmark_schwarzschild(self):
        """Benchmark black hole calculations."""
        from mechanics_dsl.domains.general_relativity import SchwarzschildMetric

        bh = SchwarzschildMetric(mass=1e30)

        start = time.perf_counter()
        for _ in range(1000):
            bh.schwarzschild_radius()
            bh.isco_radius()
            bh.photon_sphere_radius()
            bh.hawking_temperature()
        elapsed = time.perf_counter() - start

        print(f"\nSchwarzschild: {4000/elapsed:.0f} calculations/sec")
        assert elapsed < 1.0

    def test_benchmark_ising_sweep(self):
        """Benchmark Ising model Monte Carlo."""
        from mechanics_dsl.domains.statistical import IsingModel

        ising = IsingModel(L=20, dimension=2, temperature=2.5)
        ising.initialize_random()

        start = time.perf_counter()
        for _ in range(10):
            ising.monte_carlo_sweep()
        elapsed = time.perf_counter() - start

        sweeps_per_sec = 10 / elapsed
        print(f"\nIsing 20x20: {sweeps_per_sec:.1f} sweeps/sec")
        assert elapsed < 5.0

    def test_benchmark_carnot_engine(self):
        """Benchmark thermodynamic cycle calculations."""
        from mechanics_dsl.domains.thermodynamics import CarnotEngine

        start = time.perf_counter()
        for T_cold in range(100, 400, 10):
            for T_hot in range(T_cold + 10, 600, 10):
                engine = CarnotEngine(T_hot=T_hot, T_cold=T_cold)
                engine.efficiency()
                engine.work_output(1000)
        elapsed = time.perf_counter() - start

        print(f"\nCarnot: {3000/elapsed:.0f} calculations/sec")
        assert elapsed < 1.0

    def test_benchmark_em_wave(self):
        """Benchmark electromagnetic wave calculations."""
        from mechanics_dsl.domains.electromagnetic import ElectromagneticWave

        start = time.perf_counter()
        for f in np.logspace(6, 12, 100):
            wave = ElectromagneticWave(frequency=f)
            wave.wavelength()
            wave.intensity()
            wave.impedance()
        elapsed = time.perf_counter() - start

        print(f"\nEM waves: {300/elapsed:.0f} calculations/sec")
        assert elapsed < 1.0

    def test_benchmark_lorentz_transforms(self):
        """Benchmark relativistic calculations."""
        from mechanics_dsl.domains.relativistic import FourVector, gamma

        start = time.perf_counter()
        for v in np.linspace(0.1, 0.99, 100):
            gamma(v * 299792458)
            vec = FourVector(ct=1.0, x=1.0, y=0.0, z=0.0)
            _ = vec.invariant()
        elapsed = time.perf_counter() - start

        print(f"\nRelativistic: {200/elapsed:.0f} calculations/sec")
        assert elapsed < 1.0


class TestScalingBenchmarks:
    """Test computational scaling."""

    def test_ising_scaling(self):
        """Ising model should scale as O(L²) for 2D."""
        from mechanics_dsl.domains.statistical import IsingModel

        times = []
        sizes = [10, 20, 40]

        for L in sizes:
            ising = IsingModel(L=L, dimension=2, temperature=2.5)
            ising.initialize_random()

            start = time.perf_counter()
            for _ in range(5):
                ising.monte_carlo_sweep()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        # Check approximate O(L²) scaling
        ratio_1 = times[1] / times[0]
        ratio_2 = times[2] / times[1]

        # Should scale roughly as 4x (2² = 4)
        assert 2 < ratio_1 < 8, f"Scaling ratio 1: {ratio_1}"
        assert 2 < ratio_2 < 8, f"Scaling ratio 2: {ratio_2}"
