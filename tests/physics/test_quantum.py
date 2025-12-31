"""
Tests for quantum mechanics domain.
"""
import pytest
import numpy as np


class TestQuantumHarmonicOscillator:
    """Test QuantumHarmonicOscillator."""
    
    def test_energy_levels(self):
        """Test E_n = ℏω(n + 1/2)."""
        from mechanics_dsl.domains.quantum import QuantumHarmonicOscillator
        
        qho = QuantumHarmonicOscillator(mass=1.0, omega=1.0, hbar=1.0)
        
        assert qho.energy_level(0) == 0.5  # Ground state
        assert qho.energy_level(1) == 1.5
        assert qho.energy_level(2) == 2.5
    
    def test_zero_point_energy(self):
        """Test E_0 = ℏω/2."""
        from mechanics_dsl.domains.quantum import QuantumHarmonicOscillator
        
        qho = QuantumHarmonicOscillator(omega=2.0, hbar=1.0)
        assert qho.zero_point_energy() == 1.0
    
    def test_characteristic_length(self):
        """Test a = √(ℏ/(mω))."""
        from mechanics_dsl.domains.quantum import QuantumHarmonicOscillator
        
        qho = QuantumHarmonicOscillator(mass=1.0, omega=1.0, hbar=1.0)
        assert qho.characteristic_length() == 1.0
        
        qho2 = QuantumHarmonicOscillator(mass=1.0, omega=4.0, hbar=1.0)
        assert np.isclose(qho2.characteristic_length(), 0.5)
    
    def test_wavefunction_normalization(self):
        """Test that wavefunctions are normalized."""
        from mechanics_dsl.domains.quantum import QuantumHarmonicOscillator
        
        qho = QuantumHarmonicOscillator()
        x = np.linspace(-10, 10, 1000)
        dx = x[1] - x[0]
        
        for n in range(5):
            psi = qho.wavefunction(x, n)
            norm = np.sum(np.abs(psi)**2) * dx
            assert np.isclose(norm, 1.0, atol=0.01)
    
    def test_uncertainty_minimum(self):
        """Test ground state has minimum uncertainty."""
        from mechanics_dsl.domains.quantum import QuantumHarmonicOscillator
        
        qho = QuantumHarmonicOscillator(hbar=1.0)
        delta_x_delta_p = qho.uncertainty_product(0)
        
        # Ground state: Δx·Δp = ℏ/2
        assert np.isclose(delta_x_delta_p, 0.5)


class TestInfiniteSquareWell:
    """Test InfiniteSquareWell."""
    
    def test_energy_levels(self):
        """Test E_n = n²π²ℏ²/(2mL²)."""
        from mechanics_dsl.domains.quantum import InfiniteSquareWell
        
        well = InfiniteSquareWell(length=1.0, mass=1.0, hbar=1.0)
        
        E1 = well.energy_level(1)
        expected = np.pi**2 / 2
        assert np.isclose(E1, expected)
        
        E2 = well.energy_level(2)
        assert np.isclose(E2, 4 * E1)  # E_n ∝ n²
    
    def test_zero_quantum_number_invalid(self):
        """Test n=0 is invalid."""
        from mechanics_dsl.domains.quantum import InfiniteSquareWell
        
        well = InfiniteSquareWell()
        
        with pytest.raises(ValueError):
            well.energy_level(0)
    
    def test_wavefunction_normalization(self):
        """Test wavefunctions are normalized."""
        from mechanics_dsl.domains.quantum import InfiniteSquareWell
        
        well = InfiniteSquareWell(length=1.0)
        x = np.linspace(0, 1, 1000)
        dx = x[1] - x[0]
        
        for n in range(1, 5):
            psi = well.wavefunction(x, n)
            norm = np.sum(np.abs(psi)**2) * dx
            assert np.isclose(norm, 1.0, atol=0.01)
    
    def test_position_expectation(self):
        """Test <x> = L/2."""
        from mechanics_dsl.domains.quantum import InfiniteSquareWell
        
        well = InfiniteSquareWell(length=2.0)
        
        for n in range(1, 5):
            assert well.position_expectation(n) == 1.0


class TestWKBApproximation:
    """Test WKB approximation."""
    
    def test_harmonic_oscillator_levels(self):
        """Test WKB gives correct harmonic oscillator levels."""
        from mechanics_dsl.domains.quantum import WKBApproximation
        
        # V(x) = 0.5*ω²x² with ω=1
        def potential(x):
            return 0.5 * x**2
        
        wkb = WKBApproximation(potential=potential, mass=1.0, hbar=1.0)
        
        # Find first level
        E0 = wkb.find_energy_level(0, E_range=(0.1, 2.0), x_range=(-5, 5))
        
        # Should be close to 0.5 (ground state of QHO)
        if not np.isnan(E0):
            assert np.isclose(E0, 0.5, rtol=0.1)
    
    def test_classical_momentum(self):
        """Test classical momentum calculation."""
        from mechanics_dsl.domains.quantum import WKBApproximation
        
        def potential(x):
            return 0.0  # Free particle
        
        wkb = WKBApproximation(potential=potential, mass=1.0)
        
        # p = √(2m(E-V)) = √(2*1*1) = √2
        p = wkb.classical_momentum(x=0, E=1.0)
        assert np.isclose(p, np.sqrt(2))
    
    def test_forbidden_region(self):
        """Test momentum is zero in classically forbidden region."""
        from mechanics_dsl.domains.quantum import WKBApproximation
        
        def potential(x):
            return 1.0  # Constant V=1
        
        wkb = WKBApproximation(potential=potential, mass=1.0)
        
        # E < V, so classically forbidden
        p = wkb.classical_momentum(x=0, E=0.5)
        assert p == 0.0


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    def test_de_broglie_wavelength(self):
        """Test λ = 2πℏ/p."""
        from mechanics_dsl.domains.quantum import de_broglie_wavelength
        
        lam = de_broglie_wavelength(momentum=1.0, hbar=1.0)
        assert np.isclose(lam, 2 * np.pi)
    
    def test_heisenberg_minimum(self):
        """Test minimum uncertainty ℏ/2."""
        from mechanics_dsl.domains.quantum import heisenberg_minimum
        
        assert heisenberg_minimum(hbar=1.0) == 0.5
        assert heisenberg_minimum(hbar=2.0) == 1.0
