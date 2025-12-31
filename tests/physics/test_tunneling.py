"""
Tests for quantum tunneling implementations.
"""
import pytest
import numpy as np


class TestQuantumTunneling:
    """Tests for QuantumTunneling class."""
    
    def test_rectangular_barrier_below(self):
        """Test tunneling probability for E < V0."""
        from mechanics_dsl.domains.quantum import QuantumTunneling
        
        tunneling = QuantumTunneling(mass=1.0, hbar=1.0)
        
        # E < V0: particle tunnels
        T = tunneling.rectangular_barrier(E=1.0, V0=2.0, width=1.0)
        
        assert 0 < T < 1, "Transmission should be between 0 and 1"
        
    def test_rectangular_barrier_above(self):
        """Test transmission for E > V0 (above barrier)."""
        from mechanics_dsl.domains.quantum import QuantumTunneling
        
        tunneling = QuantumTunneling(mass=1.0, hbar=1.0)
        
        # E > V0: classical transmission with resonances
        T = tunneling.rectangular_barrier(E=3.0, V0=2.0, width=1.0)
        
        assert 0 < T <= 1, "Transmission should be between 0 and 1"
    
    def test_tunneling_decreases_with_barrier_width(self):
        """Wider barriers = less tunneling."""
        from mechanics_dsl.domains.quantum import QuantumTunneling
        
        tunneling = QuantumTunneling(mass=1.0, hbar=1.0)
        
        T_narrow = tunneling.rectangular_barrier(E=1.0, V0=2.0, width=0.5)
        T_wide = tunneling.rectangular_barrier(E=1.0, V0=2.0, width=2.0)
        
        assert T_narrow > T_wide, "Wider barrier should have lower transmission"
    
    def test_tunneling_decreases_with_barrier_height(self):
        """Higher barriers = less tunneling."""
        from mechanics_dsl.domains.quantum import QuantumTunneling
        
        tunneling = QuantumTunneling(mass=1.0, hbar=1.0)
        
        T_low = tunneling.rectangular_barrier(E=1.0, V0=2.0, width=1.0)
        T_high = tunneling.rectangular_barrier(E=1.0, V0=5.0, width=1.0)
        
        assert T_low > T_high, "Higher barrier should have lower transmission"
    
    def test_decay_constant(self):
        """Test κ = √(2m(V-E))/ℏ."""
        from mechanics_dsl.domains.quantum import QuantumTunneling
        
        tunneling = QuantumTunneling(mass=1.0, hbar=1.0)
        
        kappa = tunneling.decay_constant(E=1.0, V=2.0)
        expected = np.sqrt(2 * 1.0 * (2.0 - 1.0)) / 1.0
        
        assert np.isclose(kappa, expected)
    
    def test_decay_constant_allowed_region(self):
        """Decay constant should be 0 in allowed region (V < E)."""
        from mechanics_dsl.domains.quantum import QuantumTunneling
        
        tunneling = QuantumTunneling(mass=1.0, hbar=1.0)
        
        kappa = tunneling.decay_constant(E=2.0, V=1.0)
        
        assert kappa == 0.0
    
    def test_wkb_transmission(self):
        """Test WKB approximation for triangular-ish barrier."""
        from mechanics_dsl.domains.quantum import QuantumTunneling
        
        tunneling = QuantumTunneling(mass=1.0, hbar=1.0)
        
        # Simple Gaussian barrier
        def barrier(x):
            return 2.0 * np.exp(-x**2)
        
        T = tunneling.wkb_transmission(E=0.5, potential=barrier, x1=-2, x2=2)
        
        assert 0 < T < 1


class TestTunnelingPhysics:
    """Tests for physical correctness of tunneling formulas."""
    
    def test_above_barrier_resonances(self):
        """Above barrier, T should oscillate with energy (resonances)."""
        from mechanics_dsl.domains.quantum import QuantumTunneling
        
        tunneling = QuantumTunneling(mass=1.0, hbar=1.0)
        
        # Sample many energies above barier
        energies = np.linspace(2.1, 10.0, 100)
        transmissions = [tunneling.rectangular_barrier(E, V0=2.0, width=5.0) 
                        for E in energies]
        
        # Should have oscillations (max != min)
        assert max(transmissions) > min(transmissions) + 0.01
    
    def test_double_well_splitting(self):
        """Test tunnel splitting decreases with barrier height."""
        from mechanics_dsl.domains.quantum import QuantumTunneling
        
        tunneling = QuantumTunneling(mass=1.0, hbar=1.0)
        
        split_low = tunneling.double_well_splitting(omega=1.0, barrier_height=1.0, 
                                                     well_separation=2.0)
        split_high = tunneling.double_well_splitting(omega=1.0, barrier_height=5.0, 
                                                      well_separation=2.0)
        
        assert split_low > split_high, "Higher barrier should give smaller splitting"


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_tunneling_probability_rectangular(self):
        """Test convenience wrapper."""
        from mechanics_dsl.domains.quantum import tunneling_probability_rectangular
        
        T = tunneling_probability_rectangular(E=1.0, V0=2.0, width=1.0)
        
        assert 0 < T < 1
