"""
Tests for Continuous Systems Module
"""
import pytest
import numpy as np
import sympy as sp

from mechanics_dsl.domains.classical import (
    LagrangianDensity,
    FieldEulerLagrange,
    VibratingString,
    VibratingMembrane,
    StressEnergyTensor,
    WaveMode,
    FieldConfiguration,
    string_mode_frequencies,
    wave_speed
)


class TestLagrangianDensity:
    """Test LagrangianDensity class."""
    
    def test_create_density(self):
        """Test creating Lagrangian density."""
        density = LagrangianDensity()
        assert density is not None
    
    def test_string_lagrangian(self):
        """Test vibrating string Lagrangian density."""
        density = LagrangianDensity()
        
        L = density.string_lagrangian()
        
        # Should contain time and space derivatives
        assert L is not None
        # L = (1/2)*μ*u_t² - (1/2)*T*u_x²
        # Check it's a sympy expression
        assert isinstance(L, sp.Expr)
    
    def test_membrane_lagrangian(self):
        """Test vibrating membrane Lagrangian density."""
        density = LagrangianDensity()
        
        L = density.membrane_lagrangian()
        
        assert isinstance(L, sp.Expr)
    
    def test_klein_gordon_lagrangian(self):
        """Test Klein-Gordon Lagrangian density."""
        density = LagrangianDensity()
        
        L = density.klein_gordon_lagrangian()
        
        assert isinstance(L, sp.Expr)


class TestFieldEulerLagrange:
    """Test field Euler-Lagrange equations."""
    
    def test_wave_equation_1d(self):
        """Test deriving 1D wave equation."""
        field_el = FieldEulerLagrange()
        
        wave_eq = field_el.wave_equation_1d()
        
        # Should be an equation u_tt = c²*u_xx
        assert isinstance(wave_eq, sp.Equality)


class TestVibratingString:
    """Test vibrating string solver."""
    
    def test_create_string(self):
        """Test creating vibrating string."""
        string = VibratingString(length=1.0, wave_speed=100.0)
        
        assert string.L == 1.0
        assert string.c == 100.0
    
    def test_fundamental_frequency(self):
        """Test fundamental frequency."""
        string = VibratingString(length=1.0, wave_speed=100.0)
        
        f1 = string.fundamental_frequency()
        
        # f₁ = c / (2L) = 100 / 2 = 50 Hz
        assert f1 == 50.0
    
    def test_mode_frequency(self):
        """Test mode frequencies."""
        string = VibratingString(length=1.0, wave_speed=100.0)
        
        f1 = string.mode_frequency(1)
        f2 = string.mode_frequency(2)
        f3 = string.mode_frequency(3)
        
        # Harmonic series: f_n = n * f_1
        assert f2 == 2 * f1
        assert f3 == 3 * f1
    
    def test_mode_shape(self):
        """Test mode shape."""
        string = VibratingString(length=1.0, wave_speed=100.0)
        
        x = np.linspace(0, 1, 100)
        mode1 = string.mode_shape(1, x)
        
        # Mode shape should be sin(πx/L)
        # Zero at boundaries
        assert np.isclose(mode1[0], 0.0, atol=1e-10)
        assert np.isclose(mode1[-1], 0.0, atol=1e-10)
        # Maximum at center
        assert mode1[50] > 0.9
    
    def test_compute_modes(self):
        """Test computing normal modes."""
        string = VibratingString(length=1.0, wave_speed=100.0)
        
        modes = string.compute_modes(n_modes=5)
        
        assert len(modes) == 5
        assert all(isinstance(m, WaveMode) for m in modes)
        # Frequencies should increase
        for i in range(1, 5):
            assert modes[i].frequency > modes[i-1].frequency
    
    def test_fourier_coefficients(self):
        """Test Fourier coefficient calculation."""
        string = VibratingString(length=1.0, wave_speed=100.0)
        
        x = np.linspace(0, 1, 100)
        # Initial displacement: first mode shape
        initial = np.sin(np.pi * x)
        
        coeffs = string.fourier_coefficients(initial, x, n_modes=5)
        
        # First coefficient should be dominant
        assert abs(coeffs[0]) > abs(coeffs[1])


class TestVibratingMembrane:
    """Test vibrating membrane solver."""
    
    def test_create_membrane(self):
        """Test creating membrane."""
        membrane = VibratingMembrane(length_x=1.0, length_y=1.0, wave_speed=100.0)
        
        assert membrane.a == 1.0
        assert membrane.b == 1.0
    
    def test_mode_frequency(self):
        """Test membrane mode frequencies."""
        membrane = VibratingMembrane(length_x=1.0, length_y=1.0, wave_speed=100.0)
        
        f11 = membrane.mode_frequency(1, 1)
        f12 = membrane.mode_frequency(1, 2)
        f21 = membrane.mode_frequency(2, 1)
        
        # For square membrane, (1,2) and (2,1) are degenerate
        assert np.isclose(f12, f21)
    
    def test_mode_shape_2d(self):
        """Test 2D mode shape."""
        membrane = VibratingMembrane(length_x=1.0, length_y=1.0, wave_speed=100.0)
        
        x = np.linspace(0, 1, 20)
        y = np.linspace(0, 1, 20)
        
        mode = membrane.mode_shape(1, 1, x, y)
        
        # Should be zero on boundaries
        assert np.allclose(mode[0, :], 0.0, atol=1e-10)
        assert np.allclose(mode[-1, :], 0.0, atol=1e-10)
        assert np.allclose(mode[:, 0], 0.0, atol=1e-10)
        assert np.allclose(mode[:, -1], 0.0, atol=1e-10)
    
    def test_compute_modes(self):
        """Test computing membrane modes."""
        membrane = VibratingMembrane(length_x=1.0, length_y=2.0, wave_speed=100.0)
        
        modes = membrane.compute_modes(max_m=3, max_n=3)
        
        # 9 modes total (3 x 3)
        assert len(modes) == 9
        # Should be sorted by frequency
        for i in range(1, len(modes)):
            assert modes[i][2] >= modes[i-1][2]


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_string_mode_frequencies(self):
        """Test string_mode_frequencies function."""
        freqs = string_mode_frequencies(
            length=1.0, tension=100.0, density=1.0, n_modes=3
        )
        
        # c = sqrt(T/μ) = sqrt(100/1) = 10
        # f_n = n*c/(2L) = n*10/2 = 5n
        assert len(freqs) == 3
        assert np.isclose(freqs[0], 5.0)
        assert np.isclose(freqs[1], 10.0)
        assert np.isclose(freqs[2], 15.0)
    
    def test_wave_speed(self):
        """Test wave_speed function."""
        c = wave_speed(tension=100.0, density=4.0)
        
        # c = sqrt(100/4) = 5
        assert c == 5.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
