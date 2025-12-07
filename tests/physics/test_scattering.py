"""
Tests for Scattering Theory Module
"""
import pytest
import numpy as np

from mechanics_dsl.domains.classical import (
    ScatteringAnalyzer,
    ScatteringResult,
    SymbolicScattering,
    rutherford_angle,
    rutherford_cross_section
)


class TestScatteringAnalyzer:
    """Test ScatteringAnalyzer class."""
    
    def test_create_analyzer(self):
        """Test creating analyzer."""
        analyzer = ScatteringAnalyzer()
        assert analyzer is not None
    
    def test_coulomb_scattering(self):
        """Test Coulomb/Rutherford scattering."""
        analyzer = ScatteringAnalyzer()
        
        result = analyzer.coulomb_scattering(
            energy=1.0,
            impact_parameter=1.0,
            k=1.0
        )
        
        assert isinstance(result, ScatteringResult)
        assert result.scattering_angle >= 0
        assert result.scattering_angle <= np.pi
        assert result.closest_approach > 0
    
    def test_head_on_collision(self):
        """Test head-on (b=0) collision."""
        analyzer = ScatteringAnalyzer()
        
        result = analyzer.coulomb_scattering(
            energy=1.0,
            impact_parameter=0.0,
            k=1.0
        )
        
        # Head-on: θ = π (complete backscatter)
        assert np.isclose(result.scattering_angle, np.pi)
    
    def test_hard_sphere_scattering(self):
        """Test hard sphere scattering."""
        analyzer = ScatteringAnalyzer()
        
        # Impact parameter less than radius
        result = analyzer.hard_sphere_scattering(radius=1.0, impact_parameter=0.5)
        
        assert result.scattering_angle > 0
        assert result.closest_approach == 1.0  # Touches sphere surface
    
    def test_hard_sphere_miss(self):
        """Test hard sphere miss (b > R)."""
        analyzer = ScatteringAnalyzer()
        
        result = analyzer.hard_sphere_scattering(radius=1.0, impact_parameter=2.0)
        
        assert result.scattering_angle == 0.0  # No scattering
    
    def test_differential_cross_section(self):
        """Test Rutherford differential cross-section."""
        analyzer = ScatteringAnalyzer()
        
        dcs = analyzer.rutherford_differential_cross_section(
            k=1.0, energy=1.0, theta=np.pi/2
        )
        
        assert dcs > 0
        assert np.isfinite(dcs)


class TestSymbolicScattering:
    """Test symbolic scattering."""
    
    def test_rutherford_formula(self):
        """Test symbolic Rutherford formulas."""
        symbolic = SymbolicScattering()
        
        result = symbolic.rutherford_formula()
        
        assert 'scattering_angle' in result
        assert 'differential_cross_section' in result
        assert 'closest_approach' in result
    
    def test_orbit_equation(self):
        """Test symbolic orbit equation."""
        symbolic = SymbolicScattering()
        
        orbit = symbolic.orbit_equation()
        
        # Should be an equation (Equality)
        import sympy as sp
        assert isinstance(orbit, sp.Equality)


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_rutherford_angle(self):
        """Test rutherford_angle function."""
        # Large b -> small angle
        theta_large_b = rutherford_angle(1.0, 10.0, 1.0)
        theta_small_b = rutherford_angle(1.0, 0.1, 1.0)
        
        assert theta_large_b < theta_small_b
    
    def test_rutherford_cross_section(self):
        """Test rutherford_cross_section function."""
        # Cross-section should be larger at small angles
        dcs_90 = rutherford_cross_section(1.0, 1.0, np.pi/2)
        dcs_45 = rutherford_cross_section(1.0, 1.0, np.pi/4)
        
        # sin^4(θ/2) is smaller for smaller θ, so dσ/dΩ is larger
        assert dcs_45 > dcs_90
    
    def test_cross_section_symmetry(self):
        """Test that cross-section depends on sin(θ/2)."""
        # θ = π/2 and θ = 3π/2 should give same cross-section
        dcs1 = rutherford_cross_section(1.0, 1.0, np.pi/2)
        dcs2 = rutherford_cross_section(1.0, 1.0, 3*np.pi/2)
        
        # sin(π/4) = sin(3π/4), so cross-sections are equal
        assert np.isclose(dcs1, dcs2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
