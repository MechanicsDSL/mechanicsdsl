"""
Tests for MechanicsDSL Presets Module

Run with:
    pytest tests/test_presets.py -v
"""

import pytest


class TestPresets:
    """Test preset loading functionality."""
    
    def test_import_presets(self):
        """Test that presets module imports correctly."""
        from mechanics_dsl.presets import PRESETS, get_preset, list_presets
        assert PRESETS is not None
        assert callable(get_preset)
        assert callable(list_presets)
    
    def test_list_presets(self):
        """Test listing available presets."""
        from mechanics_dsl.presets import list_presets
        presets = list_presets()
        assert isinstance(presets, list)
        assert len(presets) > 0
        assert 'pendulum' in presets
    
    def test_get_pendulum_preset(self):
        """Test getting pendulum preset."""
        from mechanics_dsl.presets import get_preset
        dsl = get_preset('pendulum')
        assert '\\system{' in dsl
        assert '\\defvar{' in dsl
        assert '\\lagrangian{' in dsl
    
    def test_get_double_pendulum_preset(self):
        """Test getting double pendulum preset."""
        from mechanics_dsl.presets import get_preset
        dsl = get_preset('double_pendulum')
        assert 'theta1' in dsl
        assert 'theta2' in dsl
    
    def test_get_spring_preset(self):
        """Test getting spring mass preset."""
        from mechanics_dsl.presets import get_preset
        dsl = get_preset('spring')
        assert '\\system{' in dsl
    
    def test_get_orbit_preset(self):
        """Test getting kepler orbit preset."""
        from mechanics_dsl.presets import get_preset
        dsl = get_preset('kepler')
        assert 'kepler' in dsl.lower() or 'orbit' in dsl.lower()
    
    def test_preset_not_found(self):
        """Test that unknown preset raises KeyError."""
        from mechanics_dsl.presets import get_preset
        with pytest.raises(KeyError) as exc_info:
            get_preset('nonexistent_preset')
        assert 'not found' in str(exc_info.value).lower()
    
    def test_case_insensitive(self):
        """Test that preset names are case-insensitive."""
        from mechanics_dsl.presets import get_preset
        dsl1 = get_preset('PENDULUM')
        dsl2 = get_preset('pendulum')
        assert dsl1 == dsl2
    
    def test_preset_aliases(self):
        """Test that preset aliases work."""
        from mechanics_dsl.presets import get_preset
        # 'spring' and 'spring_mass' should return the same preset
        spring = get_preset('spring')
        spring_mass = get_preset('spring_mass')
        assert spring == spring_mass
    
    def test_all_presets_have_system(self):
        """Test that all presets have valid system declarations."""
        from mechanics_dsl.presets import PRESETS
        for name, dsl in PRESETS.items():
            assert '\\system{' in dsl, f"Preset {name} missing \\system declaration"
    
    def test_all_presets_have_lagrangian(self):
        """Test that all presets have Lagrangian definitions."""
        from mechanics_dsl.presets import PRESETS
        seen = set()
        for name, dsl in PRESETS.items():
            if dsl not in seen:
                assert '\\lagrangian{' in dsl, f"Preset {name} missing \\lagrangian"
                seen.add(dsl)


class TestPresetsAPI:
    """Test presets via main package API."""
    
    def test_get_preset_from_package(self):
        """Test get_preset from main package."""
        from mechanics_dsl import get_preset
        dsl = get_preset('pendulum')
        assert '\\system{' in dsl
    
    def test_list_presets_from_package(self):
        """Test list_presets from main package."""
        from mechanics_dsl import list_presets
        presets = list_presets()
        assert 'pendulum' in presets


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
