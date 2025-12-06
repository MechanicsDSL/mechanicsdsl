"""
Comprehensive tests for units.py
"""
import pytest
import numpy as np
from mechanics_dsl.units import Unit, UnitSystem, BASE_UNITS


class TestUnit:
    """Test Unit class operations"""
    
    def test_unit_creation(self):
        """Test basic unit creation"""
        u1 = Unit({"length": 1})
        assert u1.dimensions == {"length": 1}
        assert u1.scale == 1.0
        
        u2 = Unit({"mass": 1, "length": 2}, scale=2.5)
        assert u2.dimensions == {"mass": 1, "length": 2}
        assert u2.scale == 2.5
    
    def test_unit_multiplication_with_number(self):
        """Test Unit * number"""
        u = Unit({"length": 1})
        result = u * 5.0
        assert result.dimensions == {"length": 1}
        assert result.scale == 5.0
        
        result2 = u * 3
        assert result2.scale == 3.0
    
    def test_unit_rmul(self):
        """Test number * Unit"""
        u = Unit({"length": 1})
        result = 5.0 * u
        assert result.dimensions == {"length": 1}
        assert result.scale == 5.0
    
    def test_unit_multiplication_with_unit(self):
        """Test Unit * Unit"""
        u1 = Unit({"length": 1})
        u2 = Unit({"time": 1})
        result = u1 * u2
        assert result.dimensions == {"length": 1, "time": 1}
        assert result.scale == 1.0
        
        # Test with different scales
        u3 = Unit({"length": 1}, scale=2.0)
        u4 = Unit({"time": 1}, scale=3.0)
        result2 = u3 * u4
        assert result2.scale == 6.0
        
        # Test dimension cancellation
        u5 = Unit({"length": 1})
        u6 = Unit({"length": -1})
        result3 = u5 * u6
        assert result3.dimensions == {}
        assert result3.scale == 1.0
    
    def test_unit_division_with_number(self):
        """Test Unit / number"""
        u = Unit({"length": 1}, scale=10.0)
        result = u / 2.0
        assert result.dimensions == {"length": 1}
        assert result.scale == 5.0
    
    def test_unit_division_with_unit(self):
        """Test Unit / Unit"""
        u1 = Unit({"length": 1})
        u2 = Unit({"time": 1})
        result = u1 / u2
        assert result.dimensions == {"length": 1, "time": -1}
        assert result.scale == 1.0
        
        # Test dimension cancellation
        u3 = Unit({"length": 1})
        u4 = Unit({"length": 1})
        result2 = u3 / u4
        assert result2.dimensions == {}
        assert result2.scale == 1.0
    
    def test_unit_power(self):
        """Test Unit ** exponent"""
        u = Unit({"length": 1}, scale=2.0)
        result = u ** 2.0
        assert result.dimensions == {"length": 2}
        assert result.scale == 4.0
        
        result2 = u ** 0.5
        assert result2.dimensions == {"length": 0.5}
        assert result2.scale == pytest.approx(2.0 ** 0.5)
    
    def test_unit_is_compatible(self):
        """Test unit compatibility checking"""
        u1 = Unit({"length": 1})
        u2 = Unit({"length": 1})
        assert u1.is_compatible(u2)
        
        u3 = Unit({"length": 1, "time": -1})
        assert not u1.is_compatible(u3)
        
        u4 = Unit({})
        u5 = Unit({})
        assert u4.is_compatible(u5)
    
    def test_unit_repr_dimensionless(self):
        """Test Unit.__repr__ for dimensionless unit"""
        u = Unit({}, scale=1.5)
        repr_str = repr(u)
        assert "dimensionless" in repr_str
        assert "1.5" in repr_str
    
    def test_unit_repr_with_dimensions(self):
        """Test Unit.__repr__ with dimensions"""
        u = Unit({"length": 1, "time": -1})
        repr_str = repr(u)
        assert "length" in repr_str
        assert "time" in repr_str


class TestUnitSystem:
    """Test UnitSystem class"""
    
    def test_unit_system_init(self):
        """Test UnitSystem initialization"""
        us = UnitSystem()
        assert isinstance(us.units, dict)
        assert "m" in us.units
        assert "kg" in us.units
        assert "s" in us.units
    
    def test_parse_unit_direct_lookup(self):
        """Test parsing unit with direct lookup"""
        us = UnitSystem()
        u = us.parse_unit("m")
        assert u.dimensions == {"length": 1}
        
        u2 = us.parse_unit("kg")
        assert u2.dimensions == {"mass": 1}
    
    def test_parse_unit_empty_string(self):
        """Test parsing empty unit string"""
        us = UnitSystem()
        u = us.parse_unit("")
        assert u.dimensions == {}
        assert u.scale == 1.0
    
    def test_parse_unit_whitespace(self):
        """Test parsing unit with whitespace"""
        us = UnitSystem()
        u = us.parse_unit("  m  ")
        assert u.dimensions == {"length": 1}
    
    def test_parse_unit_unknown_simple(self):
        """Test parsing unknown simple unit"""
        us = UnitSystem()
        u = us.parse_unit("unknown_unit")
        assert u.dimensions == {}  # Returns dimensionless
    
    def test_parse_unit_expression_multiplication(self):
        """Test parsing unit expression with multiplication"""
        us = UnitSystem()
        u = us.parse_unit("kg*m")
        assert u.dimensions == {"mass": 1, "length": 1}
    
    def test_parse_unit_expression_division(self):
        """Test parsing unit expression with division"""
        us = UnitSystem()
        u = us.parse_unit("m/s")
        assert u.dimensions == {"length": 1, "time": -1}
    
    def test_parse_unit_expression_power_caret(self):
        """Test parsing unit expression with ^"""
        us = UnitSystem()
        u = us.parse_unit("m^2")
        assert u.dimensions == {"length": 2}
    
    def test_parse_unit_expression_power_double_star(self):
        """Test parsing unit expression with **"""
        us = UnitSystem()
        u = us.parse_unit("m**2")
        assert u.dimensions == {"length": 2}
    
    def test_parse_unit_expression_complex(self):
        """Test parsing complex unit expression"""
        us = UnitSystem()
        u = us.parse_unit("kg*m/s^2")
        assert u.dimensions == {"mass": 1, "length": 1, "time": -2}
    
    def test_parse_unit_expression_with_constant(self):
        """Test parsing unit expression with numeric constant"""
        us = UnitSystem()
        u = us.parse_unit("2.5*m")
        assert u.dimensions == {"length": 1}
        assert u.scale == 2.5
    
    def test_parse_unit_expression_nested(self):
        """Test parsing nested unit expression"""
        us = UnitSystem()
        u = us.parse_unit("(kg*m)/s^2")
        assert u.dimensions == {"mass": 1, "length": 1, "time": -2}
    
    def test_parse_unit_invalid_syntax(self):
        """Test parsing invalid syntax"""
        us = UnitSystem()
        u = us.parse_unit("kg*+m")  # Invalid syntax
        assert u.dimensions == {}  # Should return dimensionless
    
    def test_parse_unit_type_error(self):
        """Test parse_unit with non-string input"""
        us = UnitSystem()
        with pytest.raises(TypeError):
            us.parse_unit(123)
        with pytest.raises(TypeError):
            us.parse_unit(None)
    
    def test_parse_unit_expression_unknown_unit(self):
        """Test parsing expression with unknown unit"""
        us = UnitSystem()
        # This should raise ValueError in _parse_unit_expression
        # but parse_unit catches it and returns dimensionless
        u = us.parse_unit("unknown*m")
        assert u.dimensions == {}
    
    def test_parse_unit_expression_unsupported_operator(self):
        """Test parsing expression with unsupported operator"""
        us = UnitSystem()
        # Try to trigger unsupported operator error
        # This is tricky - we'd need to create an AST with unsupported op
        # For now, test that invalid syntax is handled
        u = us.parse_unit("m++s")  # Invalid syntax
        assert u.dimensions == {}
    
    def test_parse_unit_expression_unsupported_node_type(self):
        """Test parsing expression with unsupported AST node"""
        us = UnitSystem()
        # This is hard to trigger directly, but we can test error handling
        u = us.parse_unit("invalid***syntax")
        assert u.dimensions == {}
    
    def test_parse_unit_expression_non_numeric_constant(self):
        """Test parsing expression with non-numeric constant"""
        us = UnitSystem()
        # This is hard to trigger with normal parsing, but test error path
        u = us.parse_unit("'string'*m")  # Invalid
        assert u.dimensions == {}
    
    def test_check_compatibility_same_units(self):
        """Test checking compatibility of same units"""
        us = UnitSystem()
        assert us.check_compatibility("m", "m")
        assert us.check_compatibility("kg", "kg")
    
    def test_check_compatibility_different_units(self):
        """Test checking compatibility of different units"""
        us = UnitSystem()
        assert not us.check_compatibility("m", "kg")
        assert not us.check_compatibility("m", "s")
    
    def test_check_compatibility_compatible_units(self):
        """Test checking compatibility of dimensionally compatible units"""
        us = UnitSystem()
        # N and kg*m/s^2 should be compatible
        assert us.check_compatibility("N", "kg*m/s^2")
    
    def test_check_compatibility_type_error(self):
        """Test check_compatibility with non-string inputs"""
        us = UnitSystem()
        with pytest.raises(TypeError):
            us.check_compatibility(123, "m")
        with pytest.raises(TypeError):
            us.check_compatibility("m", 123)
        with pytest.raises(TypeError):
            us.check_compatibility(None, "m")
    
    def test_parse_unit_expression_binop_handling(self):
        """Test binary operation handling in AST parsing"""
        us = UnitSystem()
        # Test various binary operations
        u1 = us.parse_unit("m*s")
        assert u1.dimensions == {"length": 1, "time": 1}
        
        u2 = us.parse_unit("m/s")
        assert u2.dimensions == {"length": 1, "time": -1}
        
        u3 = us.parse_unit("m^3")
        assert u3.dimensions == {"length": 3}
    
    def test_base_units_exist(self):
        """Test that all base units are defined"""
        assert "m" in BASE_UNITS
        assert "kg" in BASE_UNITS
        assert "s" in BASE_UNITS
        assert "A" in BASE_UNITS
        assert "K" in BASE_UNITS
        assert "mol" in BASE_UNITS
        assert "cd" in BASE_UNITS
        assert "N" in BASE_UNITS
        assert "J" in BASE_UNITS
        assert "W" in BASE_UNITS
        assert "rad" in BASE_UNITS
        assert "deg" in BASE_UNITS
    
    def test_deg_unit_scale(self):
        """Test that deg unit has correct scale"""
        deg_unit = BASE_UNITS["deg"]
        assert deg_unit.scale == pytest.approx(np.pi / 180)
        assert deg_unit.dimensions == {"angle": 1}
    
    def test_dimensionless_units(self):
        """Test dimensionless units"""
        us = UnitSystem()
        u1 = us.parse_unit("dimensionless")
        assert u1.dimensions == {}
        
        u2 = us.parse_unit("1")
        assert u2.dimensions == {}
