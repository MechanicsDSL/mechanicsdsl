"""
Tests for Kinematic Equations Module

Comprehensive tests for the 5 kinematic equations and related utilities.
"""
import pytest
import sympy as sp
import numpy as np
import math

from mechanics_dsl.domains.kinematics.equations import (
    KinematicEquation,
    KinematicEquations,
    KINEMATIC_SYMBOLS,
    EquationSelector,
    get_equation_by_name,
    get_equation_for_unknowns,
    list_all_equations,
    format_equation_table,
)


class TestKinematicSymbols:
    """Test that kinematic symbols are properly defined."""
    
    def test_all_symbols_exist(self):
        """Verify all 7 kinematic symbols are defined."""
        required = ['x0', 'x', 'v0', 'v', 'a', 't', 'dx']
        for name in required:
            assert name in KINEMATIC_SYMBOLS
            assert isinstance(KINEMATIC_SYMBOLS[name], sp.Symbol)
    
    def test_time_is_nonnegative(self):
        """Time symbol should be non-negative."""
        t = KINEMATIC_SYMBOLS['t']
        assert t.is_nonnegative is True


class TestKinematicEquation:
    """Test KinematicEquation class."""
    
    def test_equation_1_structure(self):
        """Test Equation 1: v = v₀ + at structure."""
        eq = KinematicEquations.equation_1()
        
        assert eq.name == "velocity-time"
        assert eq.number == 1
        assert eq.missing_variable == 'x'
        assert eq.variables == {'v', 'v0', 'a', 't'}
    
    def test_equation_2_structure(self):
        """Test Equation 2: x = x₀ + v₀t + ½at² structure."""
        eq = KinematicEquations.equation_2()
        
        assert eq.name == "position-time"
        assert eq.number == 2
        assert eq.missing_variable == 'v'
        assert 'x' in eq.variables
        assert 'x0' in eq.variables
    
    def test_equation_3_structure(self):
        """Test Equation 3: v² = v₀² + 2a(x - x₀) structure."""
        eq = KinematicEquations.equation_3()
        
        assert eq.name == "velocity-position"
        assert eq.number == 3
        assert eq.missing_variable == 't'
    
    def test_equation_4_structure(self):
        """Test Equation 4: x = x₀ + ½(v + v₀)t structure."""
        eq = KinematicEquations.equation_4()
        
        assert eq.name == "average-velocity"
        assert eq.number == 4
        assert eq.missing_variable == 'a'
    
    def test_equation_5_structure(self):
        """Test Equation 5: x = x₀ + vt - ½at² structure."""
        eq = KinematicEquations.equation_5()
        
        assert eq.name == "final-velocity-form"
        assert eq.number == 5
        assert eq.missing_variable == 'v0'
    
    def test_solve_for_velocity(self):
        """Test solving Equation 1 for velocity."""
        eq = KinematicEquations.equation_1()
        v_expr = eq.solve_for('v')
        
        # Substitute values: v0=0, a=10, t=2 → v=20
        result = float(v_expr.subs({
            KINEMATIC_SYMBOLS['v0']: 0,
            KINEMATIC_SYMBOLS['a']: 10,
            KINEMATIC_SYMBOLS['t']: 2,
        }))
        
        assert np.isclose(result, 20.0)
    
    def test_solve_for_time(self):
        """Test solving Equation 1 for time."""
        eq = KinematicEquations.equation_1()
        t_expr = eq.solve_for('t')
        
        # v=30, v0=10, a=5 → t=4
        result = float(t_expr.subs({
            KINEMATIC_SYMBOLS['v']: 30,
            KINEMATIC_SYMBOLS['v0']: 10,
            KINEMATIC_SYMBOLS['a']: 5,
        }))
        
        assert np.isclose(result, 4.0)
    
    def test_solve_for_invalid_variable(self):
        """Test that solving for missing variable raises error."""
        eq = KinematicEquations.equation_1()  # Missing x
        
        with pytest.raises(ValueError, match="not in equation"):
            eq.solve_for('x')
    
    def test_substitute_and_solve(self):
        """Test numerical substitution and solving."""
        eq = KinematicEquations.equation_2()
        
        # x = 0 + 5*2 + 0.5*2*4 = 10 + 4 = 14
        x = eq.substitute_and_solve('x', x0=0, v0=5, a=2, t=2)
        
        assert np.isclose(x, 14.0)
    
    def test_evaluate_satisfied_equation(self):
        """Test evaluating equation that is satisfied."""
        eq = KinematicEquations.equation_1()
        
        # v = v0 + at → 20 = 0 + 10*2 ✓
        residual = eq.evaluate(v=20, v0=0, a=10, t=2)
        
        assert np.isclose(residual, 0.0)
    
    def test_evaluate_unsatisfied_equation(self):
        """Test evaluating equation that is NOT satisfied."""
        eq = KinematicEquations.equation_1()
        
        # v = v0 + at → 25 ≠ 0 + 10*2 = 20
        residual = eq.evaluate(v=25, v0=0, a=10, t=2)
        
        assert np.isclose(residual, 5.0)


class TestKinematicEquationsCollection:
    """Test the KinematicEquations collection class."""
    
    def test_all_equations_returns_five(self):
        """Verify all_equations returns exactly 5 equations."""
        equations = KinematicEquations.all_equations()
        
        assert len(equations) == 5
    
    def test_get_equation_by_number(self):
        """Test getting equation by number."""
        for i in range(1, 6):
            eq = KinematicEquations.get_equation(i)
            assert eq.number == i
    
    def test_get_equation_invalid_number(self):
        """Test that invalid number raises error."""
        with pytest.raises(ValueError):
            KinematicEquations.get_equation(0)
        
        with pytest.raises(ValueError):
            KinematicEquations.get_equation(6)
    
    def test_find_equation_for_unknowns(self):
        """Test finding equation given known variables."""
        # If we know v0, a, t, we can find v using Equation 1
        knowns = {'v0', 'a', 't'}
        eq = KinematicEquations.find_equation_for_unknowns(knowns)
        
        assert eq is not None
        assert eq.number == 1
    
    def test_equations_containing(self):
        """Test finding equations containing specific variables."""
        # Equations containing both v and t
        equations = KinematicEquations.equations_containing('v', 't')
        
        assert len(equations) >= 2
        for eq in equations:
            assert 'v' in eq.variables
            assert 't' in eq.variables
    
    def test_equations_missing(self):
        """Test finding equations missing a specific variable."""
        # Equations not containing time
        equations = KinematicEquations.equations_missing('t')
        
        assert len(equations) == 1
        assert equations[0].number == 3  # v² = v₀² + 2aΔx


class TestEquationSelector:
    """Test equation selection logic."""
    
    def test_select_for_velocity_from_v0_a_t(self):
        """Test selecting equation to find v given v0, a, t."""
        selector = EquationSelector()
        
        eq = selector.select_for_unknown('v', {'v0', 'a', 't'})
        
        assert eq is not None
        assert eq.number == 1
    
    def test_select_for_position_from_v0_a_t(self):
        """Test selecting equation to find x given x0, v0, a, t."""
        selector = EquationSelector()
        
        eq = selector.select_for_unknown('x', {'x0', 'v0', 'a', 't'})
        
        assert eq is not None
        assert eq.number == 2
    
    def test_minimum_knowns_for_velocity(self):
        """Test finding minimum knowns needed for velocity."""
        selector = EquationSelector()
        
        min_sets = selector.minimum_knowns_for('v')
        
        assert len(min_sets) > 0
        # One option should be {v0, a, t}
        assert {'v0', 'a', 't'} in min_sets
    
    def test_solve_order_for_multiple_unknowns(self):
        """Test determining solve order for multiple unknowns."""
        selector = EquationSelector()
        
        # Given v0, a, t, solve for v and x
        order = selector.solve_order({'x0', 'v0', 'a', 't'}, ['v', 'x'])
        
        assert len(order) == 2
        # v can be found first from Eq1, then x from Eq2


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    def test_get_equation_by_name(self):
        """Test finding equation by name."""
        eq = get_equation_by_name("velocity-time")
        
        assert eq is not None
        assert eq.number == 1
    
    def test_get_equation_by_name_case_insensitive(self):
        """Test that name lookup is case-insensitive."""
        eq = get_equation_by_name("VELOCITY-TIME")
        
        assert eq is not None
        assert eq.number == 1
    
    def test_get_equation_by_name_not_found(self):
        """Test that invalid name returns None."""
        eq = get_equation_by_name("nonexistent")
        
        assert eq is None
    
    def test_get_equation_for_unknowns_wrapper(self):
        """Test convenience wrapper for equation finding."""
        eq = get_equation_for_unknowns({'v0', 'a', 't'})
        
        assert eq is not None
    
    def test_list_all_equations_format(self):
        """Test that list_all_equations returns proper format."""
        equations = list_all_equations()
        
        assert len(equations) == 5
        for eq_dict in equations:
            assert 'number' in eq_dict
            assert 'name' in eq_dict
            assert 'latex' in eq_dict
    
    def test_format_equation_table(self):
        """Test equation table formatting."""
        table = format_equation_table()
        
        assert "Equation 1" in table
        assert "Equation 5" in table
        assert "velocity-time" in table.lower()


class TestPhysicsProblems:
    """Test equations against known physics problems."""
    
    def test_free_fall_from_rest(self):
        """Object dropped from rest: find v after falling 45m with g=10."""
        eq = KinematicEquations.equation_3()
        
        # v² = v₀² + 2a(x - x₀) = 0 + 2*10*45 = 900 → v = 30
        v_squared = eq.substitute_and_solve('v', x0=0, x=-45, v0=0, a=-10)
        
        # Note: equation gives v², and sign depends on direction
        assert np.isclose(abs(v_squared), 30.0)
    
    def test_car_braking_distance(self):
        """Car at 30 m/s brakes at -5 m/s². Find stopping distance."""
        eq = KinematicEquations.equation_3()
        
        # 0 = 30² + 2*(-5)(x-0) → x = 900/10 = 90m
        x = eq.substitute_and_solve('x', x0=0, v=0, v0=30, a=-5)
        
        assert np.isclose(x, 90.0)
    
    def test_ball_thrown_up(self):
        """Ball thrown up at 20 m/s. Find max height (a=-10)."""
        eq = KinematicEquations.equation_3()
        
        # At max height v=0: 0 = 20² + 2*(-10)(h-0) → h = 400/20 = 20m
        h = eq.substitute_and_solve('x', x0=0, v=0, v0=20, a=-10)
        
        assert np.isclose(h, 20.0)
    
    def test_race_car_acceleration(self):
        """Race car accelerates from 0 to 100 m/s in 5s. Find displacement."""
        eq = KinematicEquations.equation_4()
        
        # x = 0 + ½(0 + 100)*5 = 250m
        x = eq.substitute_and_solve('x', x0=0, v0=0, v=100, t=5)
        
        assert np.isclose(x, 250.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
