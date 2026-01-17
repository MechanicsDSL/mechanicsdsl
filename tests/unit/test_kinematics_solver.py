"""
Tests for Kinematics Solver Module

Comprehensive unit tests for the analytical kinematics solver.
"""

import numpy as np
import pytest

from mechanics_dsl.domains.kinematics.solver import (
    KinematicsSolver,
    KinematicState,
    SymbolicKinematicsSolver,
    solve_kinematics,
    verify_kinematics,
)


class TestKinematicState:
    """Test KinematicState dataclass."""

    def test_default_state(self):
        """Test default state creation."""
        state = KinematicState()

        assert state.initial_position == 0.0
        assert state.final_position is None

    def test_state_with_values(self):
        """Test state with provided values."""
        state = KinematicState(
            initial_position=0,
            final_position=100,
            displacement=100,  # Need to explicitly provide displacement
            initial_velocity=10,
            final_velocity=30,
            acceleration=20,
            time=1,
        )

        assert state.is_complete
        assert state.known_count == 6

    def test_incomplete_state(self):
        """Test that incomplete state is detected."""
        state = KinematicState(
            initial_position=0,
            initial_velocity=10,
        )

        assert not state.is_complete

    def test_get_knowns(self):
        """Test getting known values."""
        state = KinematicState(
            initial_position=5,
            initial_velocity=10,
            acceleration=2,
        )

        knowns = state.get_knowns()

        assert "x0" in knowns
        assert "v0" in knowns
        assert "a" in knowns
        assert knowns["x0"] == 5

    def test_to_dict(self):
        """Test converting state to dictionary."""
        state = KinematicState(initial_position=0)

        d = state.to_dict()

        assert "x0" in d
        assert "x" in d
        assert "v0" in d

    def test_velocity_alias(self):
        """Test that velocity is alias for final_velocity."""
        state = KinematicState(final_velocity=25)

        assert state.velocity == 25


class TestKinematicsSolver:
    """Test KinematicsSolver class."""

    def test_solve_velocity_from_v0_a_t(self):
        """Test solving for final velocity given v0, a, t."""
        solver = KinematicsSolver()

        # v = v0 + at = 0 + 10*2 = 20
        solution = solver.solve(v0=0, a=10, t=2)

        assert solution.success
        assert np.isclose(solution.state.final_velocity, 20)

    def test_solve_position_from_v0_a_t(self):
        """Test solving for position given v0, a, t."""
        solver = KinematicsSolver()

        # x = x0 + v0*t + 0.5*a*t² = 0 + 0 + 0.5*10*4 = 20
        solution = solver.solve(x0=0, v0=0, a=10, t=2)

        assert solution.success
        assert np.isclose(solution.state.final_position, 20)

    def test_solve_time_from_v0_v_a(self):
        """Test solving for time given v0, v, a."""
        solver = KinematicsSolver()

        # v = v0 + at → 30 = 10 + 5t → t = 4
        solution = solver.solve(v0=10, v=30, a=5)

        assert solution.success
        assert np.isclose(solution.state.time, 4.0)

    def test_solve_acceleration_from_v0_v_t(self):
        """Test solving for acceleration given v0, v, t."""
        solver = KinematicsSolver()

        # v = v0 + at → 25 = 5 + a*4 → a = 5
        solution = solver.solve(v0=5, v=25, t=4)

        assert solution.success
        assert np.isclose(solution.state.acceleration, 5.0)

    def test_solve_with_displacement(self):
        """Test solving with displacement input."""
        solver = KinematicsSolver()

        # dx = 20, v0 = 0, a = 10 → should give t and v
        # Need 4 knowns including x0 for complete solution
        solution = solver.solve(dx=20, v0=0, a=10)

        # May not be complete but should at least find some values
        assert solution.state.final_position is not None

    def test_insufficient_information(self):
        """Test that insufficient info returns unsuccessful."""
        solver = KinematicsSolver()

        # Only v0 and a - not enough to solve
        solution = solver.solve(v0=10, a=5)

        assert not solution.success

    def test_negative_time_validation(self):
        """Test that negative time is rejected."""
        solver = KinematicsSolver()

        solution = solver.solve(v0=0, a=10, t=-2)

        assert not solution.success

    def test_solution_shows_work(self):
        """Test that solution has step-by-step work."""
        solver = KinematicsSolver()

        solution = solver.solve(v0=0, a=10, t=2)

        assert len(solution.steps) > 0
        assert len(solution.equations_used) > 0

    def test_solution_step_structure(self):
        """Test that solution steps have proper structure."""
        solver = KinematicsSolver()

        solution = solver.solve(v0=0, a=10, t=2)

        for step in solution.steps:
            assert step.equation_used is not None
            assert step.solving_for in ["x", "v", "a", "t", "x0", "v0"]
            assert step.result is not None

    def test_show_work_output(self):
        """Test show_work returns formatted string."""
        solver = KinematicsSolver()

        solution = solver.solve(v0=0, a=10, t=2)
        work = solution.show_work()

        assert isinstance(work, str)
        assert "GIVEN" in work
        assert "SOLUTION STEPS" in work
        assert "FINAL ANSWERS" in work


class TestSolverPhysicsProblems:
    """Test solver against real physics problems."""

    def test_free_fall_from_rest(self):
        """Test free fall: dropped from 80m, find time and impact velocity."""
        solver = KinematicsSolver()

        # y = y0 + v0*t - 0.5*g*t²
        # 0 = 80 + 0 - 0.5*10*t² → t = 4s
        # v = v0 - gt = 0 - 10*4 = -40 m/s (or 40 m/s down)
        solution = solver.solve(x0=80, x=0, v0=0, a=-10)

        assert solution.success
        assert np.isclose(solution.state.time, 4.0)
        assert np.isclose(solution.state.final_velocity, -40)

    def test_car_braking(self):
        """Test car braking: v0=30m/s, a=-5m/s², find stopping distance."""
        solver = KinematicsSolver()

        # v² = v0² + 2ax → 0 = 900 + 2*(-5)*x → x = 90m
        solution = solver.solve(v0=30, v=0, a=-5)

        assert solution.success
        assert np.isclose(solution.state.displacement, 90)

    def test_rocket_acceleration(self):
        """Test rocket: accelerates from rest at 20m/s² for 5s."""
        solver = KinematicsSolver()

        solution = solver.solve(v0=0, a=20, t=5)

        assert solution.success
        # v = 0 + 20*5 = 100 m/s
        assert np.isclose(solution.state.final_velocity, 100)
        # x = 0 + 0 + 0.5*20*25 = 250 m
        assert np.isclose(solution.state.displacement, 250)

    def test_thrown_ball_max_height(self):
        """Test ball thrown up: v0=30m/s, a=-10m/s², v=0 at top."""
        solver = KinematicsSolver()

        solution = solver.solve(v0=30, v=0, a=-10)

        assert solution.success
        # t = (v - v0)/a = -30/-10 = 3s
        assert np.isclose(solution.state.time, 3.0)
        # h = v0*t + 0.5*a*t² = 90 - 45 = 45m
        assert np.isclose(solution.state.displacement, 45)


class TestSymbolicKinematicsSolver:
    """Test symbolic solver for formula derivation."""

    def test_derive_velocity_formula(self):
        """Test deriving velocity formula."""
        solver = SymbolicKinematicsSolver()

        formula = solver.derive_formula("v", knowns=["v0", "a", "t"])

        # Should be v = v0 + at
        # The formula should contain v0, a, and t symbols
        formula_str = str(formula)
        assert "v_0" in formula_str or "v0" in formula_str.lower()
        assert "a" in formula_str
        assert "t" in formula_str

    def test_derive_position_formula(self):
        """Test deriving position formula."""
        solver = SymbolicKinematicsSolver()

        formula = solver.derive_formula("x", knowns=["x0", "v0", "a", "t"])

        # Should be x = x0 + v0*t + 0.5*a*t²

        assert formula is not None

    def test_displacement_formulas(self):
        """Test getting all displacement formulas."""
        solver = SymbolicKinematicsSolver()

        formulas = solver.derive_displacement_formulas()

        assert "from_v0_a_t" in formulas
        assert "from_v_a_t" in formulas

    def test_velocity_formulas(self):
        """Test getting all velocity formulas."""
        solver = SymbolicKinematicsSolver()

        formulas = solver.derive_velocity_formulas()

        assert "from_v0_a_t" in formulas

    def test_time_formulas(self):
        """Test getting all time formulas."""
        solver = SymbolicKinematicsSolver()

        formulas = solver.derive_time_formulas()

        assert "from_v0_v_a" in formulas


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_solve_kinematics_basic(self):
        """Test solve_kinematics convenience function."""
        solution = solve_kinematics(v0=0, a=10, t=2)

        assert solution.success
        assert np.isclose(solution.state.final_velocity, 20)

    def test_solve_kinematics_show_work(self):
        """Test solve_kinematics with show_work=True."""
        work = solve_kinematics(v0=0, a=10, t=2, show_work=True)

        assert isinstance(work, str)
        assert "FINAL ANSWERS" in work

    def test_verify_kinematics_valid(self):
        """Test verifying valid kinematic values."""
        # These values satisfy v = v0 + at
        # 20 = 0 + 10*2 ✓
        # And x = x0 + v0*t + 0.5*a*t² = 0 + 0 + 20 = 20 ✓
        is_valid, residuals = verify_kinematics(x0=0, x=20, v0=0, v=20, a=10, t=2)

        assert is_valid
        for name, residual in residuals.items():
            assert abs(residual) < 1e-6

    def test_verify_kinematics_invalid(self):
        """Test verifying invalid kinematic values."""
        # v = v0 + at → v ≠ 0 + 10*2 = 20 (but we say v=25)
        is_valid, residuals = verify_kinematics(x0=0, x=20, v0=0, v=25, a=10, t=2)

        assert not is_valid


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_acceleration(self):
        """Test with zero acceleration (uniform motion)."""
        solver = KinematicsSolver()

        solution = solver.solve(v0=10, a=0, t=5)

        assert solution.success
        assert np.isclose(solution.state.final_velocity, 10)
        assert np.isclose(solution.state.displacement, 50)

    def test_zero_time(self):
        """Test with zero time."""
        solver = KinematicsSolver()

        solution = solver.solve(v0=10, a=5, t=0)

        assert solution.success
        assert np.isclose(solution.state.final_velocity, 10)
        assert np.isclose(solution.state.displacement, 0)

    def test_zero_initial_velocity(self):
        """Test starting from rest."""
        solver = KinematicsSolver()

        solution = solver.solve(v0=0, a=10, t=3)

        assert solution.success
        assert np.isclose(solution.state.final_velocity, 30)

    def test_negative_acceleration(self):
        """Test with deceleration."""
        solver = KinematicsSolver()

        solution = solver.solve(v0=30, a=-10, t=3)

        assert solution.success
        assert np.isclose(solution.state.final_velocity, 0)


class TestMultipleSteps:
    """Test problems requiring multiple solution steps."""

    def test_finds_all_unknowns(self):
        """Test that solver finds all unknown values."""
        solver = KinematicsSolver()

        solution = solver.solve(x0=0, v0=10, a=2, t=5)

        assert solution.success
        assert solution.state.is_complete

        # All values should be computed
        assert solution.state.final_velocity is not None
        assert solution.state.final_position is not None
        assert solution.state.displacement is not None

    def test_solution_order_tracked(self):
        """Test that all solution steps are tracked."""
        solver = KinematicsSolver()

        solution = solver.solve(x0=0, v0=10, a=2, t=5)

        # Should have solved for v and x
        solved_vars = {step.solving_for for step in solution.steps}

        assert "v" in solved_vars or "x" in solved_vars


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
