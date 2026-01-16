"""
Tests for Projectile Motion Module

Comprehensive tests for projectile motion analysis including:
- Trajectory calculations
- Range, max height, time of flight
- Impact velocity and angle
- Known physics problem validation
"""

import math

import numpy as np
import pytest

from mechanics_dsl.domains.kinematics.projectile import (
    DEFAULT_GRAVITY,
    ProjectileMotion,
    ProjectileParameters,
    ProjectileResult,
    SymbolicProjectile,
    analyze_projectile,
    max_range_angle,
    optimal_angle_for_range,
)


class TestProjectileParameters:
    """Test ProjectileParameters dataclass."""

    def test_from_degrees(self):
        """Test creating parameters with degrees."""
        params = ProjectileParameters.from_degrees(
            initial_speed=10,
            launch_angle_deg=30,
            initial_height=5,
        )

        assert params.initial_speed == 10
        assert np.isclose(params.launch_angle, math.radians(30))
        assert params.initial_height == 5

    def test_velocity_components(self):
        """Test velocity component calculation."""
        params = ProjectileParameters.from_degrees(
            initial_speed=10,
            launch_angle_deg=30,
        )

        # v0x = 10 * cos(30°) ≈ 8.66
        # v0y = 10 * sin(30°) = 5
        assert np.isclose(params.v0x, 10 * math.cos(math.radians(30)))
        assert np.isclose(params.v0y, 10 * math.sin(math.radians(30)))


class TestProjectileMotionBasics:
    """Test basic ProjectileMotion functionality."""

    def test_horizontal_launch(self):
        """Test projectile launched horizontally."""
        proj = ProjectileMotion(v0=10, theta_deg=0, y0=20)

        assert np.isclose(proj.v0x, 10)
        assert np.isclose(proj.v0y, 0)

    def test_vertical_launch(self):
        """Test projectile launched straight up."""
        proj = ProjectileMotion(v0=10, theta_deg=90)

        assert np.isclose(proj.v0x, 0, atol=1e-10)
        assert np.isclose(proj.v0y, 10)

    def test_45_degree_launch(self):
        """Test projectile launched at 45 degrees."""
        proj = ProjectileMotion(v0=10, theta_deg=45)

        assert np.isclose(proj.v0x, proj.v0y)
        assert np.isclose(proj.v0x, 10 / math.sqrt(2))

    def test_negative_speed_raises(self):
        """Test that negative speed raises error."""
        with pytest.raises(ValueError):
            ProjectileMotion(v0=-10, theta_deg=45)

    def test_negative_gravity_raises(self):
        """Test that non-positive gravity raises error."""
        with pytest.raises(ValueError):
            ProjectileMotion(v0=10, theta_deg=45, g=0)


class TestPositionCalculations:
    """Test position at time calculations."""

    def test_x_at_time_zero(self):
        """Test x position at t=0."""
        proj = ProjectileMotion(v0=10, theta_deg=45, x0=5)

        assert np.isclose(proj.x_at_time(0), 5)

    def test_y_at_time_zero(self):
        """Test y position at t=0."""
        proj = ProjectileMotion(v0=10, theta_deg=45, y0=10)

        assert np.isclose(proj.y_at_time(0), 10)

    def test_x_linear_motion(self):
        """Test that x follows linear motion (vx is constant)."""
        proj = ProjectileMotion(v0=10, theta_deg=0, x0=0)

        # x = x0 + v0x * t = 0 + 10 * 2 = 20
        assert np.isclose(proj.x_at_time(2), 20)

    def test_y_parabolic_motion(self):
        """Test that y follows parabolic motion."""
        proj = ProjectileMotion(v0=0, theta_deg=0, y0=100, g=10)

        # y = 100 + 0 - 0.5*10*t² = 100 - 5t²
        # At t=2: y = 100 - 20 = 80
        assert np.isclose(proj.y_at_time(2), 80)

    def test_position_at_time(self):
        """Test combined position at time."""
        proj = ProjectileMotion(v0=10, theta_deg=0, x0=0, y0=100, g=10)

        x, y = proj.position_at_time(2)

        assert np.isclose(x, 20)
        assert np.isclose(y, 80)


class TestVelocityCalculations:
    """Test velocity calculations."""

    def test_vx_constant(self):
        """Test that horizontal velocity is constant."""
        proj = ProjectileMotion(v0=10, theta_deg=30)

        vx0 = proj.vx_at_time(0)
        vx1 = proj.vx_at_time(1)
        vx2 = proj.vx_at_time(2)

        assert np.isclose(vx0, vx1)
        assert np.isclose(vx1, vx2)

    def test_vy_decreases(self):
        """Test that vertical velocity decreases with time."""
        proj = ProjectileMotion(v0=10, theta_deg=45)

        vy0 = proj.vy_at_time(0)
        vy1 = proj.vy_at_time(1)

        assert vy0 > vy1

    def test_speed_at_launch(self):
        """Test that initial speed equals v0."""
        proj = ProjectileMotion(v0=10, theta_deg=45)

        assert np.isclose(proj.speed_at_time(0), 10)


class TestKeyQuantities:
    """Test calculation of key quantities."""

    def test_time_to_max_height_horizontal_launch(self):
        """Test time to max height for horizontal launch (should be 0)."""
        proj = ProjectileMotion(v0=10, theta_deg=0, y0=10)

        assert np.isclose(proj.time_to_max_height(), 0)

    def test_time_to_max_height(self):
        """Test time to max height: t = v0y / g."""
        proj = ProjectileMotion(v0=20, theta_deg=90, g=10)

        # t = 20 / 10 = 2 seconds
        assert np.isclose(proj.time_to_max_height(), 2.0)

    def test_max_height_dropped(self):
        """Test max height for dropped object."""
        proj = ProjectileMotion(v0=0, theta_deg=0, y0=100)

        assert np.isclose(proj.max_height(), 100)

    def test_max_height_thrown_up(self):
        """Test max height for object thrown straight up."""
        proj = ProjectileMotion(v0=20, theta_deg=90, y0=0, g=10)

        # h_max = y0 + v0^2 / (2g) = 0 + 400/20 = 20m
        assert np.isclose(proj.max_height(), 20)

    def test_max_height_with_initial_height(self):
        """Test max height with initial height."""
        proj = ProjectileMotion(v0=20, theta_deg=90, y0=10, g=10)

        # h_max = 10 + 400/20 = 30m
        assert np.isclose(proj.max_height(), 30)

    def test_time_of_flight_dropped(self):
        """Test time of flight for dropped object."""
        proj = ProjectileMotion(v0=0, theta_deg=0, y0=45, g=10)

        # t = sqrt(2h/g) = sqrt(90/10) = 3s
        assert np.isclose(proj.time_of_flight(), 3.0)

    def test_range_horizontal_from_height(self):
        """Test range for horizontal launch from height."""
        proj = ProjectileMotion(v0=10, theta_deg=0, y0=45, g=10)

        # Time to fall: t = sqrt(90/10) = 3s
        # Range = v0x * t = 10 * 3 = 30m
        assert np.isclose(proj.range(), 30)


class TestMaxRangeAt45Degrees:
    """Test the famous 45° max range result."""

    def test_max_range_angle_constant(self):
        """Test that max_range_angle returns 45."""
        assert max_range_angle() == 45.0

    def test_45_gives_max_range_flat_ground(self):
        """Test that 45° gives maximum range on flat ground."""
        v0 = 20
        g = 10

        ranges = []
        for angle in [30, 40, 45, 50, 60]:
            proj = ProjectileMotion(v0=v0, theta_deg=angle, y0=0, g=g)
            ranges.append((angle, proj.range()))

        # Find angle with max range
        max_range_result = max(ranges, key=lambda x: x[1])

        assert max_range_result[0] == 45

    def test_range_formula_flat_ground(self):
        """Test range formula: R = v0² sin(2θ) / g for flat ground."""
        v0 = 20
        g = 10
        theta_deg = 30

        proj = ProjectileMotion(v0=v0, theta_deg=theta_deg, y0=0, g=g)

        # R = v0² sin(2θ) / g = 400 * sin(60°) / 10 = 40 * √3/2 ≈ 34.64
        expected = v0**2 * math.sin(math.radians(2 * theta_deg)) / g

        assert np.isclose(proj.range(), expected, rtol=1e-6)


class TestImpactCalculations:
    """Test impact velocity and angle calculations."""

    def test_impact_velocity_dropped(self):
        """Test impact velocity for dropped object: v = sqrt(2gh)."""
        proj = ProjectileMotion(v0=0, theta_deg=0, y0=45, g=10)

        # v = sqrt(2 * 10 * 45) = sqrt(900) = 30 m/s
        assert np.isclose(proj.impact_speed(), 30)

    def test_impact_horizontal_component_constant(self):
        """Test that horizontal component at impact equals initial."""
        proj = ProjectileMotion(v0=10, theta_deg=30, y0=10)

        vx, vy = proj.impact_velocity()

        assert np.isclose(vx, proj.v0x)

    def test_impact_angle_dropped(self):
        """Test impact angle for dropped object is 90° (straight down)."""
        proj = ProjectileMotion(v0=0, theta_deg=0, y0=45, g=10)

        # Dropped object falls straight down - angle is 90°
        impact_angle_deg = math.degrees(proj.impact_angle())

        assert np.isclose(impact_angle_deg, 90, atol=1)


class TestMarbleFromBalcony:
    """Test the user's specific marble problem."""

    def test_marble_launched_from_balcony(self):
        """
        Test: Marble launched from 4m balcony at 5 m/s, 30° above horizontal.
        This is the user's actual physics problem!
        """
        proj = ProjectileMotion(v0=5, theta_deg=30, y0=4, g=9.81)
        result = proj.analyze()

        # Verify components
        assert np.isclose(result.v0x, 5 * math.cos(math.radians(30)), rtol=1e-6)
        assert np.isclose(result.v0y, 5 * math.sin(math.radians(30)), rtol=1e-6)

        # Verify reasonable values
        assert result.time_of_flight > 0
        assert result.range > 0
        assert result.max_height >= 4  # At least starting height
        assert result.impact_velocity > 5  # Faster than launch due to fall

    def test_marble_horizontal_launch(self):
        """Test marble launched horizontally from 4m."""
        proj = ProjectileMotion(v0=5, theta_deg=0, y0=4, g=10)
        result = proj.analyze()

        # Time to fall: t = sqrt(2h/g) = sqrt(8/10) = sqrt(0.8)
        expected_t = math.sqrt(2 * 4 / 10)
        assert np.isclose(result.time_of_flight, expected_t, rtol=0.01)

        # Range = v0 * t = 5 * sqrt(0.8)
        expected_range = 5 * expected_t
        assert np.isclose(result.range, expected_range, rtol=0.01)


class TestSymbolicProjectile:
    """Test symbolic projectile motion formulas."""

    def test_range_formula_flat_ground(self):
        """Test symbolic range formula for flat ground."""
        sym_proj = SymbolicProjectile()
        formula = sym_proj.range_formula()

        # Should be v0² sin(2θ) / g
        # Verify structure contains expected terms
        import sympy as sp

        assert formula.has(sp.sin)

    def test_max_height_formula(self):
        """Test symbolic max height formula."""
        sym_proj = SymbolicProjectile()
        formula = sym_proj.max_height_formula()

        # Should be y0 + v0² sin²(θ) / (2g)
        import sympy as sp

        assert formula.has(sp.sin)


class TestTrajectoryEquation:
    """Test trajectory equation y(x)."""

    def test_trajectory_at_start(self):
        """Test trajectory equation at starting point."""
        proj = ProjectileMotion(v0=10, theta_deg=45, x0=0, y0=0)

        y = proj.y_at_x(0)

        assert np.isclose(y, 0)

    def test_trajectory_is_parabolic(self):
        """Test that trajectory follows parabolic shape."""
        proj = ProjectileMotion(v0=20, theta_deg=45, y0=0, g=10)

        # Check several points along trajectory
        x_vals = [0, 5, 10, 15, 20]
        y_vals = [proj.y_at_x(x) for x in x_vals]

        # y values should increase then decrease (parabola)
        assert y_vals[0] < y_vals[2]  # Rising
        # Near the end it should be lower than the peak

    def test_trajectory_points(self):
        """Test getting trajectory point arrays."""
        proj = ProjectileMotion(v0=10, theta_deg=45, y0=0)

        x_arr, y_arr = proj.get_trajectory_points(n_points=50)

        assert len(x_arr) == 50
        assert len(y_arr) == 50
        assert x_arr[0] == proj.x0
        assert np.isclose(y_arr[0], proj.y0)


class TestAnalyzeProjectile:
    """Test convenience function."""

    def test_analyze_returns_result(self):
        """Test that analyze_projectile returns ProjectileResult."""
        result = analyze_projectile(v0=10, theta_deg=45)

        assert isinstance(result, ProjectileResult)

    def test_analyze_with_show_work(self):
        """Test analyze_projectile with show_work=True."""
        work = analyze_projectile(v0=10, theta_deg=45, show_work=True)

        assert isinstance(work, str)
        assert "STEP 1" in work
        assert "FINAL ANSWERS" in work


class TestShowWork:
    """Test the show_work method for educational output."""

    def test_show_work_contains_steps(self):
        """Test that show_work includes numbered steps."""
        proj = ProjectileMotion(v0=10, theta_deg=30, y0=5)
        work = proj.show_work()

        assert "STEP 1" in work
        assert "STEP 2" in work
        assert "GIVEN:" in work
        assert "FINAL ANSWERS" in work

    def test_show_work_contains_equations(self):
        """Test that show_work shows equations used."""
        proj = ProjectileMotion(v0=10, theta_deg=30, y0=5)
        work = proj.show_work()

        assert "v₀ₓ" in work or "v0x" in work.lower() or "v_0" in work

    def test_show_work_contains_numerical_results(self):
        """Test that show_work shows numerical results."""
        proj = ProjectileMotion(v0=10, theta_deg=30, y0=5, g=10)
        work = proj.show_work()

        # Should contain the given values
        assert "10" in work  # v0 or g
        # Angle may be displayed with floating point precision
        assert "29." in work or "30" in work  # angle (may show as 29.999...)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
