"""
Projectile Motion Analysis for MechanicsDSL

This module provides comprehensive tools for analyzing projectile motion
problems, including:
- Complete trajectory analysis
- Range, maximum height, time of flight calculations
- Velocity components at any time
- Landing position and impact velocity
- Trajectory equations and plotting

Projectile motion assumes:
- Constant gravitational acceleration (typically g = 9.81 m/s²)
- No air resistance
- Motion in the x-y plane (x horizontal, y vertical)
- Positive y is upward

Key Equations:
    x(t) = x₀ + v₀ₓt           (horizontal position)
    y(t) = y₀ + v₀ᵧt - ½gt²    (vertical position)
    vₓ(t) = v₀ₓ                (horizontal velocity is constant)
    vᵧ(t) = v₀ᵧ - gt           (vertical velocity changes)

Example:
    >>> from mechanics_dsl.domains.kinematics import ProjectileMotion
    >>>
    >>> # Marble launched from 4m height at 5 m/s, 30° above horizontal
    >>> proj = ProjectileMotion(v0=5.0, theta_deg=30.0, y0=4.0)
    >>> result = proj.analyze()
    >>>
    >>> print(f"Time of flight: {result.time_of_flight:.3f} s")
    >>> print(f"Range: {result.range:.3f} m")
    >>> print(f"Max height: {result.max_height:.3f} m")
    >>> print(f"Impact velocity: {result.impact_velocity:.3f} m/s")
"""

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import sympy as sp


# ============================================================================
# CONSTANTS
# ============================================================================

STANDARD_GRAVITY = 9.80665  # m/s² (standard gravity)
DEFAULT_GRAVITY = 9.81  # m/s² (commonly used approximation)


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class ProjectileParameters:
    """
    Input parameters for a projectile motion problem.

    Attributes:
        initial_speed: Launch speed |v₀| in m/s
        launch_angle: Launch angle θ above horizontal in radians
        initial_height: Initial height y₀ in m (default 0)
        initial_x: Initial horizontal position x₀ in m (default 0)
        gravity: Gravitational acceleration g in m/s² (default 9.81)
    """

    initial_speed: float
    launch_angle: float  # radians
    initial_height: float = 0.0
    initial_x: float = 0.0
    gravity: float = DEFAULT_GRAVITY

    @classmethod
    def from_degrees(
        cls,
        initial_speed: float,
        launch_angle_deg: float,
        initial_height: float = 0.0,
        initial_x: float = 0.0,
        gravity: float = DEFAULT_GRAVITY,
    ) -> "ProjectileParameters":
        """Create parameters with angle in degrees."""
        return cls(
            initial_speed=initial_speed,
            launch_angle=math.radians(launch_angle_deg),
            initial_height=initial_height,
            initial_x=initial_x,
            gravity=gravity,
        )

    @property
    def v0x(self) -> float:
        """Initial horizontal velocity component."""
        return self.initial_speed * math.cos(self.launch_angle)

    @property
    def v0y(self) -> float:
        """Initial vertical velocity component."""
        return self.initial_speed * math.sin(self.launch_angle)

    @property
    def launch_angle_deg(self) -> float:
        """Launch angle in degrees."""
        return math.degrees(self.launch_angle)


@dataclass
class ProjectileResult:
    """
    Complete results of a projectile motion analysis.

    All positions, velocities, and times characterizing the projectile's
    motion from launch to landing.

    Attributes:
        # Initial conditions
        v0: Initial speed
        v0x: Initial horizontal velocity
        v0y: Initial vertical velocity
        x0: Initial x position
        y0: Initial y position (height)
        theta: Launch angle (radians)
        theta_deg: Launch angle (degrees)
        g: Gravitational acceleration

        # Key results
        time_of_flight: Total time in air
        range: Horizontal distance traveled (landing x - x0)
        max_height: Maximum y coordinate reached
        time_to_max_height: Time to reach apex

        # At maximum height
        apex_x: x position at maximum height
        apex_y: y position at maximum height (= max_height)

        # At impact
        impact_x: Landing x position
        impact_y: Landing y position (typically 0)
        impact_velocity: Speed at impact
        impact_vx: Horizontal velocity at impact
        impact_vy: Vertical velocity at impact
        impact_angle: Angle below horizontal at impact (radians)
        impact_angle_deg: Angle below horizontal at impact (degrees)

        # Equations used
        equations_x: List of equations used for x motion
        equations_y: List of equations used for y motion
    """

    # Initial conditions
    v0: float
    v0x: float
    v0y: float
    x0: float
    y0: float
    theta: float
    theta_deg: float
    g: float

    # Key results
    time_of_flight: float
    range: float
    max_height: float
    time_to_max_height: float

    # At apex
    apex_x: float
    apex_y: float

    # At impact
    impact_x: float
    impact_y: float
    impact_velocity: float
    impact_vx: float
    impact_vy: float
    impact_angle: float
    impact_angle_deg: float

    # Equations used
    equations_x: List[str] = field(default_factory=list)
    equations_y: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a formatted summary of results."""
        lines = [
            "=" * 60,
            "PROJECTILE MOTION ANALYSIS RESULTS",
            "=" * 60,
            "",
            "INITIAL CONDITIONS:",
            f"  Initial speed (v₀):        {self.v0:.4g} m/s",
            f"  Launch angle (θ):          {self.theta_deg:.2f}°",
            f"  Initial position:          ({self.x0:.4g}, {self.y0:.4g}) m",
            f"  Initial velocity:          ({self.v0x:.4g}, {self.v0y:.4g}) m/s",
            f"  Gravity (g):               {self.g:.4g} m/s²",
            "",
            "KEY RESULTS:",
            f"  Time of flight:            {self.time_of_flight:.4g} s",
            f"  Horizontal range:          {self.range:.4g} m",
            f"  Maximum height:            {self.max_height:.4g} m",
            f"  Time to max height:        {self.time_to_max_height:.4g} s",
            "",
            "AT APEX (maximum height):",
            f"  Position:                  ({self.apex_x:.4g}, {self.apex_y:.4g}) m",
            f"  Velocity:                  ({self.v0x:.4g}, 0) m/s",
            "",
            "AT IMPACT:",
            f"  Position:                  ({self.impact_x:.4g}, {self.impact_y:.4g}) m",
            f"  Velocity:                  ({self.impact_vx:.4g}, {self.impact_vy:.4g}) m/s",
            f"  Impact speed:              {self.impact_velocity:.4g} m/s",
            f"  Impact angle:              {self.impact_angle_deg:.2f}° below horizontal",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return (
            f"ProjectileResult(range={self.range:.3g}m, "
            f"max_height={self.max_height:.3g}m, "
            f"time={self.time_of_flight:.3g}s)"
        )


# ============================================================================
# PROJECTILE MOTION ANALYZER
# ============================================================================


class ProjectileMotion:
    """
    Complete analyzer for projectile motion problems.

    This class handles all aspects of 2D projectile motion under constant
    gravitational acceleration, providing:

    - Trajectory analysis (position at any time)
    - Velocity analysis (velocity at any time)
    - Key quantities (range, max height, time of flight)
    - Impact analysis (landing position, velocity, angle)
    - Trajectory plotting

    The motion is decomposed into independent horizontal (constant velocity)
    and vertical (uniformly accelerated) components.

    Mathematical Model:
        Horizontal: x(t) = x₀ + v₀cos(θ)t
        Vertical:   y(t) = y₀ + v₀sin(θ)t - ½gt²

    Example:
        >>> # Marble launched from a 4m balcony at 5 m/s, 30° above horizontal
        >>> proj = ProjectileMotion(v0=5.0, theta_deg=30.0, y0=4.0)
        >>> result = proj.analyze()
        >>>
        >>> print(f"The marble will land {result.range:.2f} m away")
        >>> print(f"after {result.time_of_flight:.2f} seconds")
        >>> print(f"hitting the ground at {result.impact_velocity:.2f} m/s")

    Args:
        v0: Initial speed in m/s
        theta_deg: Launch angle in degrees above horizontal
        y0: Initial height in m (default 0)
        x0: Initial horizontal position in m (default 0)
        g: Gravitational acceleration in m/s² (default 9.81)
        theta_rad: Launch angle in radians (alternative to theta_deg)
    """

    def __init__(
        self,
        v0: float,
        theta_deg: Optional[float] = None,
        y0: float = 0.0,
        x0: float = 0.0,
        g: float = DEFAULT_GRAVITY,
        theta_rad: Optional[float] = None,
    ):
        """
        Initialize the projectile motion analyzer.

        Args:
            v0: Initial speed (must be non-negative)
            theta_deg: Launch angle in degrees (default 0)
            y0: Initial height (default 0)
            x0: Initial x position (default 0)
            g: Gravity magnitude (default 9.81 m/s²)
            theta_rad: Alternative: angle in radians
        """
        if v0 < 0:
            raise ValueError(f"Initial speed must be non-negative, got {v0}")
        if g <= 0:
            raise ValueError(f"Gravity must be positive, got {g}")

        # Handle angle input
        if theta_rad is not None:
            self.theta = theta_rad
        elif theta_deg is not None:
            self.theta = math.radians(theta_deg)
        else:
            self.theta = 0.0  # Horizontal launch

        self.v0 = v0
        self.y0 = y0
        self.x0 = x0
        self.g = g

        # Compute initial velocity components
        self.v0x = v0 * math.cos(self.theta)
        self.v0y = v0 * math.sin(self.theta)

        # Store parameters for reference
        self.params = ProjectileParameters(
            initial_speed=v0,
            launch_angle=self.theta,
            initial_height=y0,
            initial_x=x0,
            gravity=g,
        )

    # ========================================================================
    # POSITION METHODS
    # ========================================================================

    def x_at_time(self, t: float) -> float:
        """
        Calculate horizontal position at time t.

        Uses Equation 2 with a=0: x = x₀ + v₀ₓt

        Args:
            t: Time in seconds (must be non-negative)

        Returns:
            Horizontal position in meters
        """
        if t < 0:
            raise ValueError(f"Time must be non-negative, got {t}")
        return self.x0 + self.v0x * t

    def y_at_time(self, t: float) -> float:
        """
        Calculate vertical position at time t.

        Uses Equation 2: y = y₀ + v₀ᵧt - ½gt²

        Args:
            t: Time in seconds

        Returns:
            Vertical position in meters
        """
        if t < 0:
            raise ValueError(f"Time must be non-negative, got {t}")
        return self.y0 + self.v0y * t - 0.5 * self.g * t**2

    def position_at_time(self, t: float) -> Tuple[float, float]:
        """
        Calculate (x, y) position at time t.

        Args:
            t: Time in seconds

        Returns:
            Tuple of (x, y) positions in meters
        """
        return self.x_at_time(t), self.y_at_time(t)

    # ========================================================================
    # VELOCITY METHODS
    # ========================================================================

    def vx_at_time(self, t: float) -> float:
        """
        Calculate horizontal velocity at time t.

        For projectile motion, horizontal velocity is constant: vₓ = v₀ₓ

        Args:
            t: Time in seconds (not used, included for API consistency)

        Returns:
            Horizontal velocity in m/s
        """
        return self.v0x

    def vy_at_time(self, t: float) -> float:
        """
        Calculate vertical velocity at time t.

        Uses Equation 1: vᵧ = v₀ᵧ - gt

        Args:
            t: Time in seconds

        Returns:
            Vertical velocity in m/s (positive = upward)
        """
        return self.v0y - self.g * t

    def velocity_at_time(self, t: float) -> Tuple[float, float]:
        """
        Calculate (vx, vy) velocity at time t.

        Args:
            t: Time in seconds

        Returns:
            Tuple of (vx, vy) velocities in m/s
        """
        return self.vx_at_time(t), self.vy_at_time(t)

    def speed_at_time(self, t: float) -> float:
        """
        Calculate speed (magnitude of velocity) at time t.

        Args:
            t: Time in seconds

        Returns:
            Speed in m/s
        """
        vx, vy = self.velocity_at_time(t)
        return math.sqrt(vx**2 + vy**2)

    def direction_at_time(self, t: float) -> float:
        """
        Calculate velocity direction at time t.

        Args:
            t: Time in seconds

        Returns:
            Angle in radians from horizontal (positive = above)
        """
        vx, vy = self.velocity_at_time(t)
        return math.atan2(vy, vx)

    # ========================================================================
    # KEY QUANTITIES
    # ========================================================================

    def time_to_max_height(self) -> float:
        """
        Calculate time to reach maximum height.

        At maximum height, vᵧ = 0:
        0 = v₀ᵧ - gt_apex → t_apex = v₀ᵧ/g

        Returns:
            Time to apex in seconds
        """
        if self.v0y <= 0:
            return 0.0  # Ball is moving downward from start
        return self.v0y / self.g

    def max_height(self) -> float:
        """
        Calculate maximum height reached.

        Uses Equation 3 with v=0: 0 = v₀ᵧ² - 2g(y_max - y₀)
        → y_max = y₀ + v₀ᵧ²/(2g)

        Returns:
            Maximum height in meters
        """
        t_apex = self.time_to_max_height()
        if t_apex == 0:
            return self.y0  # Starting point is highest for downward launch
        return self.y_at_time(t_apex)

    def apex_position(self) -> Tuple[float, float]:
        """
        Calculate position at apex (maximum height).

        Returns:
            Tuple of (x, y) at apex
        """
        t = self.time_to_max_height()
        return self.position_at_time(t)

    def time_of_flight(self, y_final: float = 0.0) -> float:
        """
        Calculate total time of flight until projectile reaches y_final.

        Solves: y_final = y₀ + v₀ᵧt - ½gt²
        This is quadratic: -½gt² + v₀ᵧt + (y₀ - y_final) = 0

        Args:
            y_final: Final y position (default 0, ground level)

        Returns:
            Time of flight in seconds (the positive root)
        """
        # Quadratic formula: t = (-b ± √(b² - 4ac)) / 2a
        # where a = -g/2, b = v0y, c = y0 - y_final
        a = -0.5 * self.g
        b = self.v0y
        c = self.y0 - y_final

        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            raise ValueError(
                f"Projectile never reaches y = {y_final}. "
                f"Minimum height is {self.y0} (when launched downward) or check trajectory."
            )

        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b + sqrt_disc) / (2 * a)
        t2 = (-b - sqrt_disc) / (2 * a)

        # Take the positive root (future time)
        # For projectile, we typically want the larger positive root
        # (the time when it comes back down)
        times = [t for t in [t1, t2] if t >= 0]

        if not times:
            raise ValueError("No valid positive time found")

        return max(times)  # Return larger positive time (landing time)

    def range(self, y_final: float = 0.0) -> float:
        """
        Calculate horizontal range.

        Range = horizontal distance from x₀ to landing point.

        Args:
            y_final: Final y position (default 0)

        Returns:
            Horizontal range in meters
        """
        t = self.time_of_flight(y_final)
        return self.x_at_time(t) - self.x0

    def impact_velocity(self, y_final: float = 0.0) -> Tuple[float, float]:
        """
        Calculate velocity at impact.

        Args:
            y_final: Landing height

        Returns:
            Tuple of (vx, vy) at impact
        """
        t = self.time_of_flight(y_final)
        return self.velocity_at_time(t)

    def impact_speed(self, y_final: float = 0.0) -> float:
        """
        Calculate speed at impact.

        Args:
            y_final: Landing height

        Returns:
            Impact speed in m/s
        """
        vx, vy = self.impact_velocity(y_final)
        return math.sqrt(vx**2 + vy**2)

    def impact_angle(self, y_final: float = 0.0) -> float:
        """
        Calculate impact angle (below horizontal).

        Args:
            y_final: Landing height

        Returns:
            Impact angle in radians (positive = below horizontal)
        """
        vx, vy = self.impact_velocity(y_final)
        # Angle below horizontal is positive
        return -math.atan2(vy, vx)

    # ========================================================================
    # TRAJECTORY EQUATION
    # ========================================================================

    def trajectory_equation_y_of_x(self) -> Callable[[float], float]:
        """
        Get the parabolic trajectory equation y(x).

        Eliminates time to get: y = y₀ + tan(θ)(x-x₀) - g(x-x₀)²/(2v₀²cos²θ)

        Returns:
            Function that takes x and returns y
        """

        def y_of_x(x: float) -> float:
            dx = x - self.x0
            if abs(self.v0x) < 1e-10:
                # Vertical launch - no horizontal motion
                return float("nan")
            return self.y0 + (self.v0y / self.v0x) * dx - (self.g * dx**2) / (2 * self.v0x**2)

        return y_of_x

    def y_at_x(self, x: float) -> float:
        """
        Calculate y position at a given x position.

        Args:
            x: Horizontal position

        Returns:
            Vertical position
        """
        return self.trajectory_equation_y_of_x()(x)

    def time_at_x(self, x: float) -> float:
        """
        Calculate time when projectile reaches x position.

        Args:
            x: Target horizontal position

        Returns:
            Time in seconds
        """
        if abs(self.v0x) < 1e-10:
            raise ValueError("Cannot calculate time_at_x for vertical launch")
        return (x - self.x0) / self.v0x

    def time_at_y(self, y: float) -> List[float]:
        """
        Calculate time(s) when projectile is at height y.

        May have 0, 1, or 2 solutions.

        Args:
            y: Target height

        Returns:
            List of times (may be empty, 1, or 2 elements)
        """
        # Solve: y = y₀ + v₀ᵧt - ½gt²
        a = -0.5 * self.g
        b = self.v0y
        c = self.y0 - y

        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            return []
        elif discriminant == 0:
            t = -b / (2 * a)
            return [t] if t >= 0 else []
        else:
            sqrt_disc = math.sqrt(discriminant)
            t1 = (-b + sqrt_disc) / (2 * a)
            t2 = (-b - sqrt_disc) / (2 * a)
            return sorted([t for t in [t1, t2] if t >= 0])

    # ========================================================================
    # TRAJECTORY DATA
    # ========================================================================

    def get_trajectory_points(
        self,
        n_points: int = 100,
        y_final: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get arrays of (x, y) points along the trajectory.

        Args:
            n_points: Number of points to generate
            y_final: Final y position for time calculation

        Returns:
            Tuple of (x_array, y_array) numpy arrays
        """
        t_total = self.time_of_flight(y_final)
        t_arr = np.linspace(0, t_total, n_points)

        x_arr = self.x0 + self.v0x * t_arr
        y_arr = self.y0 + self.v0y * t_arr - 0.5 * self.g * t_arr**2

        return x_arr, y_arr

    def get_velocity_vectors(
        self,
        n_points: int = 10,
        y_final: float = 0.0,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Get velocity vectors at evenly spaced times.

        Args:
            n_points: Number of vectors
            y_final: Final y position

        Returns:
            List of (x, y, vx, vy) tuples for plotting arrows
        """
        t_total = self.time_of_flight(y_final)
        t_arr = np.linspace(0, t_total, n_points)

        vectors = []
        for t in t_arr:
            x, y = self.position_at_time(t)
            vx, vy = self.velocity_at_time(t)
            vectors.append((x, y, vx, vy))

        return vectors

    # ========================================================================
    # COMPLETE ANALYSIS
    # ========================================================================

    def analyze(self, y_final: float = 0.0) -> ProjectileResult:
        """
        Perform complete trajectory analysis.

        This calculates all key quantities and returns them in a
        structured ProjectileResult object.

        Args:
            y_final: Final y position (landing height)

        Returns:
            ProjectileResult with all computed quantities
        """
        # Time calculations
        t_apex = self.time_to_max_height()
        t_flight = self.time_of_flight(y_final)

        # Position calculations
        apex_x, apex_y = self.apex_position()
        impact_x = self.x_at_time(t_flight)

        # Velocity at impact
        impact_vx, impact_vy = self.impact_velocity(y_final)
        impact_speed = math.sqrt(impact_vx**2 + impact_vy**2)
        impact_ang = math.atan2(-impact_vy, impact_vx)  # Angle below horizontal

        # Build equations used lists
        eqs_x = [
            "x(t) = x₀ + v₀cos(θ)t",
            "vₓ = v₀cos(θ) = constant",
        ]
        eqs_y = [
            "y(t) = y₀ + v₀sin(θ)t - ½gt²",
            "vᵧ(t) = v₀sin(θ) - gt",
            "vᵧ² = v₀ᵧ² - 2g(y - y₀)",
        ]

        return ProjectileResult(
            # Initial conditions
            v0=self.v0,
            v0x=self.v0x,
            v0y=self.v0y,
            x0=self.x0,
            y0=self.y0,
            theta=self.theta,
            theta_deg=math.degrees(self.theta),
            g=self.g,
            # Key results
            time_of_flight=t_flight,
            range=impact_x - self.x0,
            max_height=apex_y,
            time_to_max_height=t_apex,
            # At apex
            apex_x=apex_x,
            apex_y=apex_y,
            # At impact
            impact_x=impact_x,
            impact_y=y_final,
            impact_velocity=impact_speed,
            impact_vx=impact_vx,
            impact_vy=impact_vy,
            impact_angle=impact_ang,
            impact_angle_deg=math.degrees(impact_ang),
            # Equations
            equations_x=eqs_x,
            equations_y=eqs_y,
        )

    def show_work(self, y_final: float = 0.0) -> str:
        """
        Generate a detailed solution showing all work.

        This produces a step-by-step solution suitable for
        physics homework or teaching purposes.

        Args:
            y_final: Landing height

        Returns:
            Formatted multi-line string with complete solution
        """
        result = self.analyze(y_final)

        lines = [
            "=" * 70,
            "PROJECTILE MOTION - COMPLETE SOLUTION",
            "=" * 70,
            "",
            "GIVEN:",
            f"  Initial speed:       v₀ = {self.v0} m/s",
            f"  Launch angle:        θ = {math.degrees(self.theta)}°",
            f"  Initial height:      y₀ = {self.y0} m",
            f"  Initial x position:  x₀ = {self.x0} m",
            f"  Gravity:             g = {self.g} m/s²",
            "",
            "-" * 70,
            "STEP 1: DECOMPOSE INITIAL VELOCITY",
            "-" * 70,
            "",
            "  The initial velocity has horizontal and vertical components:",
            "",
            f"  v₀ₓ = v₀ cos(θ) = {self.v0} × cos({math.degrees(self.theta)}°)",
            f"      = {self.v0x:.6g} m/s",
            "",
            f"  v₀ᵧ = v₀ sin(θ) = {self.v0} × sin({math.degrees(self.theta)}°)",
            f"      = {self.v0y:.6g} m/s",
            "",
            "-" * 70,
            "STEP 2: HORIZONTAL MOTION (constant velocity, aₓ = 0)",
            "-" * 70,
            "",
            "  Using Kinematic Equation #2 with a = 0:",
            "    x(t) = x₀ + v₀ₓt",
            f"    x(t) = {self.x0} + ({self.v0x:.6g})t",
            "",
            "-" * 70,
            "STEP 3: VERTICAL MOTION (uniformly accelerated, aᵧ = -g)",
            "-" * 70,
            "",
            "  Using Kinematic Equation #2:",
            "    y(t) = y₀ + v₀ᵧt + ½aᵧt²",
            "    y(t) = y₀ + v₀ᵧt - ½gt²",
            f"    y(t) = {self.y0} + ({self.v0y:.6g})t - ½({self.g})t²",
            "",
            "  Vertical velocity (Equation #1):",
            "    vᵧ(t) = v₀ᵧ + aᵧt = v₀ᵧ - gt",
            f"    vᵧ(t) = {self.v0y:.6g} - ({self.g})t",
            "",
            "-" * 70,
            "STEP 4: TIME TO MAXIMUM HEIGHT",
            "-" * 70,
            "",
            "  At maximum height, vertical velocity = 0:",
            "    vᵧ = 0 = v₀ᵧ - gt_apex",
            "    t_apex = v₀ᵧ / g",
            f"    t_apex = {self.v0y:.6g} / {self.g}",
            f"    t_apex = {result.time_to_max_height:.6g} s",
            "",
            "-" * 70,
            "STEP 5: MAXIMUM HEIGHT",
            "-" * 70,
            "",
            "  Using Equation #3 with v = 0:",
            "    0 = v₀ᵧ² - 2g(y_max - y₀)",
            "    y_max = y₀ + v₀ᵧ²/(2g)",
            f"    y_max = {self.y0} + ({self.v0y:.6g})²/(2 × {self.g})",
            f"    y_max = {result.max_height:.6g} m",
            "",
            "-" * 70,
            "STEP 6: TIME OF FLIGHT",
            "-" * 70,
            "",
            f"  Solve for when y = {y_final}:",
            f"    {y_final} = {self.y0} + ({self.v0y:.6g})t - ½({self.g})t²",
            "",
            "  Rearranging to standard quadratic form at² + bt + c = 0:",
            f"    -½({self.g})t² + ({self.v0y:.6g})t + ({self.y0} - {y_final}) = 0",
            "",
            "  Using quadratic formula:",
            f"    t = {result.time_of_flight:.6g} s",
            "",
            "-" * 70,
            "STEP 7: HORIZONTAL RANGE",
            "-" * 70,
            "",
            "  Range = horizontal distance traveled",
            f"    Range = v₀ₓ × t_flight",
            f"    Range = {self.v0x:.6g} × {result.time_of_flight:.6g}",
            f"    Range = {result.range:.6g} m",
            "",
            "-" * 70,
            "STEP 8: IMPACT VELOCITY",
            "-" * 70,
            "",
            "  Horizontal component (unchanged):",
            f"    vₓ = {result.impact_vx:.6g} m/s",
            "",
            "  Vertical component at impact:",
            f"    vᵧ = v₀ᵧ - gt",
            f"    vᵧ = {self.v0y:.6g} - ({self.g})({result.time_of_flight:.6g})",
            f"    vᵧ = {result.impact_vy:.6g} m/s",
            "",
            "  Impact speed (Pythagorean theorem):",
            f"    v = √(vₓ² + vᵧ²)",
            f"    v = √(({result.impact_vx:.6g})² + ({result.impact_vy:.6g})²)",
            f"    v = {result.impact_velocity:.6g} m/s",
            "",
            "  Impact angle below horizontal:",
            f"    θ_impact = arctan(|vᵧ|/vₓ)",
            f"    θ_impact = {result.impact_angle_deg:.2f}°",
            "",
            "=" * 70,
            "FINAL ANSWERS",
            "=" * 70,
            "",
            f"  Time of flight:    {result.time_of_flight:.4g} s",
            f"  Horizontal range:  {result.range:.4g} m",
            f"  Maximum height:    {result.max_height:.4g} m",
            f"  Impact speed:      {result.impact_velocity:.4g} m/s",
            f"  Impact angle:      {result.impact_angle_deg:.2f}° below horizontal",
            "",
            "=" * 70,
        ]

        return "\n".join(lines)


# ============================================================================
# SYMBOLIC PROJECTILE
# ============================================================================


class SymbolicProjectile:
    """
    Symbolic projectile motion analysis.

    Derives general formulas for projectile motion quantities
    using SymPy symbolic mathematics.
    """

    def __init__(self):
        # Define symbols
        self.v0 = sp.Symbol("v_0", positive=True)
        self.theta = sp.Symbol("theta", real=True)
        self.g = sp.Symbol("g", positive=True)
        self.t = sp.Symbol("t", nonnegative=True)
        self.x0 = sp.Symbol("x_0", real=True)
        self.y0 = sp.Symbol("y_0", real=True)

        # Derived quantities
        self.v0x = self.v0 * sp.cos(self.theta)
        self.v0y = self.v0 * sp.sin(self.theta)

    def range_formula(self, y0_val: float = 0, y_final: float = 0) -> sp.Expr:
        """
        Derive the range formula.

        For launching and landing at the same height (y₀ = y_final = 0):
            Range = v₀² sin(2θ) / g

        Returns:
            Symbolic expression for range
        """
        if y0_val == y_final == 0:
            # Simple case: same height launch and landing
            return self.v0**2 * sp.sin(2 * self.theta) / self.g
        else:
            # General case requires solving quadratic
            # This returns the general formula
            t_flight = (self.v0y + sp.sqrt(self.v0y**2 + 2 * self.g * (self.y0 - y_final))) / self.g
            return self.v0x * t_flight

    def max_height_formula(self) -> sp.Expr:
        """
        Derive the maximum height formula.

        y_max = y₀ + v₀² sin²(θ) / (2g)

        Returns:
            Symbolic expression for maximum height
        """
        return self.y0 + (self.v0**2 * sp.sin(self.theta) ** 2) / (2 * self.g)

    def time_of_flight_formula(self) -> sp.Expr:
        """
        Derive time of flight formula (for landing at y = 0).

        For y₀ = 0: T = 2v₀ sin(θ) / g

        Returns:
            Symbolic expression for time of flight
        """
        # Solve y = 0: 0 = y0 + v0y*t - g*t^2/2
        return (self.v0y + sp.sqrt(self.v0y**2 + 2 * self.g * self.y0)) / self.g

    def trajectory_equation(self) -> sp.Expr:
        """
        Derive trajectory equation y(x).

        y = y₀ + tan(θ)(x - x₀) - g(x - x₀)²/(2v₀² cos²θ)

        Returns:
            Symbolic expression y as function of x
        """
        x = sp.Symbol("x", real=True)
        dx = x - self.x0
        return (
            self.y0
            + sp.tan(self.theta) * dx
            - self.g * dx**2 / (2 * self.v0**2 * sp.cos(self.theta) ** 2)
        )

    def print_all_formulas(self) -> str:
        """Print all key formulas."""
        lines = [
            "PROJECTILE MOTION FORMULAS",
            "=" * 50,
            "",
            "Position as function of time:",
            f"  x(t) = {self.x0} + {self.v0x} * t",
            f"  y(t) = {self.y0} + {self.v0y} * t - (1/2) * {self.g} * t²",
            "",
            "Velocity as function of time:",
            f"  vₓ(t) = {self.v0x}",
            f"  vᵧ(t) = {self.v0y} - {self.g} * t",
            "",
            "Maximum height (above y₀):",
            f"  H = {sp.simplify(self.max_height_formula() - self.y0)}",
            "",
            "Range (for y₀ = y_final = 0):",
            f"  R = {self.range_formula()}",
            "",
            "Time of flight (general):",
            f"  T = {self.time_of_flight_formula()}",
        ]
        return "\n".join(lines)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def analyze_projectile(
    v0: float,
    theta_deg: float,
    y0: float = 0.0,
    g: float = DEFAULT_GRAVITY,
    y_final: float = 0.0,
    show_work: bool = False,
) -> Union[ProjectileResult, str]:
    """
    Convenience function for projectile analysis.

    Args:
        v0: Initial speed in m/s
        theta_deg: Launch angle in degrees
        y0: Initial height in m
        g: Gravity in m/s²
        y_final: Landing height
        show_work: If True, return detailed solution string

    Returns:
        ProjectileResult or formatted solution string
    """
    proj = ProjectileMotion(v0=v0, theta_deg=theta_deg, y0=y0, g=g)

    if show_work:
        return proj.show_work(y_final)
    return proj.analyze(y_final)


def max_range_angle() -> float:
    """
    Get the launch angle for maximum range (when y₀ = y_final = 0).

    For flat ground, maximum range occurs at 45°.

    Returns:
        Optimal angle in degrees (45.0)
    """
    return 45.0


def optimal_angle_for_range(
    target_range: float, v0: float, y0: float = 0.0, g: float = DEFAULT_GRAVITY
) -> List[float]:
    """
    Calculate launch angle(s) needed to hit a target at given range.

    For a given range, there are typically two angles that work
    (one high, one low trajectory) unless at maximum range.

    Args:
        target_range: Desired horizontal range
        v0: Initial speed
        y0: Initial height
        g: Gravity

    Returns:
        List of angles in degrees (may be 0, 1, or 2 values)
    """
    if y0 == 0:
        # Simple case: launching from ground
        # Range = v0^2 * sin(2θ) / g
        # sin(2θ) = Range * g / v0^2
        sin_2theta = target_range * g / v0**2

        if abs(sin_2theta) > 1:
            return []  # Impossible range

        angle_2theta = math.asin(sin_2theta)
        theta1 = angle_2theta / 2
        theta2 = (math.pi - angle_2theta) / 2

        return sorted(set([math.degrees(theta1), math.degrees(theta2)]))
    else:
        # General case: requires numerical solution
        # Would need to solve transcendental equation
        # For now, use numerical search
        from scipy.optimize import brentq

        angles = []

        def range_error(theta_rad):
            proj = ProjectileMotion(v0=v0, theta_rad=theta_rad, y0=y0, g=g)
            try:
                return proj.range(0.0) - target_range
            except ValueError:
                return float("inf")

        # Search for low angle solution (0 to 45 deg)
        try:
            theta_low = brentq(range_error, 0.01, math.pi / 4)
            angles.append(math.degrees(theta_low))
        except (ValueError, RuntimeError):
            pass

        # Search for high angle solution (45 to 90 deg)
        try:
            theta_high = brentq(range_error, math.pi / 4, math.pi / 2 - 0.01)
            angles.append(math.degrees(theta_high))
        except (ValueError, RuntimeError):
            pass

        return sorted(angles)
