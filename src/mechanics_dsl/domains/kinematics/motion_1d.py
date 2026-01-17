"""
1D Motion Analysis for MechanicsDSL

This module provides classes for analyzing one-dimensional motion:
- UniformMotion: Constant velocity motion (a = 0)
- UniformlyAcceleratedMotion: Constant acceleration motion
- FreeFall: Vertical motion under gravity
- VerticalThrow: Object thrown straight up or down

Each class provides methods to calculate position, velocity, and time
for various scenarios, with results derived analytically using the
kinematic equations.

Example:
    >>> from mechanics_dsl.domains.kinematics import FreeFall
    >>>
    >>> # Ball dropped from 100m height
    >>> fall = FreeFall(y0=100, v0=0)
    >>> print(f"Time to ground: {fall.time_to_ground():.2f} s")
    >>> print(f"Impact velocity: {fall.impact_velocity():.2f} m/s")
"""

import math
from dataclasses import dataclass
from typing import List, Optional

# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class Motion1DState:
    """
    Complete state of 1D motion at a specific time.

    Attributes:
        t: Time (seconds)
        x: Position (meters)
        v: Velocity (m/s)
        a: Acceleration (m/s²)
    """

    t: float
    x: float
    v: float
    a: float

    def __str__(self) -> str:
        return f"t={self.t:.4g}s: x={self.x:.4g}m, v={self.v:.4g}m/s, a={self.a:.4g}m/s²"


# ============================================================================
# UNIFORM MOTION (a = 0)
# ============================================================================


class UniformMotion:
    """
    Constant velocity motion (no acceleration).

    The simplest case of kinematics where velocity is constant:
        x(t) = x₀ + vt
        v(t) = v (constant)
        a = 0

    Attributes:
        x0: Initial position (m)
        v: Constant velocity (m/s)

    Example:
        >>> motion = UniformMotion(x0=10, v=5)
        >>> print(motion.position_at_time(4))
        30.0
        >>> print(motion.time_at_position(25))
        3.0
    """

    def __init__(self, x0: float = 0.0, v: float = 0.0):
        """
        Initialize uniform motion.

        Args:
            x0: Initial position in meters
            v: Constant velocity in m/s
        """
        self.x0 = x0
        self.v = v
        self.a = 0.0

    def position_at_time(self, t: float) -> float:
        """
        Calculate position at time t.

        x(t) = x₀ + vt

        Args:
            t: Time in seconds

        Returns:
            Position in meters
        """
        return self.x0 + self.v * t

    def velocity_at_time(self, t: float) -> float:
        """
        Get velocity at time t (constant).

        Args:
            t: Time in seconds (not used)

        Returns:
            Velocity in m/s
        """
        return self.v

    def time_at_position(self, x: float) -> Optional[float]:
        """
        Calculate time when object reaches position x.

        t = (x - x₀) / v

        Args:
            x: Target position

        Returns:
            Time in seconds, or None if unreachable
        """
        if abs(self.v) < 1e-10:
            # Object is stationary
            if abs(x - self.x0) < 1e-10:
                return 0.0  # Already there
            return None  # Can never reach

        t = (x - self.x0) / self.v

        # Only return if time is non-negative
        return t if t >= 0 else None

    def displacement(self, t: float) -> float:
        """
        Calculate displacement in time t.

        Δx = vt

        Args:
            t: Time interval

        Returns:
            Displacement in meters
        """
        return self.v * t

    def state_at_time(self, t: float) -> Motion1DState:
        """
        Get complete state at time t.

        Args:
            t: Time in seconds

        Returns:
            Motion1DState object
        """
        return Motion1DState(
            t=t,
            x=self.position_at_time(t),
            v=self.v,
            a=0.0,
        )

    def __repr__(self) -> str:
        return f"UniformMotion(x₀={self.x0}, v={self.v})"


# ============================================================================
# UNIFORMLY ACCELERATED MOTION
# ============================================================================


class UniformlyAcceleratedMotion:
    """
    Constant acceleration motion.

    The fundamental case for kinematics with non-zero acceleration:
        x(t) = x₀ + v₀t + ½at²
        v(t) = v₀ + at
        a = constant

    This class uses all 5 kinematic equations as appropriate.

    Attributes:
        x0: Initial position (m)
        v0: Initial velocity (m/s)
        a: Constant acceleration (m/s²)

    Example:
        >>> motion = UniformlyAcceleratedMotion(x0=0, v0=10, a=-2)
        >>> print(f"Position at t=3s: {motion.position_at_time(3):.2f} m")
        >>> print(f"Time to stop: {motion.time_to_reach_velocity(0):.2f} s")
    """

    def __init__(self, x0: float = 0.0, v0: float = 0.0, a: float = 0.0):
        """
        Initialize uniformly accelerated motion.

        Args:
            x0: Initial position in meters
            v0: Initial velocity in m/s
            a: Constant acceleration in m/s²
        """
        self.x0 = x0
        self.v0 = v0
        self.a = a

    def position_at_time(self, t: float) -> float:
        """
        Calculate position at time t.

        Uses Equation 2: x = x₀ + v₀t + ½at²

        Args:
            t: Time in seconds

        Returns:
            Position in meters
        """
        return self.x0 + self.v0 * t + 0.5 * self.a * t**2

    def velocity_at_time(self, t: float) -> float:
        """
        Calculate velocity at time t.

        Uses Equation 1: v = v₀ + at

        Args:
            t: Time in seconds

        Returns:
            Velocity in m/s
        """
        return self.v0 + self.a * t

    def velocity_at_position(self, x: float) -> Optional[float]:
        """
        Calculate velocity at position x.

        Uses Equation 3: v² = v₀² + 2a(x - x₀)

        Args:
            x: Position in meters

        Returns:
            Velocity magnitude (always positive), or None if position unreachable
        """
        v_squared = self.v0**2 + 2 * self.a * (x - self.x0)

        if v_squared < 0:
            return None  # Position is unreachable

        return math.sqrt(v_squared)

    def time_to_reach_position(self, x: float) -> List[float]:
        """
        Calculate time(s) to reach position x.

        Solves: x = x₀ + v₀t + ½at²

        Args:
            x: Target position

        Returns:
            List of times (may have 0, 1, or 2 solutions)
        """
        # Special case: no acceleration
        if abs(self.a) < 1e-10:
            if abs(self.v0) < 1e-10:
                return [0.0] if abs(x - self.x0) < 1e-10 else []
            t = (x - self.x0) / self.v0
            return [t] if t >= 0 else []

        # Quadratic: ½at² + v₀t + (x₀ - x) = 0
        a_coef = 0.5 * self.a
        b_coef = self.v0
        c_coef = self.x0 - x

        discriminant = b_coef**2 - 4 * a_coef * c_coef

        if discriminant < 0:
            return []
        elif discriminant == 0:
            t = -b_coef / (2 * a_coef)
            return [t] if t >= 0 else []
        else:
            sqrt_d = math.sqrt(discriminant)
            t1 = (-b_coef + sqrt_d) / (2 * a_coef)
            t2 = (-b_coef - sqrt_d) / (2 * a_coef)
            return sorted([t for t in [t1, t2] if t >= 0])

    def time_to_reach_velocity(self, v: float) -> Optional[float]:
        """
        Calculate time to reach velocity v.

        Uses Equation 1: t = (v - v₀) / a

        Args:
            v: Target velocity

        Returns:
            Time in seconds, or None if impossible
        """
        if abs(self.a) < 1e-10:
            # No acceleration - can only have constant velocity
            return 0.0 if abs(v - self.v0) < 1e-10 else None

        t = (v - self.v0) / self.a
        return t if t >= 0 else None

    def displacement_during_interval(self, t1: float, t2: float) -> float:
        """
        Calculate displacement between two times.

        Args:
            t1: Start time
            t2: End time

        Returns:
            Displacement in meters
        """
        return self.position_at_time(t2) - self.position_at_time(t1)

    def stopping_distance(self) -> Optional[float]:
        """
        Calculate distance to stop (reach v=0).

        Uses Equation 3: 0 = v₀² + 2a(x - x₀)
        → x - x₀ = -v₀² / (2a)

        Only valid when decelerating (a and v₀ have opposite signs).

        Returns:
            Stopping distance in meters, or None if not decelerating
        """
        if self.a == 0 or self.v0 == 0:
            return 0.0 if self.v0 == 0 else None

        # Check if decelerating (will eventually stop)
        if self.a * self.v0 > 0:
            return None  # Accelerating, won't stop

        return -self.v0**2 / (2 * self.a)

    def stopping_time(self) -> Optional[float]:
        """
        Calculate time to stop (reach v=0).

        Uses Equation 1: 0 = v₀ + at → t = -v₀/a

        Returns:
            Stopping time in seconds, or None if not decelerating
        """
        return self.time_to_reach_velocity(0.0)

    def state_at_time(self, t: float) -> Motion1DState:
        """
        Get complete state at time t.

        Args:
            t: Time in seconds

        Returns:
            Motion1DState object
        """
        return Motion1DState(
            t=t,
            x=self.position_at_time(t),
            v=self.velocity_at_time(t),
            a=self.a,
        )

    def is_accelerating(self) -> bool:
        """Check if object is speeding up."""
        return self.a * self.v0 > 0

    def is_decelerating(self) -> bool:
        """Check if object is slowing down."""
        return self.a * self.v0 < 0

    def direction_changes(self) -> bool:
        """Check if object will change direction."""
        return self.is_decelerating()

    def time_of_direction_change(self) -> Optional[float]:
        """
        Calculate when direction changes (v = 0).

        Returns:
            Time in seconds, or None if direction doesn't change
        """
        if not self.direction_changes():
            return None
        return self.stopping_time()

    def __repr__(self) -> str:
        return f"UniformlyAcceleratedMotion(x₀={self.x0}, v₀={self.v0}, a={self.a})"


# ============================================================================
# FREE FALL
# ============================================================================


class FreeFall:
    """
    Free fall motion under gravity.

    A special case of uniformly accelerated motion with a = -g
    (taking positive direction as upward).

    Uses the convention:
        - Positive y is upward
        - Gravity acts downward (a = -g)
        - g is positive (typically 9.81 m/s²)

    Attributes:
        y0: Initial height (m)
        v0: Initial velocity (m/s, positive = upward)
        g: Gravitational acceleration magnitude (m/s², positive)

    Example:
        >>> fall = FreeFall(y0=100, v0=0, g=9.81)
        >>> print(f"Time to ground: {fall.time_to_ground():.2f} s")
        >>> print(f"Impact velocity: {fall.impact_velocity():.2f} m/s")
    """

    DEFAULT_G = 9.81

    def __init__(self, y0: float = 0.0, v0: float = 0.0, g: float = DEFAULT_G):
        """
        Initialize free fall motion.

        Args:
            y0: Initial height in meters
            v0: Initial velocity in m/s (positive = upward)
            g: Gravitational acceleration magnitude (default 9.81 m/s²)
        """
        if g <= 0:
            raise ValueError(f"Gravity must be positive, got {g}")

        self.y0 = y0
        self.v0 = v0
        self.g = g

        # Create underlying motion object
        self._motion = UniformlyAcceleratedMotion(x0=y0, v0=v0, a=-g)

    def height_at_time(self, t: float) -> float:
        """
        Calculate height at time t.

        y(t) = y₀ + v₀t - ½gt²

        Args:
            t: Time in seconds

        Returns:
            Height in meters
        """
        return self._motion.position_at_time(t)

    def velocity_at_time(self, t: float) -> float:
        """
        Calculate velocity at time t.

        v(t) = v₀ - gt

        Args:
            t: Time in seconds

        Returns:
            Velocity in m/s (positive = upward)
        """
        return self._motion.velocity_at_time(t)

    def velocity_at_height(self, y: float) -> Optional[float]:
        """
        Calculate velocity magnitude at height y.

        Uses Equation 3: v² = v₀² - 2g(y - y₀)

        Args:
            y: Height in meters

        Returns:
            Speed in m/s, or None if height unreachable
        """
        return self._motion.velocity_at_position(y)

    def time_to_ground(self, y_ground: float = 0.0) -> float:
        """
        Calculate time to reach ground level.

        Args:
            y_ground: Ground level height (default 0)

        Returns:
            Time in seconds
        """
        times = self._motion.time_to_reach_position(y_ground)
        if not times:
            raise ValueError(f"Object never reaches y = {y_ground}")
        return max(times)  # Take the landing time, not going up time

    def impact_velocity(self, y_ground: float = 0.0) -> float:
        """
        Calculate velocity at impact with ground.

        Uses Equation 3: v² = v₀² + 2g(y₀ - y_ground)

        Args:
            y_ground: Ground level height

        Returns:
            Impact speed in m/s (always positive)
        """
        v_squared = self.v0**2 + 2 * self.g * (self.y0 - y_ground)
        return math.sqrt(v_squared)

    def max_height(self) -> float:
        """
        Calculate maximum height reached.

        At max height, v = 0:
        y_max = y₀ + v₀²/(2g)

        Returns:
            Maximum height in meters
        """
        if self.v0 <= 0:
            return self.y0  # Already at max height if moving down
        return self.y0 + self.v0**2 / (2 * self.g)

    def time_to_max_height(self) -> float:
        """
        Calculate time to reach maximum height.

        At max height, v = 0: t = v₀/g

        Returns:
            Time to apex in seconds
        """
        if self.v0 <= 0:
            return 0.0  # Already at or past apex
        return self.v0 / self.g

    def time_at_height(self, y: float) -> List[float]:
        """
        Calculate time(s) when object is at height y.

        Args:
            y: Target height

        Returns:
            List of times (going up and/or coming down)
        """
        return self._motion.time_to_reach_position(y)

    def state_at_time(self, t: float) -> Motion1DState:
        """
        Get complete state at time t.

        Args:
            t: Time in seconds

        Returns:
            Motion1DState object
        """
        return Motion1DState(
            t=t,
            x=self.height_at_time(t),
            v=self.velocity_at_time(t),
            a=-self.g,
        )

    def __repr__(self) -> str:
        return f"FreeFall(y₀={self.y0}m, v₀={self.v0}m/s, g={self.g}m/s²)"


# ============================================================================
# VERTICAL THROW
# ============================================================================


class VerticalThrow:
    """
    Object thrown vertically (up or down).

    Extends FreeFall with additional analysis for thrown objects.

    Example:
        >>> throw = VerticalThrow(y0=0, v0=20, g=9.81)  # Thrown upward
        >>> print(f"Max height: {throw.max_height():.2f} m")
        >>> print(f"Total time in air: {throw.total_time_in_air():.2f} s")
    """

    def __init__(self, y0: float = 0.0, v0: float = 0.0, g: float = 9.81):
        """
        Initialize vertical throw.

        Args:
            y0: Initial height in meters
            v0: Initial velocity in m/s (positive = upward)
            g: Gravity magnitude
        """
        self._fall = FreeFall(y0=y0, v0=v0, g=g)
        self.y0 = y0
        self.v0 = v0
        self.g = g

    # Delegate to FreeFall
    def height_at_time(self, t: float) -> float:
        return self._fall.height_at_time(t)

    def velocity_at_time(self, t: float) -> float:
        return self._fall.velocity_at_time(t)

    def max_height(self) -> float:
        return self._fall.max_height()

    def time_to_max_height(self) -> float:
        return self._fall.time_to_max_height()

    def time_to_ground(self, y_ground: float = 0.0) -> float:
        return self._fall.time_to_ground(y_ground)

    def impact_velocity(self, y_ground: float = 0.0) -> float:
        return self._fall.impact_velocity(y_ground)

    # Additional methods specific to throws
    def total_time_in_air(self, y_ground: float = 0.0) -> float:
        """
        Calculate total time in air (from throw to landing).

        Args:
            y_ground: Landing height

        Returns:
            Total time in seconds
        """
        return self.time_to_ground(y_ground)

    def height_above_start(self) -> float:
        """
        Calculate maximum height gained above starting point.

        Returns:
            Height gain in meters
        """
        return self.max_height() - self.y0

    def is_thrown_upward(self) -> bool:
        """Check if object was thrown upward."""
        return self.v0 > 0

    def is_thrown_downward(self) -> bool:
        """Check if object was thrown downward."""
        return self.v0 < 0

    def is_dropped(self) -> bool:
        """Check if object was dropped (v0 = 0)."""
        return abs(self.v0) < 1e-10

    def describe(self) -> str:
        """Generate a description of the throw."""
        if self.is_dropped():
            motion_type = "dropped"
        elif self.is_thrown_upward():
            motion_type = f"thrown upward at {self.v0} m/s"
        else:
            motion_type = f"thrown downward at {abs(self.v0)} m/s"

        lines = [
            f"Object {motion_type} from height {self.y0} m",
            f"  Maximum height: {self.max_height():.4g} m",
            f"  Time to max height: {self.time_to_max_height():.4g} s",
            f"  Time to ground: {self.time_to_ground():.4g} s",
            f"  Impact velocity: {self.impact_velocity():.4g} m/s",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"VerticalThrow(y₀={self.y0}m, v₀={self.v0}m/s)"


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def stopping_distance(v0: float, a: float) -> Optional[float]:
    """
    Calculate stopping distance for given initial velocity and deceleration.

    Uses Equation 3: 0 = v₀² + 2aΔx → Δx = -v₀²/(2a)

    Args:
        v0: Initial velocity in m/s
        a: Acceleration in m/s² (should be opposite sign to v0)

    Returns:
        Stopping distance in meters, or None if not decelerating
    """
    motion = UniformlyAcceleratedMotion(x0=0, v0=v0, a=a)
    return motion.stopping_distance()


def stopping_time(v0: float, a: float) -> Optional[float]:
    """
    Calculate time to stop for given initial velocity and deceleration.

    Uses Equation 1: 0 = v₀ + at → t = -v₀/a

    Args:
        v0: Initial velocity in m/s
        a: Acceleration in m/s²

    Returns:
        Stopping time in seconds, or None if not decelerating
    """
    motion = UniformlyAcceleratedMotion(x0=0, v0=v0, a=a)
    return motion.stopping_time()


def free_fall_time(height: float, g: float = 9.81) -> float:
    """
    Calculate time for object to fall from given height.

    For object dropped from rest: t = √(2h/g)

    Args:
        height: Drop height in meters
        g: Gravity in m/s²

    Returns:
        Time in seconds
    """
    if height < 0:
        raise ValueError(f"Height must be non-negative, got {height}")
    return math.sqrt(2 * height / g)


def free_fall_velocity(height: float, g: float = 9.81) -> float:
    """
    Calculate velocity after falling from given height.

    For object dropped from rest: v = √(2gh)

    Args:
        height: Fall height in meters
        g: Gravity in m/s²

    Returns:
        Impact velocity in m/s
    """
    if height < 0:
        raise ValueError(f"Height must be non-negative, got {height}")
    return math.sqrt(2 * g * height)
