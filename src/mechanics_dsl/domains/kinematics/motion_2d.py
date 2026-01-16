"""
2D Motion Analysis for MechanicsDSL

This module provides classes for analyzing two-dimensional motion:
- Vector2D: Simple 2D vector with operations
- Motion2D: General 2D motion with constant acceleration
- Component decomposition and composition utilities

Motion in 2D is decomposed into independent x and y components,
each analyzed using the 1D kinematic equations.

Example:
    >>> from mechanics_dsl.domains.kinematics import Motion2D, Vector2D
    >>>
    >>> # Ball launched with initial velocity at an angle
    >>> r0 = Vector2D(0, 0)
    >>> v0 = Vector2D(10, 15)  # m/s
    >>> a = Vector2D(0, -9.81)  # m/s²
    >>>
    >>> motion = Motion2D(r0, v0, a)
    >>> print(motion.position_at_time(2.0))
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

from .motion_1d import Motion1DState, UniformlyAcceleratedMotion

# ============================================================================
# VECTOR2D CLASS
# ============================================================================


@dataclass
class Vector2D:
    """
    A simple 2D vector with standard operations.

    Attributes:
        x: x-component
        y: y-component

    Example:
        >>> v = Vector2D(3, 4)
        >>> print(v.magnitude)  # 5.0
        >>> print(v.direction_deg)  # 53.13...
    """

    x: float
    y: float

    @classmethod
    def from_polar(cls, r: float, theta: float) -> "Vector2D":
        """
        Create vector from polar coordinates.

        Args:
            r: Magnitude
            theta: Angle in radians (from +x axis)

        Returns:
            Vector2D
        """
        return cls(
            x=r * math.cos(theta),
            y=r * math.sin(theta),
        )

    @classmethod
    def from_polar_degrees(cls, r: float, theta_deg: float) -> "Vector2D":
        """
        Create vector from polar coordinates with angle in degrees.

        Args:
            r: Magnitude
            theta_deg: Angle in degrees

        Returns:
            Vector2D
        """
        return cls.from_polar(r, math.radians(theta_deg))

    @classmethod
    def zero(cls) -> "Vector2D":
        """Create zero vector."""
        return cls(0.0, 0.0)

    @classmethod
    def unit_x(cls) -> "Vector2D":
        """Create unit vector in x direction."""
        return cls(1.0, 0.0)

    @classmethod
    def unit_y(cls) -> "Vector2D":
        """Create unit vector in y direction."""
        return cls(0.0, 1.0)

    @property
    def magnitude(self) -> float:
        """Get magnitude (length) of vector."""
        return math.sqrt(self.x**2 + self.y**2)

    @property
    def magnitude_squared(self) -> float:
        """Get magnitude squared (avoids sqrt)."""
        return self.x**2 + self.y**2

    @property
    def direction(self) -> float:
        """Get direction angle in radians from +x axis."""
        return math.atan2(self.y, self.x)

    @property
    def direction_deg(self) -> float:
        """Get direction angle in degrees."""
        return math.degrees(self.direction)

    def unit(self) -> "Vector2D":
        """Get unit vector in same direction."""
        mag = self.magnitude
        if mag < 1e-10:
            return Vector2D.zero()
        return Vector2D(self.x / mag, self.y / mag)

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple."""
        return (self.x, self.y)

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y])

    # Vector operations
    def __add__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vector2D":
        return Vector2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vector2D":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vector2D":
        return Vector2D(self.x / scalar, self.y / scalar)

    def __neg__(self) -> "Vector2D":
        return Vector2D(-self.x, -self.y)

    def dot(self, other: "Vector2D") -> float:
        """Dot product with another vector."""
        return self.x * other.x + self.y * other.y

    def cross_z(self, other: "Vector2D") -> float:
        """
        Cross product z-component (2D cross product).

        Returns the z-component of the 3D cross product.
        """
        return self.x * other.y - self.y * other.x

    def project_onto(self, other: "Vector2D") -> "Vector2D":
        """Project this vector onto another vector."""
        other_mag_sq = other.magnitude_squared
        if other_mag_sq < 1e-10:
            return Vector2D.zero()
        scalar = self.dot(other) / other_mag_sq
        return other * scalar

    def rotate(self, angle: float) -> "Vector2D":
        """
        Rotate vector by angle (radians).

        Args:
            angle: Rotation angle in radians (positive = counterclockwise)

        Returns:
            Rotated vector
        """
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vector2D(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a,
        )

    def rotate_degrees(self, angle_deg: float) -> "Vector2D":
        """Rotate vector by angle in degrees."""
        return self.rotate(math.radians(angle_deg))

    def __repr__(self) -> str:
        return f"Vector2D({self.x:.4g}, {self.y:.4g})"

    def __str__(self) -> str:
        return f"({self.x:.4g}, {self.y:.4g})"


# ============================================================================
# MOTION 2D STATE
# ============================================================================


@dataclass
class Motion2DState:
    """
    Complete state of 2D motion at a specific time.

    Attributes:
        t: Time (seconds)
        position: Position vector (m)
        velocity: Velocity vector (m/s)
        acceleration: Acceleration vector (m/s²)
    """

    t: float
    position: Vector2D
    velocity: Vector2D
    acceleration: Vector2D

    @property
    def x(self) -> float:
        """x position."""
        return self.position.x

    @property
    def y(self) -> float:
        """y position."""
        return self.position.y

    @property
    def vx(self) -> float:
        """x velocity."""
        return self.velocity.x

    @property
    def vy(self) -> float:
        """y velocity."""
        return self.velocity.y

    @property
    def speed(self) -> float:
        """Speed (magnitude of velocity)."""
        return self.velocity.magnitude

    @property
    def direction(self) -> float:
        """Direction of velocity in radians."""
        return self.velocity.direction

    @property
    def direction_deg(self) -> float:
        """Direction of velocity in degrees."""
        return self.velocity.direction_deg

    def __str__(self) -> str:
        return (
            f"t={self.t:.4g}s: " f"r={self.position}, " f"v={self.velocity} ({self.speed:.4g} m/s)"
        )


# ============================================================================
# MOTION 2D CLASS
# ============================================================================


class Motion2D:
    """
    General 2D motion with constant acceleration.

    Motion is decomposed into independent x and y components:
        x(t) = x₀ + v₀ₓt + ½aₓt²
        y(t) = y₀ + v₀ᵧt + ½aᵧt²

    Each component follows the 1D kinematic equations.

    Attributes:
        r0: Initial position vector
        v0: Initial velocity vector
        a: Acceleration vector (constant)

    Example:
        >>> r0 = Vector2D(0, 10)  # Starting at (0, 10)
        >>> v0 = Vector2D(5, 15)  # Initial velocity
        >>> a = Vector2D(0, -9.81)  # Gravity
        >>>
        >>> motion = Motion2D(r0, v0, a)
        >>> pos = motion.position_at_time(2.0)
        >>> print(f"Position at t=2s: {pos}")
    """

    def __init__(
        self,
        r0: Union[Vector2D, Tuple[float, float]],
        v0: Union[Vector2D, Tuple[float, float]],
        a: Union[Vector2D, Tuple[float, float]],
    ):
        """
        Initialize 2D motion.

        Args:
            r0: Initial position (Vector2D or (x, y) tuple)
            v0: Initial velocity
            a: Constant acceleration
        """
        # Convert tuples to Vector2D
        self.r0 = r0 if isinstance(r0, Vector2D) else Vector2D(*r0)
        self.v0 = v0 if isinstance(v0, Vector2D) else Vector2D(*v0)
        self.a = a if isinstance(a, Vector2D) else Vector2D(*a)

        # Create 1D motion objects for each component
        self._x_motion = UniformlyAcceleratedMotion(x0=self.r0.x, v0=self.v0.x, a=self.a.x)
        self._y_motion = UniformlyAcceleratedMotion(x0=self.r0.y, v0=self.v0.y, a=self.a.y)

    def position_at_time(self, t: float) -> Vector2D:
        """
        Calculate position at time t.

        r(t) = r₀ + v₀t + ½at²

        Args:
            t: Time in seconds

        Returns:
            Position vector
        """
        return Vector2D(
            self._x_motion.position_at_time(t),
            self._y_motion.position_at_time(t),
        )

    def velocity_at_time(self, t: float) -> Vector2D:
        """
        Calculate velocity at time t.

        v(t) = v₀ + at

        Args:
            t: Time in seconds

        Returns:
            Velocity vector
        """
        return Vector2D(
            self._x_motion.velocity_at_time(t),
            self._y_motion.velocity_at_time(t),
        )

    def speed_at_time(self, t: float) -> float:
        """
        Calculate speed at time t.

        Args:
            t: Time in seconds

        Returns:
            Speed in m/s
        """
        return self.velocity_at_time(t).magnitude

    def displacement(self, t: float) -> Vector2D:
        """
        Calculate displacement from initial position.

        Args:
            t: Time in seconds

        Returns:
            Displacement vector
        """
        return self.position_at_time(t) - self.r0

    def distance_from_origin(self, t: float) -> float:
        """
        Calculate distance from origin at time t.

        Args:
            t: Time in seconds

        Returns:
            Distance in meters
        """
        return self.position_at_time(t).magnitude

    def distance_from_start(self, t: float) -> float:
        """
        Calculate distance from starting position at time t.

        Args:
            t: Time in seconds

        Returns:
            Distance in meters
        """
        return self.displacement(t).magnitude

    def state_at_time(self, t: float) -> Motion2DState:
        """
        Get complete state at time t.

        Args:
            t: Time in seconds

        Returns:
            Motion2DState object
        """
        return Motion2DState(
            t=t,
            position=self.position_at_time(t),
            velocity=self.velocity_at_time(t),
            acceleration=self.a,
        )

    def x_at_time(self, t: float) -> float:
        """Get x position at time t."""
        return self._x_motion.position_at_time(t)

    def y_at_time(self, t: float) -> float:
        """Get y position at time t."""
        return self._y_motion.position_at_time(t)

    def vx_at_time(self, t: float) -> float:
        """Get x velocity at time t."""
        return self._x_motion.velocity_at_time(t)

    def vy_at_time(self, t: float) -> float:
        """Get y velocity at time t."""
        return self._y_motion.velocity_at_time(t)

    def time_at_x(self, x: float) -> List[float]:
        """
        Find time(s) when x equals given value.

        Args:
            x: Target x position

        Returns:
            List of times (may be 0, 1, or 2 values)
        """
        return self._x_motion.time_to_reach_position(x)

    def time_at_y(self, y: float) -> List[float]:
        """
        Find time(s) when y equals given value.

        Args:
            y: Target y position

        Returns:
            List of times
        """
        return self._y_motion.time_to_reach_position(y)

    def get_x_solver(self) -> UniformlyAcceleratedMotion:
        """Get the 1D motion solver for x component."""
        return self._x_motion

    def get_y_solver(self) -> UniformlyAcceleratedMotion:
        """Get the 1D motion solver for y component."""
        return self._y_motion

    def get_trajectory_points(
        self, t_max: float, n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get arrays of trajectory points.

        Args:
            t_max: Maximum time
            n_points: Number of points

        Returns:
            Tuple of (x_array, y_array)
        """
        t = np.linspace(0, t_max, n_points)
        x = self.r0.x + self.v0.x * t + 0.5 * self.a.x * t**2
        y = self.r0.y + self.v0.y * t + 0.5 * self.a.y * t**2
        return x, y

    def __repr__(self) -> str:
        return f"Motion2D(r₀={self.r0}, v₀={self.v0}, a={self.a})"


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def decompose_velocity(speed: float, angle: float, degrees: bool = True) -> Tuple[float, float]:
    """
    Decompose velocity into x and y components.

    Args:
        speed: Speed (magnitude of velocity)
        angle: Direction angle from horizontal
        degrees: If True, angle is in degrees; if False, radians

    Returns:
        Tuple of (vx, vy)

    Example:
        >>> vx, vy = decompose_velocity(10, 30)  # 10 m/s at 30°
        >>> print(f"vx = {vx:.2f}, vy = {vy:.2f}")
    """
    if degrees:
        angle = math.radians(angle)
    return (
        speed * math.cos(angle),
        speed * math.sin(angle),
    )


def compose_velocity(vx: float, vy: float, return_degrees: bool = True) -> Tuple[float, float]:
    """
    Compose velocity components into speed and direction.

    Args:
        vx: x-component of velocity
        vy: y-component of velocity
        return_degrees: If True, return angle in degrees

    Returns:
        Tuple of (speed, angle)

    Example:
        >>> speed, angle = compose_velocity(8.66, 5.0)
        >>> print(f"speed = {speed:.2f} m/s at {angle:.1f}°")
    """
    speed = math.sqrt(vx**2 + vy**2)
    angle = math.atan2(vy, vx)
    if return_degrees:
        angle = math.degrees(angle)
    return (speed, angle)


def relative_position(r1: Vector2D, r2: Vector2D) -> Vector2D:
    """
    Calculate position of object 2 relative to object 1.

    r₂₁ = r₂ - r₁

    Args:
        r1: Position of object 1
        r2: Position of object 2

    Returns:
        Relative position vector (from 1 to 2)
    """
    return r2 - r1


def distance_between(r1: Vector2D, r2: Vector2D) -> float:
    """
    Calculate distance between two positions.

    Args:
        r1: First position
        r2: Second position

    Returns:
        Distance in meters
    """
    return (r2 - r1).magnitude
