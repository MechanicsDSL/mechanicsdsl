"""
Relative Motion for MechanicsDSL

This module provides classes for analyzing motion in different reference frames:
- ReferenceFrame: A coordinate system with position and velocity
- RelativeMotion: Tools for transforming between frames
- Relative velocity and position calculations
- Collision time calculations

In classical mechanics, velocities add:
    v_AC = v_AB + v_BC

where v_AC is velocity of A relative to C, v_AB is velocity of A relative to B,
and v_BC is velocity of B relative to C.

Example:
    >>> from mechanics_dsl.domains.kinematics import relative_velocity
    >>> 
    >>> # Car A moving at 30 m/s, Car B moving at 20 m/s (same direction)
    >>> v_rel = relative_velocity(v_a=[30, 0], v_b=[20, 0])
    >>> print(f"Relative velocity: {v_rel} m/s")  # [10, 0]
"""
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass
import numpy as np
import math


# ============================================================================
# REFERENCE FRAME
# ============================================================================

@dataclass
class ReferenceFrame:
    """
    A reference frame with position and velocity relative to an inertial frame.
    
    Attributes:
        name: Human-readable name for the frame
        position: Position of frame origin relative to inertial frame
        velocity: Velocity of frame relative to inertial frame
    
    Example:
        >>> # Train moving along x-axis at 20 m/s
        >>> train = ReferenceFrame(
        ...     name="Train",
        ...     position=np.array([100, 0]),
        ...     velocity=np.array([20, 0])
        ... )
    """
    name: str
    position: np.ndarray  # Position of origin
    velocity: np.ndarray  # Velocity of frame
    
    def __post_init__(self):
        """Convert to numpy arrays if needed."""
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)
    
    @classmethod
    def stationary(cls, name: str, position: np.ndarray = None, dim: int = 2) -> 'ReferenceFrame':
        """Create a stationary reference frame."""
        if position is None:
            position = np.zeros(dim)
        return cls(name=name, position=np.asarray(position), velocity=np.zeros(len(position)))
    
    @classmethod
    def moving(cls, name: str, velocity: np.ndarray, 
               position: np.ndarray = None) -> 'ReferenceFrame':
        """Create a moving reference frame."""
        velocity = np.asarray(velocity)
        if position is None:
            position = np.zeros(len(velocity))
        return cls(name=name, position=position, velocity=velocity)
    
    def position_at_time(self, t: float) -> np.ndarray:
        """Get position of frame origin at time t."""
        return self.position + self.velocity * t
    
    def speed(self) -> float:
        """Get speed of frame."""
        return float(np.linalg.norm(self.velocity))
    
    def direction(self) -> float:
        """Get direction of velocity (2D only, in radians)."""
        if len(self.velocity) != 2:
            raise ValueError("direction() only valid for 2D")
        return float(np.arctan2(self.velocity[1], self.velocity[0]))
    
    def __repr__(self) -> str:
        return f"ReferenceFrame('{self.name}', v={self.velocity})"


# ============================================================================
# RELATIVE VELOCITY
# ============================================================================

@dataclass
class RelativeVelocity:
    """
    Result of a relative velocity calculation.
    
    Attributes:
        v_rel: Relative velocity vector (v_A - v_B)
        v_a: Velocity of object A
        v_b: Velocity of object B (or frame)
        speed_rel: Magnitude of relative velocity
        direction: Direction of relative velocity (radians, 2D only)
    """
    v_rel: np.ndarray
    v_a: np.ndarray
    v_b: np.ndarray
    
    @property
    def speed_rel(self) -> float:
        """Relative speed (magnitude of relative velocity)."""
        return float(np.linalg.norm(self.v_rel))
    
    @property
    def direction(self) -> Optional[float]:
        """Direction of relative velocity in radians (2D only)."""
        if len(self.v_rel) != 2:
            return None
        return float(np.arctan2(self.v_rel[1], self.v_rel[0]))
    
    @property
    def direction_deg(self) -> Optional[float]:
        """Direction in degrees (2D only)."""
        d = self.direction
        return math.degrees(d) if d is not None else None
    
    @property
    def approaching(self) -> bool:
        """Check if objects are approaching each other."""
        # This is a simplified check - true approaching depends on positions
        return True  # Would need positions for full check
    
    def __repr__(self) -> str:
        return f"RelativeVelocity(v_rel={self.v_rel}, speed={self.speed_rel:.4g} m/s)"


# ============================================================================
# RELATIVE MOTION
# ============================================================================

class RelativeMotion:
    """
    Tools for relative motion calculations.
    
    Handles transformations between reference frames and relative
    velocity calculations using the Galilean transformation:
    
        v_AC = v_AB + v_BC
    
    where subscripts denote "of first relative to second".
    
    Example:
        >>> rm = RelativeMotion()
        >>> 
        >>> # Observer in a car moving at [20, 0] sees a ball moving at [5, 10]
        >>> # What is ball's velocity in ground frame?
        >>> v_ball_ground = rm.transform_to_ground(
        ...     v_in_frame=[5, 10],
        ...     frame_velocity=[20, 0]
        ... )
        >>> print(v_ball_ground)  # [25, 10]
    """
    
    def transform_velocity_to_frame(
        self,
        v_inertial: np.ndarray,
        frame_velocity: np.ndarray,
    ) -> np.ndarray:
        """
        Transform velocity from inertial frame to moving frame.
        
        v_in_frame = v_inertial - v_frame
        
        Args:
            v_inertial: Velocity in inertial (ground) frame
            frame_velocity: Velocity of the moving frame
            
        Returns:
            Velocity as observed in the moving frame
        """
        return np.asarray(v_inertial) - np.asarray(frame_velocity)
    
    def transform_to_ground(
        self,
        v_in_frame: np.ndarray,
        frame_velocity: np.ndarray,
    ) -> np.ndarray:
        """
        Transform velocity from moving frame to inertial (ground) frame.
        
        v_ground = v_in_frame + v_frame
        
        Args:
            v_in_frame: Velocity as measured in moving frame
            frame_velocity: Velocity of the frame relative to ground
            
        Returns:
            Velocity in ground frame
        """
        return np.asarray(v_in_frame) + np.asarray(frame_velocity)
    
    def transform_position_to_frame(
        self,
        r_inertial: np.ndarray,
        frame_position: np.ndarray,
    ) -> np.ndarray:
        """
        Transform position from inertial frame to moving frame.
        
        r_in_frame = r_inertial - r_frame
        
        Args:
            r_inertial: Position in inertial frame
            frame_position: Position of frame origin
            
        Returns:
            Position in moving frame coordinates
        """
        return np.asarray(r_inertial) - np.asarray(frame_position)
    
    def relative_velocity(
        self,
        v_a: np.ndarray,
        v_b: np.ndarray,
    ) -> RelativeVelocity:
        """
        Calculate velocity of A relative to B.
        
        v_AB = v_A - v_B
        
        Args:
            v_a: Velocity of object A (in some common frame)
            v_b: Velocity of object B (in same frame)
            
        Returns:
            RelativeVelocity object
        """
        v_a = np.asarray(v_a)
        v_b = np.asarray(v_b)
        return RelativeVelocity(
            v_rel=v_a - v_b,
            v_a=v_a,
            v_b=v_b,
        )
    
    def closing_speed(
        self,
        v_a: np.ndarray,
        v_b: np.ndarray,
        r_a: np.ndarray,
        r_b: np.ndarray,
    ) -> float:
        """
        Calculate closing speed (rate of approach).
        
        Positive = approaching, Negative = separating
        
        closing_speed = -(v_rel · r_rel) / |r_rel|
        
        Args:
            v_a, v_b: Velocities of objects A and B
            r_a, r_b: Positions of objects A and B
            
        Returns:
            Closing speed in m/s
        """
        v_a, v_b = np.asarray(v_a), np.asarray(v_b)
        r_a, r_b = np.asarray(r_a), np.asarray(r_b)
        
        v_rel = v_a - v_b  # Velocity of A relative to B
        r_rel = r_a - r_b  # Position of A relative to B
        
        distance = np.linalg.norm(r_rel)
        if distance < 1e-10:
            return 0.0  # Already at same position
        
        r_unit = r_rel / distance
        
        # Closing speed is negative of velocity component along separation
        return -float(np.dot(v_rel, r_unit))
    
    def time_to_collision(
        self,
        r_a: np.ndarray,
        v_a: np.ndarray,
        r_b: np.ndarray,
        v_b: np.ndarray,
        collision_distance: float = 0.0,
    ) -> Optional[float]:
        """
        Calculate time until two objects collide (assuming constant velocities).
        
        Solves: |r_a + v_a*t - r_b - v_b*t| = collision_distance
        
        Args:
            r_a, v_a: Position and velocity of object A
            r_b, v_b: Position and velocity of object B
            collision_distance: Distance at which collision occurs (default 0)
            
        Returns:
            Time until collision (positive), or None if no collision
        """
        r_a, v_a = np.asarray(r_a), np.asarray(v_a)
        r_b, v_b = np.asarray(r_b), np.asarray(v_b)
        
        dr = r_a - r_b  # Relative position
        dv = v_a - v_b  # Relative velocity
        
        # Solve: |dr + dv*t|² = collision_distance²
        # dr² + 2(dr·dv)t + dv²*t² = d²
        
        a = np.dot(dv, dv)
        b = 2 * np.dot(dr, dv)
        c = np.dot(dr, dr) - collision_distance**2
        
        if abs(a) < 1e-10:
            # Constant relative position
            if c <= 0:
                return 0.0  # Already colliding
            return None  # Never collide
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return None  # No real solution, no collision
        
        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b + sqrt_disc) / (2 * a)
        t2 = (-b - sqrt_disc) / (2 * a)
        
        # Take the smallest positive time
        positive_times = [t for t in [t1, t2] if t > 1e-10]
        if not positive_times:
            return None
        
        return min(positive_times)
    
    def closest_approach(
        self,
        r_a: np.ndarray,
        v_a: np.ndarray,
        r_b: np.ndarray,
        v_b: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Calculate time and distance of closest approach.
        
        Args:
            r_a, v_a: Position and velocity of object A
            r_b, v_b: Position and velocity of object B
            
        Returns:
            Tuple of (time, minimum_distance)
        """
        r_a, v_a = np.asarray(r_a), np.asarray(v_a)
        r_b, v_b = np.asarray(r_b), np.asarray(v_b)
        
        dr = r_a - r_b
        dv = v_a - v_b
        
        # d(|dr + dv*t|²)/dt = 0 at closest approach
        # 2(dr + dv*t)·dv = 0
        # dr·dv + dv²*t = 0
        # t = -dr·dv / dv²
        
        dv_sq = np.dot(dv, dv)
        
        if dv_sq < 1e-10:
            # No relative motion, distance stays constant
            distance = float(np.linalg.norm(dr))
            return (0.0, distance)
        
        t_closest = -np.dot(dr, dv) / dv_sq
        
        # If t_closest is negative, closest approach was in the past
        if t_closest < 0:
            t_closest = 0.0
        
        # Calculate distance at that time
        r_rel_at_t = dr + dv * t_closest
        min_distance = float(np.linalg.norm(r_rel_at_t))
        
        return (t_closest, min_distance)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def relative_velocity(
    v_a: Union[np.ndarray, List[float], Tuple[float, ...]],
    v_b: Union[np.ndarray, List[float], Tuple[float, ...]],
) -> np.ndarray:
    """
    Calculate velocity of A relative to B.
    
    v_AB = v_A - v_B
    
    Args:
        v_a: Velocity of object A
        v_b: Velocity of object B
        
    Returns:
        Relative velocity as numpy array
    
    Example:
        >>> v_rel = relative_velocity([30, 0], [20, 0])
        >>> print(v_rel)  # [10, 0]
    """
    return np.asarray(v_a) - np.asarray(v_b)


def closing_speed(
    v_a: Union[np.ndarray, List[float]],
    v_b: Union[np.ndarray, List[float]],
    r_a: Union[np.ndarray, List[float]],
    r_b: Union[np.ndarray, List[float]],
) -> float:
    """
    Calculate closing speed between two objects.
    
    Positive = approaching, Negative = separating
    
    Args:
        v_a, v_b: Velocities
        r_a, r_b: Positions
        
    Returns:
        Closing speed in m/s
    """
    rm = RelativeMotion()
    return rm.closing_speed(v_a, v_b, r_a, r_b)


def time_to_collision(
    r_a: Union[np.ndarray, List[float]],
    v_a: Union[np.ndarray, List[float]],
    r_b: Union[np.ndarray, List[float]],
    v_b: Union[np.ndarray, List[float]],
    collision_distance: float = 0.0,
) -> Optional[float]:
    """
    Calculate time until two objects collide.
    
    Args:
        r_a, v_a: Position and velocity of A
        r_b, v_b: Position and velocity of B
        collision_distance: Distance for collision
        
    Returns:
        Time until collision, or None if no collision
    """
    rm = RelativeMotion()
    return rm.time_to_collision(r_a, v_a, r_b, v_b, collision_distance)


def pursuit_time(
    r_pursuer: np.ndarray,
    v_pursuer: np.ndarray,
    r_target: np.ndarray,
    v_target: np.ndarray,
) -> Optional[float]:
    """
    Calculate time for pursuer to catch target (1D).
    
    Assumes pursuer is faster and both move in straight lines.
    
    Args:
        r_pursuer, v_pursuer: Position and velocity of pursuer
        r_target, v_target: Position and velocity of target
        
    Returns:
        Time to catch, or None if pursuit fails
    """
    return time_to_collision(r_pursuer, v_pursuer, r_target, v_target)
