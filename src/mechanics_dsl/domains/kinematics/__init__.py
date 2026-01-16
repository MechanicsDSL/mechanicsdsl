"""
Kinematics Domain for MechanicsDSL

This module provides comprehensive analytical tools for kinematics problems
using the five kinematic equations for constant acceleration motion.

The five kinematic equations are:
1. v = v₀ + at                     (velocity-time)
2. x = x₀ + v₀t + ½at²             (position-time)
3. v² = v₀² + 2a(x - x₀)           (velocity-position)
4. x = x₀ + ½(v + v₀)t             (average velocity)
5. x = x₀ + vt - ½at²              (final velocity form)

This package includes:
- equations: The 5 kinematic equations as symbolic expressions
- solver: Analytical equation solver (given 3 knowns, find the rest)
- projectile: Complete projectile motion analysis
- motion_1d: 1D uniform and uniformly accelerated motion
- motion_2d: 2D motion with component decomposition
- relative: Relative motion between reference frames

Example:
    >>> from mechanics_dsl.domains.kinematics import ProjectileMotion
    >>>
    >>> # Marble launched from 4m height at 5 m/s, 30° above horizontal
    >>> proj = ProjectileMotion(v0=5.0, theta_deg=30.0, y0=4.0)
    >>> result = proj.analyze()
    >>> print(f"Range: {result.range:.2f} m")
    >>> print(f"Time of flight: {result.time_of_flight:.2f} s")
"""

from .equations import (
    KINEMATIC_SYMBOLS,
    KinematicEquation,
    KinematicEquations,
    get_equation_by_name,
    get_equation_for_unknowns,
    list_all_equations,
)
from .motion_1d import (
    FreeFall,
    Motion1DState,
    UniformlyAcceleratedMotion,
    UniformMotion,
    VerticalThrow,
    stopping_distance,
    stopping_time,
)
from .motion_2d import (
    Motion2D,
    Motion2DState,
    Vector2D,
    compose_velocity,
    decompose_velocity,
)
from .projectile import (
    ProjectileMotion,
    ProjectileParameters,
    ProjectileResult,
    SymbolicProjectile,
    analyze_projectile,
    max_range_angle,
    optimal_angle_for_range,
)
from .relative import (
    ReferenceFrame,
    RelativeMotion,
    RelativeVelocity,
    closing_speed,
    relative_velocity,
    time_to_collision,
)
from .solver import (
    KinematicSolution,
    KinematicsSolver,
    KinematicState,
    SymbolicKinematicsSolver,
    solve_kinematics,
)

__all__ = [
    # Equations
    "KinematicEquation",
    "KinematicEquations",
    "KINEMATIC_SYMBOLS",
    "get_equation_by_name",
    "get_equation_for_unknowns",
    "list_all_equations",
    # Solver
    "KinematicState",
    "KinematicSolution",
    "KinematicsSolver",
    "SymbolicKinematicsSolver",
    "solve_kinematics",
    # Projectile motion
    "ProjectileParameters",
    "ProjectileResult",
    "ProjectileMotion",
    "SymbolicProjectile",
    "analyze_projectile",
    "max_range_angle",
    "optimal_angle_for_range",
    # 1D motion
    "Motion1DState",
    "UniformMotion",
    "UniformlyAcceleratedMotion",
    "FreeFall",
    "VerticalThrow",
    "stopping_distance",
    "stopping_time",
    # 2D motion
    "Vector2D",
    "Motion2DState",
    "Motion2D",
    "decompose_velocity",
    "compose_velocity",
    # Relative motion
    "ReferenceFrame",
    "RelativeMotion",
    "RelativeVelocity",
    "relative_velocity",
    "closing_speed",
    "time_to_collision",
]
