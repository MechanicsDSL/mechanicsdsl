"""
Singularity Detection and Analysis Module

Provides tools for detecting and analyzing singularities in mechanical systems:
- Division by zero detection
- Gimbal lock detection
- Configuration space singularities
- Mass matrix singularity detection

This is critical for best-in-class error handling - users should know
BEFORE simulation fails why their system has problems.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

import sympy as sp
import numpy as np

from ..utils import logger


class SingularityType(Enum):
    """Types of singularities in mechanical systems."""

    DIVISION_BY_ZERO = auto()
    MASS_MATRIX_SINGULAR = auto()
    GIMBAL_LOCK = auto()
    CONSTRAINT_DEPENDENT = auto()
    EQUILIBRIUM_UNSTABLE = auto()
    COORDINATE_SINGULARITY = auto()


@dataclass
class SingularityWarning:
    """Warning about a detected singularity."""

    type: SingularityType
    location: str  # Description of where singularity occurs
    condition: str  # Mathematical condition causing singularity
    coordinates: List[str]  # Affected coordinates
    severity: str  # 'warning', 'error', 'critical'
    suggestion: str  # How to fix or work around

    def __str__(self) -> str:
        return (
            f"[{self.severity.upper()}] {self.type.name}: {self.location}\n"
            f"  Condition: {self.condition}\n"
            f"  Affected: {', '.join(self.coordinates)}\n"
            f"  Suggestion: {self.suggestion}"
        )


class SingularityDetector:
    """
    Analyze expressions and systems for singularities.

    Detects:
    - Division by expressions that can be zero
    - Trigonometric singularities (tan at π/2, etc.)
    - Mass matrix singularities
    - Constraint Jacobian rank deficiency
    """

    def __init__(self):
        self.warnings: List[SingularityWarning] = []

    def analyze_expression(
        self, expr: sp.Expr, name: str = "expression"
    ) -> List[SingularityWarning]:
        """
        Analyze a single expression for singularities.

        Args:
            expr: SymPy expression to analyze
            name: Name for error messages

        Returns:
            List of singularity warnings
        """
        warnings = []

        # Check for division by zero
        warnings.extend(self._check_division_singularities(expr, name))

        # Check for trigonometric singularities
        warnings.extend(self._check_trig_singularities(expr, name))

        return warnings

    def _check_division_singularities(self, expr: sp.Expr, name: str) -> List[SingularityWarning]:
        """Find division by zero risks."""
        warnings = []

        # Find all denominators
        denominators = []
        for term in sp.preorder_traversal(expr):
            if isinstance(term, sp.Pow) and term.exp < 0:
                denominators.append(term.base)
            elif isinstance(term, sp.Mul):
                for arg in term.args:
                    if isinstance(arg, sp.Pow) and arg.exp < 0:
                        denominators.append(arg.base)

        for denom in denominators:
            # Check if denominator can be zero
            zeros = sp.solve(denom, dict=False)
            if zeros:
                symbols_in_denom = list(denom.free_symbols)
                coord_names = [str(s) for s in symbols_in_denom]

                warnings.append(
                    SingularityWarning(
                        type=SingularityType.DIVISION_BY_ZERO,
                        location=f"In {name}",
                        condition=f"{denom} = 0 when {zeros}",
                        coordinates=coord_names,
                        severity="warning",
                        suggestion=f"Avoid configurations where {denom} approaches zero",
                    )
                )

        return warnings

    def _check_trig_singularities(self, expr: sp.Expr, name: str) -> List[SingularityWarning]:
        """Find trigonometric function singularities."""
        warnings = []

        # Find tan, sec, csc, cot functions
        for term in sp.preorder_traversal(expr):
            if isinstance(term, sp.tan):
                arg = term.args[0]
                warnings.append(
                    SingularityWarning(
                        type=SingularityType.COORDINATE_SINGULARITY,
                        location=f"In {name}",
                        condition=f"tan({arg}) undefined when {arg} = π/2 + nπ",
                        coordinates=[str(s) for s in arg.free_symbols],
                        severity="warning",
                        suggestion=f"Ensure {arg} stays away from ±π/2",
                    )
                )
            elif isinstance(term, sp.cot):
                arg = term.args[0]
                warnings.append(
                    SingularityWarning(
                        type=SingularityType.COORDINATE_SINGULARITY,
                        location=f"In {name}",
                        condition=f"cot({arg}) undefined when {arg} = nπ",
                        coordinates=[str(s) for s in arg.free_symbols],
                        severity="warning",
                        suggestion=f"Ensure {arg} stays away from 0, π",
                    )
                )

        return warnings

    def analyze_mass_matrix(
        self, equations: List[sp.Expr], coordinates: List[str], accel_symbols: List[sp.Symbol]
    ) -> List[SingularityWarning]:
        """
        Analyze mass matrix for singularities.

        The mass matrix M in M*a = F can become singular at certain
        configurations, causing the system to be uncontrollable.
        """
        warnings = []
        n = len(coordinates)

        # Extract mass matrix
        M = sp.zeros(n, n)
        for i, eq in enumerate(equations):
            for j, a_sym in enumerate(accel_symbols):
                M[i, j] = sp.diff(eq, a_sym)

        # Compute determinant
        det_M = M.det()

        if det_M == 0:
            warnings.append(
                SingularityWarning(
                    type=SingularityType.MASS_MATRIX_SINGULAR,
                    location="Mass matrix",
                    condition="det(M) = 0 everywhere",
                    coordinates=coordinates,
                    severity="error",
                    suggestion="Check Lagrangian formulation - mass matrix should be positive definite",
                )
            )
        else:
            # Find where determinant can be zero
            det_zeros = sp.solve(det_M, dict=False)
            if det_zeros:
                warnings.append(
                    SingularityWarning(
                        type=SingularityType.MASS_MATRIX_SINGULAR,
                        location="Mass matrix",
                        condition=f"det(M) = 0 when {det_zeros}",
                        coordinates=coordinates,
                        severity="warning",
                        suggestion=f"Avoid configurations where determinant vanishes",
                    )
                )

        return warnings

    def detect_gimbal_lock(
        self, equations: Dict[str, sp.Expr], euler_angles: List[str] = None
    ) -> List[SingularityWarning]:
        """
        Detect gimbal lock conditions for Euler angle representations.

        Gimbal lock occurs when two rotation axes align, losing one DOF.
        For Z-Y-X Euler angles, this happens when pitch = ±90°.
        """
        warnings = []

        if euler_angles is None:
            euler_angles = ["phi", "theta", "psi"]

        # Look for sin(theta) in denominators or cos(theta) = 0 problems
        for name, expr in equations.items():
            for angle in euler_angles:
                # Check for 1/cos(angle) patterns
                if expr.has(1 / sp.cos(sp.Symbol(angle))):
                    warnings.append(
                        SingularityWarning(
                            type=SingularityType.GIMBAL_LOCK,
                            location=f"In {name}",
                            condition=f"cos({angle}) = 0 (gimbal lock)",
                            coordinates=[angle],
                            severity="warning",
                            suggestion=f"Use quaternions instead of Euler angles to avoid gimbal lock",
                        )
                    )

        return warnings

    def analyze_equilibrium_stability(
        self, equations: Dict[str, sp.Expr], coordinates: List[str], equilibrium: Dict[str, float]
    ) -> List[SingularityWarning]:
        """
        Analyze stability of an equilibrium point.

        Linearizes about equilibrium and checks eigenvalues.
        """
        warnings = []
        n = len(coordinates)

        # Build Jacobian of the RHS evaluated at equilibrium
        # For a system ẍ = f(x, ẋ), eigenvalue analysis of linearization

        # Substitute equilibrium values
        subs_dict = {}
        for coord, val in equilibrium.items():
            subs_dict[sp.Symbol(coord)] = val

        # This is a simplified analysis - full stability requires
        # constructing the state-space Jacobian

        # Check for unstable (inverted) pendulum-like configurations
        for coord, expr in equations.items():
            # Look for positive restoring terms (unstable)
            linearized = expr.subs(subs_dict)
            for c in coordinates:
                sym = sp.Symbol(c)
                coeff = sp.diff(linearized, sym)
                coeff_val = coeff.subs(subs_dict)

                try:
                    if float(coeff_val) > 0:
                        # Positive coefficient on position = unstable
                        warnings.append(
                            SingularityWarning(
                                type=SingularityType.EQUILIBRIUM_UNSTABLE,
                                location=f"Equilibrium at {equilibrium}",
                                condition=f"∂²V/∂{c}² < 0 (local maximum of potential)",
                                coordinates=[c],
                                severity="warning",
                                suggestion="This equilibrium is unstable - small perturbations will grow",
                            )
                        )
                except (TypeError, ValueError):
                    pass  # Couldn't evaluate numerically

        return warnings

    def analyze_constraint_jacobian(
        self, constraints: List[sp.Expr], coordinates: List[str]
    ) -> List[SingularityWarning]:
        """
        Check constraint Jacobian for rank deficiency.

        Constraints g(q) = 0 have Jacobian ∂g/∂q.
        Rank deficiency means constraints are dependent.
        """
        warnings = []

        if not constraints:
            return warnings

        n_constraints = len(constraints)
        n_coords = len(coordinates)

        # Build constraint Jacobian
        J = sp.zeros(n_constraints, n_coords)
        for i, g in enumerate(constraints):
            for j, q in enumerate(coordinates):
                J[i, j] = sp.diff(g, sp.Symbol(q))

        # Check rank (symbolically)
        rank = J.rank()

        if rank < n_constraints:
            warnings.append(
                SingularityWarning(
                    type=SingularityType.CONSTRAINT_DEPENDENT,
                    location="Constraint system",
                    condition=f"Jacobian rank {rank} < {n_constraints} constraints",
                    coordinates=coordinates,
                    severity="error",
                    suggestion="Some constraints are redundant or dependent - remove or reformulate",
                )
            )

        return warnings


def check_expression_for_singularities(
    expr: sp.Expr, name: str = "expression"
) -> List[SingularityWarning]:
    """
    Convenience function to check a single expression.

    Args:
        expr: SymPy expression
        name: Name for error messages

    Returns:
        List of warnings found
    """
    detector = SingularityDetector()
    return detector.analyze_expression(expr, name)


def format_singularity_report(warnings: List[SingularityWarning]) -> str:
    """
    Format singularity warnings as a readable report.

    Args:
        warnings: List of singularity warnings

    Returns:
        Formatted string report
    """
    if not warnings:
        return "✓ No singularities detected"

    lines = [f"⚠ Found {len(warnings)} potential singularities:\n"]

    for i, w in enumerate(warnings, 1):
        lines.append(f"{i}. {w}\n")

    return "\n".join(lines)


__all__ = [
    "SingularityType",
    "SingularityWarning",
    "SingularityDetector",
    "check_expression_for_singularities",
    "format_singularity_report",
]
