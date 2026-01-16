"""
Analytical Kinematics Solver for MechanicsDSL

This module provides solvers for kinematic problems that:
1. Accept any 3 of the 6 kinematic variables as knowns
2. Automatically select the appropriate equation(s)
3. Solve for all unknown variables
4. Show the work (equations used to derivation steps)

The solver uses the 5 kinematic equations to find analytical solutions
rather than numerical integration, making it ideal for physics education
where showing work is important.

Example:
    >>> solver = KinematicsSolver()
    >>> solution = solver.solve(v0=0, a=9.81, t=2.0)
    >>> print(f"Final velocity: {solution.state.velocity:.2f} m/s")
    >>> print(f"Displacement: {solution.state.displacement:.2f} m")
    >>> for step in solution.steps:
    ...     print(step)
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import sympy as sp

from .equations import (
    KINEMATIC_SYMBOLS,
    EquationSelector,
    KinematicEquation,
    KinematicEquations,
)

# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class KinematicState:
    """
    Complete state of a kinematic system.

    Represents all 6 kinematic variables for a 1D constant-acceleration
    motion problem.

    Attributes:
        initial_position: Starting position x₀
        final_position: Ending position x
        displacement: Change in position Δx = x - x₀
        initial_velocity: Starting velocity v₀
        final_velocity: Ending velocity v (can be None if not yet solved)
        acceleration: Constant acceleration a
        time: Elapsed time t
    """

    initial_position: float = 0.0
    final_position: Optional[float] = None
    displacement: Optional[float] = None
    initial_velocity: Optional[float] = None
    final_velocity: Optional[float] = None
    acceleration: Optional[float] = None
    time: Optional[float] = None

    @property
    def velocity(self) -> Optional[float]:
        """Alias for final_velocity."""
        return self.final_velocity

    @property
    def is_complete(self) -> bool:
        """Check if all variables are known."""
        return all(
            [
                self.final_position is not None,
                self.displacement is not None,
                self.initial_velocity is not None,
                self.final_velocity is not None,
                self.acceleration is not None,
                self.time is not None,
            ]
        )

    @property
    def known_count(self) -> int:
        """Count of known variables."""
        count = 1  # x0 is always known
        if self.final_position is not None:
            count += 1
        if self.initial_velocity is not None:
            count += 1
        if self.final_velocity is not None:
            count += 1
        if self.acceleration is not None:
            count += 1
        if self.time is not None:
            count += 1
        return count

    def get_knowns(self) -> Dict[str, float]:
        """Get dictionary of known values."""
        result = {"x0": self.initial_position}

        if self.final_position is not None:
            result["x"] = self.final_position
        if self.initial_velocity is not None:
            result["v0"] = self.initial_velocity
        if self.final_velocity is not None:
            result["v"] = self.final_velocity
        if self.acceleration is not None:
            result["a"] = self.acceleration
        if self.time is not None:
            result["t"] = self.time

        return result

    def to_dict(self) -> Dict[str, Optional[float]]:
        """Convert to dictionary with all values."""
        return {
            "x0": self.initial_position,
            "x": self.final_position,
            "dx": self.displacement,
            "v0": self.initial_velocity,
            "v": self.final_velocity,
            "a": self.acceleration,
            "t": self.time,
        }

    def __repr__(self) -> str:
        parts = []
        if self.initial_position != 0:
            parts.append(f"x₀={self.initial_position}")
        if self.final_position is not None:
            parts.append(f"x={self.final_position:.4g}")
        if self.displacement is not None:
            parts.append(f"Δx={self.displacement:.4g}")
        if self.initial_velocity is not None:
            parts.append(f"v₀={self.initial_velocity:.4g}")
        if self.final_velocity is not None:
            parts.append(f"v={self.final_velocity:.4g}")
        if self.acceleration is not None:
            parts.append(f"a={self.acceleration:.4g}")
        if self.time is not None:
            parts.append(f"t={self.time:.4g}")
        return f"KinematicState({', '.join(parts)})"


@dataclass
class SolutionStep:
    """
    A single step in the solution process.

    Used for showing work in physics problems.

    Attributes:
        equation_used: Which kinematic equation was applied
        solving_for: Variable being solved for
        symbolic_form: Symbolic representation of the step
        substitution: Values substituted
        result: Numerical result
        explanation: Human-readable explanation
    """

    equation_used: KinematicEquation
    solving_for: str
    symbolic_form: sp.Expr
    substitution: Dict[str, float]
    result: float
    explanation: str

    def __str__(self) -> str:
        eq_str = self.equation_used.latex
        subs_str = ", ".join(f"{k}={v}" for k, v in self.substitution.items())
        return (
            f"Using Equation {self.equation_used.number} ({self.equation_used.name}):\n"
            f"  {eq_str}\n"
            f"  Substitute: {subs_str}\n"
            f"  Solve for {self.solving_for}: {self.result:.6g}\n"
            f"  ({self.explanation})"
        )


@dataclass
class KinematicSolution:
    """
    Complete solution to a kinematics problem.

    Contains the final state, all steps taken to reach it,
    and metadata about the solution process.

    Attributes:
        state: Complete kinematic state with all values
        steps: List of solution steps (for showing work)
        equations_used: Set of equations used in solution
        given: Dictionary of initially given values
        solved: Dictionary of solved values
        success: Whether solution was found
        error_message: Error description if not successful
    """

    state: KinematicState
    steps: List[SolutionStep] = field(default_factory=list)
    equations_used: List[KinematicEquation] = field(default_factory=list)
    given: Dict[str, float] = field(default_factory=dict)
    solved: Dict[str, float] = field(default_factory=dict)
    success: bool = True
    error_message: str = ""

    def show_work(self) -> str:
        """
        Generate a formatted string showing all solution steps.

        Returns:
            Multi-line string with complete solution work
        """
        lines = [
            "=" * 60,
            "KINEMATICS PROBLEM SOLUTION",
            "=" * 60,
            "",
            "GIVEN:",
        ]

        for var, val in self.given.items():
            lines.append(f"  {var} = {val}")

        lines.extend(["", "SOLUTION STEPS:", ""])

        for i, step in enumerate(self.steps, 1):
            lines.append(f"Step {i}:")
            lines.append(str(step))
            lines.append("")

        lines.extend(
            [
                "FINAL ANSWERS:",
                "-" * 40,
            ]
        )

        state_dict = self.state.to_dict()
        var_names = {
            "x0": "Initial position (x₀)",
            "x": "Final position (x)",
            "dx": "Displacement (Δx)",
            "v0": "Initial velocity (v₀)",
            "v": "Final velocity (v)",
            "a": "Acceleration (a)",
            "t": "Time (t)",
        }

        for var, val in state_dict.items():
            if val is not None:
                name = var_names.get(var, var)
                lines.append(f"  {name}: {val:.6g}")

        lines.append("=" * 60)

        return "\n".join(lines)

    def __str__(self) -> str:
        if not self.success:
            return f"KinematicSolution(failed: {self.error_message})"
        return f"KinematicSolution(success, solved for {list(self.solved.keys())})"


# ============================================================================
# KINEMATIC SOLVER
# ============================================================================


class KinematicsSolver:
    """
    Analytical solver for kinematic problems.

    Given sufficient known variables (typically 3 for unconstrained problems),
    this solver determines all unknown kinematic quantities using the
    5 kinematic equations.

    The solver:
    1. Validates that sufficient information is provided
    2. Selects appropriate equation(s) to solve the problem
    3. Applies equations in the correct order
    4. Records all steps for showing work

    Example:
        >>> solver = KinematicsSolver()
        >>>
        >>> # Free fall from rest for 3 seconds
        >>> solution = solver.solve(v0=0, a=-9.81, t=3.0)
        >>> print(f"Displacement: {solution.state.displacement:.2f} m")
        Displacement: -44.15 m
        >>>
        >>> # Show all work
        >>> print(solution.show_work())

    Minimum Requirements:
        - For problems with x₀ given: need 3 more variables
        - The variables must allow solving (can't have all positions or all velocities)
    """

    def __init__(self, x0: float = 0.0):
        """
        Initialize the solver.

        Args:
            x0: Default initial position (typically 0)
        """
        self.default_x0 = x0
        self.equations = KinematicEquations()
        self.selector = EquationSelector()

    def solve(
        self,
        x0: Optional[float] = None,
        x: Optional[float] = None,
        v0: Optional[float] = None,
        v: Optional[float] = None,
        a: Optional[float] = None,
        t: Optional[float] = None,
        dx: Optional[float] = None,
    ) -> KinematicSolution:
        """
        Solve a kinematics problem given known values.

        Provide any combination of known variables. The solver will
        determine if sufficient information exists and solve for unknowns.

        Args:
            x0: Initial position (default 0 if not specified)
            x: Final position
            v0: Initial velocity
            v: Final velocity
            a: Acceleration
            t: Time elapsed
            dx: Displacement (alternative to specifying x when x0=0)

        Returns:
            KinematicSolution with complete state and solution steps

        Raises:
            ValueError: If insufficient information provided
        """
        # Handle defaults
        if x0 is None:
            x0 = self.default_x0

        # Handle displacement as alternative input
        if dx is not None and x is None:
            x = x0 + dx

        # Build the initial state
        state = KinematicState(
            initial_position=x0,
            final_position=x,
            displacement=x - x0 if x is not None else None,
            initial_velocity=v0,
            final_velocity=v,
            acceleration=a,
            time=t,
        )

        # Record given values
        given = state.get_knowns()

        # Validate inputs
        is_valid, error_msg = self._validate_inputs(state)
        if not is_valid:
            return KinematicSolution(
                state=state,
                given=given,
                success=False,
                error_message=error_msg,
            )

        # Solve iteratively
        steps = []
        equations_used = []
        solved = {}

        # Keep solving until we have all values or can't progress
        max_iterations = 10  # Prevent infinite loops
        for _ in range(max_iterations):
            if state.is_complete:
                break

            # Find an equation we can use
            step = self._solve_one_step(state)
            if step is None:
                break

            steps.append(step)
            equations_used.append(step.equation_used)
            solved[step.solving_for] = step.result

            # Update state with new value
            state = self._update_state(state, step.solving_for, step.result)

        # Ensure displacement is computed
        if state.displacement is None and state.final_position is not None:
            state.displacement = state.final_position - state.initial_position

        return KinematicSolution(
            state=state,
            steps=steps,
            equations_used=equations_used,
            given=given,
            solved=solved,
            success=state.is_complete,
            error_message="" if state.is_complete else "Could not solve completely",
        )

    def _validate_inputs(self, state: KinematicState) -> Tuple[bool, str]:
        """
        Validate that sufficient information is provided.

        For 1D kinematics with 6 variables (x0, x, v0, v, a, t),
        we need at least 3 knowns (plus x0 which defaults to 0).

        Args:
            state: Current kinematic state

        Returns:
            Tuple of (is_valid, error_message)
        """
        knowns = state.get_knowns()
        known_set = set(knowns.keys())

        # Need at least 4 knowns (including x0) to solve
        if len(knowns) < 4:
            return (
                False,
                f"Insufficient information: need at least 3 knowns besides x₀, got {len(knowns) - 1}",
            )

        # Check for time validity
        if "t" in knowns and knowns["t"] < 0:
            return False, "Time must be non-negative"

        # Check for specific impossible combinations
        # (e.g., can't solve if we only have positions)
        if known_set == {"x0", "x"}:
            return False, "Cannot solve with only positions - need velocity, acceleration, or time"

        return True, ""

    def _solve_one_step(self, state: KinematicState) -> Optional[SolutionStep]:
        """
        Attempt to solve for one unknown variable.

        Args:
            state: Current kinematic state

        Returns:
            SolutionStep if successful, None otherwise
        """
        knowns = state.get_knowns()
        known_set = set(knowns.keys())

        # Determine what we need to solve for
        all_vars = {"x0", "x", "v0", "v", "a", "t"}
        unknowns = all_vars - known_set

        if not unknowns:
            return None

        # Try each unknown and find an equation
        for unknown in unknowns:
            eq = self.selector.select_for_unknown(unknown, known_set)
            if eq is not None:
                # Solve using this equation
                try:
                    result = eq.substitute_and_solve(unknown, **knowns)

                    # Handle multiple solutions (quadratic)
                    if isinstance(result, (list, tuple)):
                        # For time, take positive solution
                        if unknown == "t":
                            result = max(r for r in result if r >= 0)
                        else:
                            result = result[0]

                    # Validate result
                    if unknown == "t" and result < 0:
                        continue  # Skip negative time, try another approach

                    if not math.isfinite(result):
                        continue

                    # Build the solution step
                    explanation = self._get_explanation(unknown, eq, result)

                    return SolutionStep(
                        equation_used=eq,
                        solving_for=unknown,
                        symbolic_form=eq.solve_for(unknown),
                        substitution=knowns,
                        result=result,
                        explanation=explanation,
                    )
                except (ValueError, ZeroDivisionError):
                    continue

        return None

    def _update_state(self, state: KinematicState, variable: str, value: float) -> KinematicState:
        """
        Create new state with updated variable.

        Args:
            state: Current state
            variable: Variable to update
            value: New value

        Returns:
            Updated KinematicState
        """
        new_state = KinematicState(
            initial_position=state.initial_position,
            final_position=state.final_position if variable != "x" else value,
            displacement=state.displacement,
            initial_velocity=state.initial_velocity if variable != "v0" else value,
            final_velocity=state.final_velocity if variable != "v" else value,
            acceleration=state.acceleration if variable != "a" else value,
            time=state.time if variable != "t" else value,
        )

        # Update displacement if position changed
        if new_state.final_position is not None:
            new_state.displacement = new_state.final_position - new_state.initial_position

        return new_state

    def _get_explanation(self, variable: str, equation: KinematicEquation, result: float) -> str:
        """Generate human-readable explanation."""
        explanations = {
            "x": f"Final position is {result:.4g}",
            "v": f"Final velocity is {result:.4g}",
            "v0": f"Initial velocity was {result:.4g}",
            "a": f"Acceleration is {result:.4g}",
            "t": f"Time elapsed is {result:.4g}",
        }
        return explanations.get(variable, f"{variable} = {result:.4g}")


# ============================================================================
# SYMBOLIC SOLVER
# ============================================================================


class SymbolicKinematicsSolver:
    """
    Symbolic solver for deriving kinematic formulas.

    Unlike the numerical KinematicsSolver, this class works with
    symbolic expressions to derive general formulas.

    Example:
        >>> solver = SymbolicKinematicsSolver()
        >>>
        >>> # Derive formula for displacement given v0, a, t
        >>> formula = solver.derive_formula('x', knowns=['x0', 'v0', 'a', 't'])
        >>> print(formula)
        x_0 + v_0*t + a*t**2/2
    """

    def __init__(self):
        self.equations = KinematicEquations()
        self.symbols = KINEMATIC_SYMBOLS

    def derive_formula(self, solve_for: str, knowns: List[str]) -> sp.Expr:
        """
        Derive a formula for one variable in terms of others.

        Args:
            solve_for: Variable to express
            knowns: List of known variables

        Returns:
            SymPy expression for solve_for in terms of knowns

        Raises:
            ValueError: If derivation is not possible
        """
        known_set = set(knowns)

        # Find equation containing solve_for and all knowns
        for eq in self.equations.all_equations():
            if solve_for in eq.variables:
                needed = eq.variables - {solve_for}
                if needed <= known_set:
                    return eq.solve_for(solve_for)

        raise ValueError(
            f"Cannot derive formula for {solve_for} from {knowns}. "
            "No single equation contains all required variables."
        )

    def derive_displacement_formulas(self) -> Dict[str, sp.Expr]:
        """
        Derive all displacement formulas.

        Returns:
            Dictionary of displacement formulas for different known sets
        """
        x0, x, v0, v, a, t = (self.symbols[k] for k in ["x0", "x", "v0", "v", "a", "t"])

        return {
            "from_v0_a_t": x0 + v0 * t + sp.Rational(1, 2) * a * t**2,
            "from_v_a_t": x0 + v * t - sp.Rational(1, 2) * a * t**2,
            "from_v0_v_t": x0 + sp.Rational(1, 2) * (v0 + v) * t,
            "from_v0_v_a": x0 + (v**2 - v0**2) / (2 * a),
        }

    def derive_velocity_formulas(self) -> Dict[str, sp.Expr]:
        """
        Derive all final velocity formulas.

        Returns:
            Dictionary of velocity formulas
        """
        x0, x, v0, v, a, t = (self.symbols[k] for k in ["x0", "x", "v0", "v", "a", "t"])

        return {
            "from_v0_a_t": v0 + a * t,
            "from_v0_a_dx": sp.sqrt(v0**2 + 2 * a * (x - x0)),
            "from_dx_t_a": (x - x0) / t + sp.Rational(1, 2) * a * t,
        }

    def derive_time_formulas(self) -> Dict[str, sp.Expr]:
        """
        Derive time formulas.

        Returns:
            Dictionary of time formulas
        """
        x0, x, v0, v, a, t = (self.symbols[k] for k in ["x0", "x", "v0", "v", "a", "t"])

        return {
            "from_v0_v_a": (v - v0) / a,
            "from_v0_v_dx": 2 * (x - x0) / (v0 + v),
            "from_v0_a_dx_quadratic": (-v0 + sp.sqrt(v0**2 + 2 * a * (x - x0))) / a,
        }

    def pretty_print_formula(self, expr: sp.Expr) -> str:
        """
        Pretty print a formula.

        Args:
            expr: SymPy expression

        Returns:
            Nicely formatted string
        """
        return sp.pretty(expr, use_unicode=True)

    def latex_formula(self, expr: sp.Expr) -> str:
        """
        Convert formula to LaTeX.

        Args:
            expr: SymPy expression

        Returns:
            LaTeX string
        """
        return sp.latex(expr)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def solve_kinematics(
    x0: float = 0.0,
    x: Optional[float] = None,
    v0: Optional[float] = None,
    v: Optional[float] = None,
    a: Optional[float] = None,
    t: Optional[float] = None,
    show_work: bool = False,
) -> Union[KinematicSolution, str]:
    """
    Convenience function to solve a kinematics problem.

    Args:
        x0: Initial position (default 0)
        x: Final position
        v0: Initial velocity
        v: Final velocity
        a: Acceleration
        t: Time
        show_work: If True, return formatted solution string

    Returns:
        KinematicSolution, or formatted string if show_work=True

    Example:
        >>> solution = solve_kinematics(v0=0, a=9.81, t=2.0)
        >>> print(f"Fell {-solution.state.displacement:.2f} m")

        >>> # Or show all work:
        >>> print(solve_kinematics(v0=0, a=-9.81, t=2.0, show_work=True))
    """
    solver = KinematicsSolver(x0=x0)
    solution = solver.solve(x0=x0, x=x, v0=v0, v=v, a=a, t=t)

    if show_work:
        return solution.show_work()
    return solution


def verify_kinematics(
    x0: float, x: float, v0: float, v: float, a: float, t: float, tolerance: float = 1e-6
) -> Tuple[bool, Dict[str, float]]:
    """
    Verify that a set of kinematic values are self-consistent.

    Checks all 5 kinematic equations to ensure the values satisfy them.

    Args:
        x0, x, v0, v, a, t: The kinematic values to verify
        tolerance: Maximum allowed residual

    Returns:
        Tuple of (all_valid, residual_dict)
    """
    residuals = {}
    all_valid = True

    for eq in KinematicEquations.all_equations():
        residual = eq.evaluate(x0=x0, x=x, v0=v0, v=v, a=a, t=t)
        residuals[eq.name] = residual
        if abs(residual) > tolerance:
            all_valid = False

    return all_valid, residuals
