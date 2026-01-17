"""
The Five Kinematic Equations for MechanicsDSL

This module implements the five fundamental kinematic equations for
constant acceleration motion, providing both symbolic representations
and numerical evaluation capabilities.

The Five Kinematic Equations:
    1. v = v₀ + at                  (velocity-time relationship)
    2. x = x₀ + v₀t + ½at²          (position-time relationship)
    3. v² = v₀² + 2a(x - x₀)        (velocity-position relationship)
    4. x = x₀ + ½(v + v₀)t          (average velocity method)
    5. x = x₀ + vt - ½at²           (final velocity form)

Each equation relates a specific subset of the kinematic variables:
    - x₀: initial position
    - x: final position (or Δx = x - x₀ for displacement)
    - v₀: initial velocity
    - v: final velocity
    - a: constant acceleration
    - t: time elapsed

References:
    - Halliday, Resnick, Walker - Fundamentals of Physics
    - Serway & Jewett - Physics for Scientists and Engineers
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import sympy as sp

# ============================================================================
# SYMBOLIC DEFINITIONS
# ============================================================================


class KinematicVariable(Enum):
    """Enumeration of kinematic variables."""

    INITIAL_POSITION = "x0"
    FINAL_POSITION = "x"
    DISPLACEMENT = "dx"  # Δx = x - x₀
    INITIAL_VELOCITY = "v0"
    FINAL_VELOCITY = "v"
    ACCELERATION = "a"
    TIME = "t"


# Standard symbols for kinematics
# Using descriptive names that match physics notation
_x0 = sp.Symbol("x_0", real=True)  # Initial position
_x = sp.Symbol("x", real=True)  # Final position
_v0 = sp.Symbol("v_0", real=True)  # Initial velocity
_v = sp.Symbol("v", real=True)  # Final velocity
_a = sp.Symbol("a", real=True)  # Acceleration
_t = sp.Symbol("t", real=True, nonnegative=True)  # Time (non-negative)
_dx = sp.Symbol("Delta_x", real=True)  # Displacement

# Export symbol dictionary for external use
KINEMATIC_SYMBOLS: Dict[str, sp.Symbol] = {
    "x0": _x0,
    "x": _x,
    "v0": _v0,
    "v": _v,
    "a": _a,
    "t": _t,
    "dx": _dx,
}


# ============================================================================
# KINEMATIC EQUATION CLASS
# ============================================================================


@dataclass
class KinematicEquation:
    """
    Represents one of the five kinematic equations.

    Each equation is stored in its standard form (equation = 0) as well
    as various solved forms for convenience.

    Attributes:
        name: Human-readable name (e.g., "velocity-time")
        number: Equation number (1-5)
        standard_form: The equation in form LHS = RHS
        lhs: Left-hand side of equation
        rhs: Right-hand side of equation
        variables: Set of variables in this equation
        missing_variable: The variable NOT in this equation
        latex: LaTeX representation
        description: Physical interpretation
    """

    name: str
    number: int
    lhs: sp.Expr
    rhs: sp.Expr
    variables: Set[str]
    missing_variable: str
    latex: str
    description: str

    @property
    def standard_form(self) -> sp.Eq:
        """Return the equation in sympy Eq form."""
        return sp.Eq(self.lhs, self.rhs)

    @property
    def zero_form(self) -> sp.Expr:
        """Return the equation in form LHS - RHS = 0."""
        return self.lhs - self.rhs

    def solve_for(self, variable: str) -> sp.Expr:
        """
        Solve the equation for a specific variable.

        Args:
            variable: Name of variable to solve for ('x', 'v', 'v0', 'a', 't', 'x0')

        Returns:
            SymPy expression for the variable

        Raises:
            ValueError: If variable is not in this equation
        """
        if variable not in self.variables:
            raise ValueError(
                f"Variable '{variable}' is not in equation {self.number} "
                f"(missing: {self.missing_variable})"
            )

        sym = KINEMATIC_SYMBOLS[variable]
        solutions = sp.solve(self.standard_form, sym)

        if not solutions:
            raise ValueError(f"Could not solve equation {self.number} for {variable}")

        # Return first solution (may have multiple for quadratics)
        return solutions[0] if isinstance(solutions, list) else solutions

    def solve_for_all(self) -> Dict[str, sp.Expr]:
        """
        Solve the equation for all variables it contains.

        Returns:
            Dictionary mapping variable names to solved expressions
        """
        result = {}
        for var in self.variables:
            try:
                result[var] = self.solve_for(var)
            except ValueError:
                pass  # Some variables may not be solvable
        return result

    def evaluate(self, **values: float) -> float:
        """
        Evaluate the equation with given values.

        This evaluates LHS - RHS, which should be 0 if the equation is satisfied.

        Args:
            **values: Variable values (e.g., v0=5.0, a=9.81, t=2.0)

        Returns:
            Residual (should be ~0 if equation is satisfied)
        """
        expr = self.zero_form

        for var_name, value in values.items():
            if var_name in KINEMATIC_SYMBOLS:
                expr = expr.subs(KINEMATIC_SYMBOLS[var_name], value)

        return float(expr.evalf())

    def substitute_and_solve(self, solve_for: str, **known_values: float) -> float:
        """
        Substitute known values and solve for unknown.

        Args:
            solve_for: Variable to solve for
            **known_values: Known variable values

        Returns:
            Numerical value of the solved variable
        """
        solved_expr = self.solve_for(solve_for)

        for var_name, value in known_values.items():
            if var_name in KINEMATIC_SYMBOLS:
                solved_expr = solved_expr.subs(KINEMATIC_SYMBOLS[var_name], value)

        return float(solved_expr.evalf())

    def __repr__(self) -> str:
        return f"KinematicEquation({self.number}: {self.name})"

    def __str__(self) -> str:
        return f"Equation {self.number}: {self.lhs} = {self.rhs}"


# ============================================================================
# THE FIVE KINEMATIC EQUATIONS
# ============================================================================


class KinematicEquations:
    """
    Container for the five kinematic equations.

    The Five Kinematic Equations for Constant Acceleration:

    1. v = v₀ + at
       - Relates: v, v₀, a, t
       - Missing: x (position)
       - Use when: position is not needed

    2. x = x₀ + v₀t + ½at²
       - Relates: x, x₀, v₀, a, t
       - Missing: v (final velocity)
       - Use when: final velocity is not needed

    3. v² = v₀² + 2a(x - x₀)
       - Relates: v, v₀, a, x, x₀
       - Missing: t (time)
       - Use when: time is not needed (work-energy theorem analog)

    4. x = x₀ + ½(v + v₀)t
       - Relates: x, x₀, v, v₀, t
       - Missing: a (acceleration)
       - Use when: acceleration is not needed

    5. x = x₀ + vt - ½at²
       - Relates: x, x₀, v, a, t
       - Missing: v₀ (initial velocity)
       - Use when: initial velocity is not needed

    Example:
        >>> equations = KinematicEquations()
        >>> eq1 = equations.equation_1()
        >>> print(eq1.latex)
        'v = v_0 + at'
        >>> v_final = eq1.substitute_and_solve('v', v0=0, a=9.81, t=2)
        >>> print(f"v = {v_final:.2f} m/s")
        v = 19.62 m/s
    """

    @staticmethod
    def equation_1() -> KinematicEquation:
        """
        Equation 1: v = v₀ + at

        The velocity-time equation. Relates final velocity to initial
        velocity, acceleration, and time. Does not involve position.

        Derivation: a = dv/dt → ∫dv = ∫a dt → v - v₀ = at

        Returns:
            KinematicEquation for v = v₀ + at
        """
        return KinematicEquation(
            name="velocity-time",
            number=1,
            lhs=_v,
            rhs=_v0 + _a * _t,
            variables={"v", "v0", "a", "t"},
            missing_variable="x",
            latex=r"v = v_0 + at",
            description="Relates final velocity to initial velocity, acceleration, and time",
        )

    @staticmethod
    def equation_2() -> KinematicEquation:
        """
        Equation 2: x = x₀ + v₀t + ½at²

        The position-time equation (kinematic equation of motion).
        Gives position as a function of time for constant acceleration.

        Derivation: x = x₀ + ∫v dt = x₀ + ∫(v₀ + at) dt = x₀ + v₀t + ½at²

        Returns:
            KinematicEquation for x = x₀ + v₀t + ½at²
        """
        return KinematicEquation(
            name="position-time",
            number=2,
            lhs=_x,
            rhs=_x0 + _v0 * _t + sp.Rational(1, 2) * _a * _t**2,
            variables={"x", "x0", "v0", "a", "t"},
            missing_variable="v",
            latex=r"x = x_0 + v_0 t + \frac{1}{2}at^2",
            description="Gives position as a function of time",
        )

    @staticmethod
    def equation_3() -> KinematicEquation:
        """
        Equation 3: v² = v₀² + 2a(x - x₀)

        The velocity-position equation (time-independent). Related to
        the work-energy theorem: ΔKE = W = FΔx = maΔx

        Derivation: From Eq.1: t = (v-v₀)/a. Substitute into Eq.2 and simplify.

        Returns:
            KinematicEquation for v² = v₀² + 2a(x - x₀)
        """
        return KinematicEquation(
            name="velocity-position",
            number=3,
            lhs=_v**2,
            rhs=_v0**2 + 2 * _a * (_x - _x0),
            variables={"v", "v0", "a", "x", "x0"},
            missing_variable="t",
            latex=r"v^2 = v_0^2 + 2a(x - x_0)",
            description="Time-independent equation relating velocities and displacement",
        )

    @staticmethod
    def equation_4() -> KinematicEquation:
        """
        Equation 4: x = x₀ + ½(v + v₀)t

        The average velocity equation. Uses the fact that for constant
        acceleration, average velocity equals the arithmetic mean of
        initial and final velocities.

        Derivation: v_avg = (v + v₀)/2 and x - x₀ = v_avg × t

        Returns:
            KinematicEquation for x = x₀ + ½(v + v₀)t
        """
        return KinematicEquation(
            name="average-velocity",
            number=4,
            lhs=_x,
            rhs=_x0 + sp.Rational(1, 2) * (_v + _v0) * _t,
            variables={"x", "x0", "v", "v0", "t"},
            missing_variable="a",
            latex=r"x = x_0 + \frac{1}{2}(v + v_0)t",
            description="Position from average velocity (assumes constant acceleration)",
        )

    @staticmethod
    def equation_5() -> KinematicEquation:
        """
        Equation 5: x = x₀ + vt - ½at²

        The final velocity form. Useful when initial velocity is unknown
        but final velocity is known.

        Derivation: Substitute v₀ = v - at into Eq.2

        Returns:
            KinematicEquation for x = x₀ + vt - ½at²
        """
        return KinematicEquation(
            name="final-velocity-form",
            number=5,
            lhs=_x,
            rhs=_x0 + _v * _t - sp.Rational(1, 2) * _a * _t**2,
            variables={"x", "x0", "v", "a", "t"},
            missing_variable="v0",
            latex=r"x = x_0 + vt - \frac{1}{2}at^2",
            description="Position using final velocity instead of initial",
        )

    @classmethod
    def all_equations(cls) -> List[KinematicEquation]:
        """
        Get all five kinematic equations.

        Returns:
            List of all 5 KinematicEquation objects
        """
        return [
            cls.equation_1(),
            cls.equation_2(),
            cls.equation_3(),
            cls.equation_4(),
            cls.equation_5(),
        ]

    @classmethod
    def get_equation(cls, number: int) -> KinematicEquation:
        """
        Get a specific equation by number.

        Args:
            number: Equation number (1-5)

        Returns:
            The requested KinematicEquation

        Raises:
            ValueError: If number is not 1-5
        """
        if number < 1 or number > 5:
            raise ValueError(f"Equation number must be 1-5, got {number}")

        equations = [cls.equation_1, cls.equation_2, cls.equation_3, cls.equation_4, cls.equation_5]
        return equations[number - 1]()

    @classmethod
    def find_equation_for_unknowns(cls, knowns: Set[str]) -> Optional[KinematicEquation]:
        """
        Find an equation that can solve for unknowns given the knowns.

        Given a set of known variables, finds an equation where:
        - All but one variable is in the knowns set
        - That equation can be used to solve for the unknown

        Args:
            knowns: Set of known variable names

        Returns:
            A suitable KinematicEquation, or None if none found
        """

        for eq in cls.all_equations():
            # Check how many of this equation's variables we know
            knowns & eq.variables
            unknown_in_eq = eq.variables - knowns

            # If we know all but one variable in this equation, we can solve
            if len(unknown_in_eq) == 1:
                return eq

        return None

    @classmethod
    def equations_containing(cls, *variables: str) -> List[KinematicEquation]:
        """
        Get all equations containing the specified variables.

        Args:
            *variables: Variable names to search for

        Returns:
            List of equations containing ALL specified variables
        """
        var_set = set(variables)
        return [eq for eq in cls.all_equations() if var_set <= eq.variables]

    @classmethod
    def equations_missing(cls, variable: str) -> List[KinematicEquation]:
        """
        Get all equations NOT containing a specific variable.

        Args:
            variable: Variable name to exclude

        Returns:
            List of equations missing the specified variable
        """
        return [eq for eq in cls.all_equations() if eq.missing_variable == variable]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def get_equation_by_name(name: str) -> Optional[KinematicEquation]:
    """
    Get an equation by its name.

    Args:
        name: Equation name (e.g., "velocity-time", "position-time")

    Returns:
        The matching KinematicEquation, or None if not found
    """
    name_lower = name.lower().strip()
    for eq in KinematicEquations.all_equations():
        if eq.name.lower() == name_lower:
            return eq
    return None


def get_equation_for_unknowns(knowns: Set[str]) -> Optional[KinematicEquation]:
    """
    Find an equation that can determine unknowns from the given knowns.

    Convenience wrapper for KinematicEquations.find_equation_for_unknowns.

    Args:
        knowns: Set of known variable names

    Returns:
        A suitable equation, or None if none found
    """
    return KinematicEquations.find_equation_for_unknowns(knowns)


def list_all_equations() -> List[Dict[str, Any]]:
    """
    Get a summary of all equations for display purposes.

    Returns:
        List of dictionaries with equation information
    """
    return [
        {
            "number": eq.number,
            "name": eq.name,
            "equation": str(eq.standard_form),
            "latex": eq.latex,
            "variables": eq.variables,
            "missing": eq.missing_variable,
            "description": eq.description,
        }
        for eq in KinematicEquations.all_equations()
    ]


# ============================================================================
# EQUATION SELECTION HELPER
# ============================================================================


class EquationSelector:
    """
    Helper class for selecting the optimal equation(s) for a given problem.

    Example:
        >>> selector = EquationSelector()
        >>> given = {'v0': 0, 'a': 9.81, 't': 2.0}
        >>> eq = selector.select_for_unknown('v', set(given.keys()))
        >>> print(eq.name)
        'velocity-time'
    """

    def __init__(self):
        self.equations = KinematicEquations.all_equations()

    def select_for_unknown(self, unknown: str, knowns: Set[str]) -> Optional[KinematicEquation]:
        """
        Select an equation to solve for a specific unknown.

        Args:
            unknown: The variable to solve for
            knowns: Set of known variable names

        Returns:
            Best equation to use, or None if impossible
        """
        candidates = []

        for eq in self.equations:
            # Equation must contain the unknown
            if unknown not in eq.variables:
                continue

            # All other variables in equation must be known
            needed = eq.variables - {unknown}
            if needed <= knowns:
                candidates.append(eq)

        # Prefer simpler equations (fewer variables)
        if candidates:
            return min(candidates, key=lambda e: len(e.variables))

        return None

    def minimum_knowns_for(self, unknown: str) -> List[Set[str]]:
        """
        Find minimum sets of knowns needed to solve for an unknown.

        Args:
            unknown: Variable to solve for

        Returns:
            List of minimal known sets that allow solving for unknown
        """
        result = []

        for eq in self.equations:
            if unknown in eq.variables:
                needed = eq.variables - {unknown}
                result.append(needed)

        return result

    def solve_order(
        self, knowns: Set[str], targets: List[str]
    ) -> List[Tuple[str, KinematicEquation]]:
        """
        Determine the order to solve for multiple unknowns.

        Args:
            knowns: Initially known variables
            targets: Variables to solve for

        Returns:
            List of (variable, equation) pairs in solve order
        """
        current_knowns = set(knowns)
        remaining = list(targets)
        order = []

        while remaining:
            # Find a target we can solve for now
            solved_one = False

            for target in remaining[:]:  # Copy to allow modification
                eq = self.select_for_unknown(target, current_knowns)
                if eq:
                    order.append((target, eq))
                    current_knowns.add(target)
                    remaining.remove(target)
                    solved_one = True
                    break

            if not solved_one:
                break  # Can't solve any more

        return order


# ============================================================================
# EQUATION DISPLAY UTILITIES
# ============================================================================


def format_equation_table() -> str:
    """
    Format all equations as a nice text table.

    Returns:
        Formatted string table of equations
    """
    lines = ["=" * 70, "THE FIVE KINEMATIC EQUATIONS (Constant Acceleration)", "=" * 70, ""]

    for eq in KinematicEquations.all_equations():
        lines.append(f"Equation {eq.number}: {eq.name.upper()}")
        lines.append(f"  Formula: {eq.lhs} = {eq.rhs}")
        lines.append(f"  LaTeX:   {eq.latex}")
        lines.append(f"  Missing: {eq.missing_variable}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("Variables: x₀ (initial position), x (final position),")
    lines.append("           v₀ (initial velocity), v (final velocity),")
    lines.append("           a (acceleration), t (time)")
    lines.append("=" * 70)

    return "\n".join(lines)


def show_equations() -> None:
    """Print all equations to stdout."""
    print(format_equation_table())
