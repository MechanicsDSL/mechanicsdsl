"""
Custom exceptions for MechanicsDSL with actionable error messages.

This module provides exception classes that include suggestions for fixing
common errors, making debugging easier for users.
"""

from typing import List, Optional


class MechanicsDSLError(Exception):
    """Base exception for all MechanicsDSL errors."""

    def __init__(
        self, message: str, suggestion: Optional[str] = None, docs_url: Optional[str] = None
    ):
        """
        Initialize exception with optional suggestion and documentation link.

        Args:
            message: The error message
            suggestion: Optional suggestion for fixing the error
            docs_url: Optional URL to relevant documentation
        """
        self.suggestion = suggestion
        self.docs_url = docs_url

        full_message = message
        if suggestion:
            full_message += f"\n\n💡 Suggestion: {suggestion}"
        if docs_url:
            full_message += f"\n📖 Documentation: {docs_url}"

        super().__init__(full_message)


class ParseError(MechanicsDSLError):
    """Raised when the DSL source code cannot be parsed."""

    def __init__(
        self,
        message: str,
        line: Optional[int] = None,
        column: Optional[int] = None,
        source_snippet: Optional[str] = None,
    ):
        suggestion = "Check your DSL syntax. Common issues include:\n"
        suggestion += "  - Missing curly braces: \\lagrangian{expression}\n"
        suggestion += "  - Unmatched parentheses\n"
        suggestion += "  - Invalid characters in variable names"

        if line is not None:
            message = f"Line {line}: {message}"
        if source_snippet:
            message += f"\n\nNear: {source_snippet}"

        super().__init__(message, suggestion=suggestion)
        self.line = line
        self.column = column
        self.source_snippet = source_snippet


class TokenizationError(MechanicsDSLError):
    """Raised when source code cannot be tokenized."""

    def __init__(self, message: str, position: Optional[int] = None):
        suggestion = (
            "Check for:\n"
            "  - Invalid characters or escape sequences\n"
            "  - Malformed numbers (e.g., 1.2.3)\n"
            "  - Unclosed string literals"
        )
        super().__init__(message, suggestion=suggestion)
        self.position = position


class SemanticError(MechanicsDSLError):
    """Raised when DSL has semantic errors (valid syntax but invalid meaning)."""

    pass


class NoLagrangianError(SemanticError):
    """Raised when Lagrangian is required but not defined."""

    def __init__(self, system_name: str = "system"):
        message = f"No Lagrangian defined for {system_name}"
        suggestion = (
            "Add a Lagrangian definition to your DSL source:\n\n"
            "  \\lagrangian{T - V}\n\n"
            "Where T is kinetic energy and V is potential energy.\n\n"
            "Example for a simple pendulum:\n"
            "  \\lagrangian{\\frac{1}{2}*m*l^2*\\dot{theta}^2 + m*g*l*cos(theta)}"
        )
        super().__init__(
            message,
            suggestion=suggestion,
            docs_url="https://mechanicsdsl.readthedocs.io/en/latest/physics/lagrangian_mechanics.html",
        )


class NoCoordinatesError(SemanticError):
    """Raised when no generalized coordinates are found."""

    def __init__(self):
        message = "No generalized coordinates found in the system"
        suggestion = (
            "Define coordinate variables using \\defvar:\n\n"
            "  \\defvar{theta}{Angle}\n"
            "  \\defvar{x}{Position}\n\n"
            "Variable types that become coordinates: Angle, Position, Coordinate"
        )
        super().__init__(message, suggestion=suggestion)


class ConstraintError(SemanticError):
    """Raised when there are issues with constraint definitions."""

    def __init__(self, message: str, constraint_expr: Optional[str] = None):
        suggestion = (
            "Constraints should be expressions that equal zero:\n\n"
            "  \\constraint{x^2 + y^2 - l^2}  % Constrains to circle of radius l\n\n"
            "Make sure all variables in constraints are defined."
        )
        if constraint_expr:
            message += f"\nConstraint: {constraint_expr}"
        super().__init__(message, suggestion=suggestion)


class SimulationError(MechanicsDSLError):
    """Base class for simulation-related errors."""

    pass


class IntegrationError(SimulationError):
    """Raised when numerical integration fails."""

    def __init__(self, message: str, t_failed: Optional[float] = None, is_stiff: bool = False):
        suggestions = []

        if is_stiff:
            suggestions.append(
                "The system appears to be stiff. Try:\n"
                "  - method='LSODA' or method='Radau' (implicit solvers)\n"
                "  - Smaller time step"
            )
        else:
            suggestions.append(
                "Try:\n"
                "  - Reducing time step (increase num_points)\n"
                "  - Using a more stable method: method='LSODA'\n"
                "  - Checking for singularities in your equations"
            )

        if t_failed is not None:
            message += f" (failed at t={t_failed:.4f})"

        super().__init__(message, suggestion="\n".join(suggestions))
        self.t_failed = t_failed
        self.is_stiff = is_stiff


class InitialConditionError(SimulationError):
    """Raised when initial conditions are invalid or inconsistent."""

    def __init__(self, message: str, missing_vars: Optional[List[str]] = None):
        suggestion = "Set initial conditions for all coordinates and velocities:\n"

        if missing_vars:
            suggestion += f"\nMissing: {', '.join(missing_vars)}\n"

        suggestion += (
            "\nExample:\n"
            "  \\initial{theta}{0.5}     % Initial angle\n"
            "  \\initial{theta_dot}{0}   % Initial angular velocity"
        )
        super().__init__(message, suggestion=suggestion)
        self.missing_vars = missing_vars


class ParameterError(MechanicsDSLError):
    """Raised when there are issues with physical parameters."""

    def __init__(self, message: str, param_name: Optional[str] = None):
        suggestion = (
            "Define parameters using \\parameter:\n\n"
            "  \\parameter{m}{1.0}    % Mass in kg\n"
            "  \\parameter{g}{9.81}   % Gravitational acceleration\n"
            "  \\parameter{l}{1.0}    % Length in meters"
        )
        if param_name:
            message = f"Parameter '{param_name}': {message}"
        super().__init__(message, suggestion=suggestion)


class CodeGenerationError(MechanicsDSLError):
    """Raised when code generation fails."""

    def __init__(self, message: str, target_language: Optional[str] = None):
        suggestion = "Check that:\n"
        suggestion += "  - All equations are properly compiled\n"
        suggestion += "  - Required code generation dependencies are installed"

        if target_language:
            suggestion += f"\n  - {target_language} codegen backend is available"
            message = f"[{target_language}] {message}"

        super().__init__(message, suggestion=suggestion)


class VisualizationError(MechanicsDSLError):
    """Raised when visualization fails."""

    def __init__(self, message: str, missing_deps: Optional[List[str]] = None):
        suggestion = "For visualization, ensure matplotlib is properly configured:\n"
        suggestion += "  - In Jupyter: use %matplotlib inline\n"
        suggestion += "  - In scripts: use plt.show() after creating animations"

        if missing_deps:
            suggestion += f"\n\nMissing dependencies: {', '.join(missing_deps)}\n"
            suggestion += "Install with: pip install " + " ".join(missing_deps)

        super().__init__(message, suggestion=suggestion)


class FileValidationError(MechanicsDSLError):
    """Raised when file path validation fails."""

    def __init__(self, message: str, path: Optional[str] = None):
        suggestion = (
            "Check that:\n"
            "  - The file path exists and is accessible\n"
            "  - You have read/write permissions\n"
            "  - The path doesn't contain special characters"
        )
        if path:
            message = f"File '{path}': {message}"
        super().__init__(message, suggestion=suggestion)


__all__ = [
    "MechanicsDSLError",
    "ParseError",
    "TokenizationError",
    "SemanticError",
    "NoLagrangianError",
    "NoCoordinatesError",
    "ConstraintError",
    "SimulationError",
    "IntegrationError",
    "InitialConditionError",
    "ParameterError",
    "CodeGenerationError",
    "VisualizationError",
    "FileValidationError",
]
