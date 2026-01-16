"""
Abstract Syntax Tree (AST) node definitions for MechanicsDSL.

This module defines all AST nodes used to represent parsed DSL code.
Each node type corresponds to a different syntactic construct in the language.

Base Classes:
    ASTNode: Base class for all AST nodes.
    Expression: Base class for all expressions.

Expression Nodes:
    NumberExpr: Numeric literals (e.g., 3.14, 2e-5).
    IdentExpr: Identifiers (e.g., x, theta).
    GreekLetterExpr: Greek letters (e.g., \\alpha, \\omega).
    DerivativeVarExpr: Derivative notation (e.g., \\dot{x}, \\ddot{y}).
    BinaryOpExpr: Binary operations (+, -, *, /, ^).
    UnaryOpExpr: Unary operations (+, -).
    VectorExpr: Vector literals.
    VectorOpExpr: Vector operations (dot, cross, grad).
    DerivativeExpr: Derivative expressions.
    IntegralExpr: Integral expressions.
    FunctionCallExpr: Function calls.
    FractionExpr: Fraction expressions (\\frac).

Statement Nodes:
    SystemDef: System declaration.
    VarDef: Variable definition.
    ParameterDef: Parameter definition.
    DefineDef: Custom function definition.
    LagrangianDef: Lagrangian expression.
    HamiltonianDef: Hamiltonian expression.
    ConstraintDef: Holonomic constraint.
    ForceDef: Non-conservative force.
    InitialCondition: Initial conditions.
    SolveDef: Solution method.
    And more...
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

# ============================================================================
# BASE CLASSES
# ============================================================================


class ASTNode:
    """
    Base class for all AST nodes.

    All AST nodes inherit from this class, providing a common
    interface for tree traversal and manipulation.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Expression(ASTNode):
    """
    Base class for all expressions.

    Expressions are AST nodes that evaluate to a value,
    as opposed to statements which perform actions.
    """

    pass


# ============================================================================
# BASIC EXPRESSIONS
# ============================================================================


@dataclass
class NumberExpr(Expression):
    """
    Numeric literal expression.

    Attributes:
        value: The numeric value (float).

    Example:
        >>> expr = NumberExpr(3.14159)
        >>> print(expr)
        Num(3.14159)
    """

    value: float

    def __repr__(self) -> str:
        return f"Num({self.value})"


@dataclass
class IdentExpr(Expression):
    """
    Identifier expression (variable name).

    Attributes:
        name: The identifier string.

    Example:
        >>> expr = IdentExpr('theta')
        >>> print(expr)
        Id(theta)
    """

    name: str

    def __repr__(self) -> str:
        return f"Id({self.name})"


@dataclass
class GreekLetterExpr(Expression):
    """
    Greek letter expression (e.g., \\alpha, \\omega).

    Attributes:
        letter: The Greek letter name (without backslash).

    Example:
        >>> expr = GreekLetterExpr('omega')
        >>> print(expr)
        Greek(omega)
    """

    letter: str

    def __repr__(self) -> str:
        return f"Greek({self.letter})"


@dataclass
class DerivativeVarExpr(Expression):
    """
    Derivative notation expression (\\dot{x} or \\ddot{x}).

    Represents time derivatives using Newton's dot notation.

    Attributes:
        var: The variable name being differentiated.
        order: Order of derivative (1 for dot, 2 for ddot).

    Example:
        >>> expr = DerivativeVarExpr('x', order=2)
        >>> print(expr)
        DerivativeVar(x, order=2)
    """

    var: str
    order: int = 1

    def __repr__(self) -> str:
        return f"DerivativeVar({self.var}, order={self.order})"


# ============================================================================
# BINARY AND UNARY OPERATIONS
# ============================================================================


@dataclass
class BinaryOpExpr(Expression):
    """
    Binary operation expression.

    Represents operations with two operands: +, -, *, /, ^.

    Attributes:
        left: Left operand expression.
        operator: Operator symbol (+, -, *, /, ^).
        right: Right operand expression.

    Example:
        >>> expr = BinaryOpExpr(IdentExpr('x'), '+', NumberExpr(1))
        >>> print(expr)
        BinOp(Id(x) + Num(1))
    """

    left: Expression
    operator: Literal["+", "-", "*", "/", "^"]
    right: Expression

    def __repr__(self) -> str:
        return f"BinOp({self.left} {self.operator} {self.right})"


@dataclass
class UnaryOpExpr(Expression):
    """
    Unary operation expression.

    Represents unary + and - operators.

    Attributes:
        operator: Operator symbol (+ or -).
        operand: The expression being operated on.

    Example:
        >>> expr = UnaryOpExpr('-', NumberExpr(5))
        >>> print(expr)
        UnaryOp(-Num(5))
    """

    operator: Literal["+", "-"]
    operand: Expression

    def __repr__(self) -> str:
        return f"UnaryOp({self.operator}{self.operand})"


# ============================================================================
# VECTOR EXPRESSIONS
# ============================================================================


@dataclass
class VectorExpr(Expression):
    """
    Vector literal expression.

    Represents a vector with multiple components.

    Attributes:
        components: List of component expressions.

    Example:
        >>> expr = VectorExpr([IdentExpr('x'), IdentExpr('y'), IdentExpr('z')])
        >>> print(expr)
        Vector([Id(x), Id(y), Id(z)])
    """

    components: List[Expression]

    def __repr__(self) -> str:
        return f"Vector({self.components})"


@dataclass
class VectorOpExpr(Expression):
    """
    Vector operation expression.

    Represents vector operations like dot product, cross product,
    gradient, divergence, and curl.

    Attributes:
        operation: Operation name (e.g., 'dot', 'cross', 'grad').
        left: First operand expression.
        right: Optional second operand (for binary operations).

    Example:
        >>> expr = VectorOpExpr('cross', IdentExpr('A'), IdentExpr('B'))
        >>> print(expr)
        VectorOp(cross, Id(A), Id(B))
    """

    operation: str
    left: Expression
    right: Optional[Expression] = None

    def __repr__(self) -> str:
        if self.right:
            return f"VectorOp({self.operation}, {self.left}, {self.right})"
        return f"VectorOp({self.operation}, {self.left})"


# ============================================================================
# CALCULUS EXPRESSIONS
# ============================================================================


@dataclass
class DerivativeExpr(Expression):
    """
    Derivative expression.

    Represents total or partial derivatives of expressions.

    Attributes:
        expr: The expression being differentiated.
        var: The variable of differentiation.
        order: Order of the derivative (default 1).
        partial: True for partial derivative, False for total.

    Example:
        >>> expr = DerivativeExpr(IdentExpr('f'), 'x', order=2, partial=True)
        >>> print(expr)
        PartialDeriv(Id(f), x, order=2)
    """

    expr: Expression
    var: str
    order: int = 1
    partial: bool = False

    def __repr__(self) -> str:
        type_str = "Partial" if self.partial else "Total"
        return f"{type_str}Deriv({self.expr}, {self.var}, order={self.order})"


@dataclass
class IntegralExpr(Expression):
    """
    Integral expression.

    Represents definite and indefinite integrals, including line integrals.

    Attributes:
        expr: The integrand expression.
        var: The variable of integration.
        lower: Optional lower bound expression.
        upper: Optional upper bound expression.
        line_integral: True for line integrals (\\oint).

    Example:
        >>> expr = IntegralExpr(IdentExpr('f'), 'x', NumberExpr(0), NumberExpr(1))
        >>> print(expr)
        Integral(Id(f), x, Num(0), Num(1))
    """

    expr: Expression
    var: str
    lower: Optional[Expression] = None
    upper: Optional[Expression] = None
    line_integral: bool = False

    def __repr__(self) -> str:
        return f"Integral({self.expr}, {self.var}, {self.lower}, {self.upper})"


# ============================================================================
# FUNCTION EXPRESSIONS
# ============================================================================


@dataclass
class FunctionCallExpr(Expression):
    """
    Function call expression.

    Represents calls to built-in or user-defined functions.

    Attributes:
        name: Function name.
        args: List of argument expressions.

    Example:
        >>> expr = FunctionCallExpr('sin', [IdentExpr('theta')])
        >>> print(expr)
        Call(sin, [Id(theta)])
    """

    name: str
    args: List[Expression]

    def __repr__(self) -> str:
        return f"Call({self.name}, {self.args})"


@dataclass
class FractionExpr(Expression):
    """
    Fraction expression (\\frac{num}{denom}).

    Represents fractions in LaTeX-style notation.

    Attributes:
        numerator: Numerator expression.
        denominator: Denominator expression.

    Example:
        >>> expr = FractionExpr(NumberExpr(1), NumberExpr(2))
        >>> print(expr)
        Frac(Num(1)/Num(2))
    """

    numerator: Expression
    denominator: Expression

    def __repr__(self) -> str:
        return f"Frac({self.numerator}/{self.denominator})"


# ============================================================================
# STATEMENT NODES
# ============================================================================


@dataclass
class SystemDef(ASTNode):
    """
    System definition statement (\\system{name}).

    Declares a new physics system by name.

    Attributes:
        name: The system name.

    Example:
        >>> stmt = SystemDef('double_pendulum')
        >>> print(stmt)
        System(double_pendulum)
    """

    name: str

    def __repr__(self) -> str:
        return f"System({self.name})"


@dataclass
class VarDef(ASTNode):
    """
    Variable definition statement (\\defvar{name}{type}{unit}).

    Defines a generalized coordinate or other variable.

    Attributes:
        name: Variable name.
        vartype: Variable type (e.g., 'Angle', 'Length', 'Vector').
        unit: Unit specification (e.g., 'rad', 'm').
        vector: True if this is a vector variable.

    Example:
        >>> stmt = VarDef('theta', 'Angle', 'rad')
        >>> print(stmt)
        VarDef(theta: Angle[rad])
    """

    name: str
    vartype: str
    unit: str
    vector: bool = False

    def __repr__(self) -> str:
        vec_str = " [Vector]" if self.vector else ""
        return f"VarDef({self.name}: {self.vartype}[{self.unit}]{vec_str})"


@dataclass
class ParameterDef(ASTNode):
    """
    Parameter definition statement (\\parameter{name}{value}{unit}).

    Defines a constant parameter with a value.

    Attributes:
        name: Parameter name.
        value: Numeric value.
        unit: Unit specification.

    Example:
        >>> stmt = ParameterDef('g', 9.81, 'm/s^2')
        >>> print(stmt)
        Parameter(g = 9.81 [m/s^2])
    """

    name: str
    value: float
    unit: str

    def __repr__(self) -> str:
        return f"Parameter({self.name} = {self.value} [{self.unit}])"


@dataclass
class DefineDef(ASTNode):
    """
    Function definition statement (\\define{...}).

    Defines a custom function or macro.

    Attributes:
        name: Function name.
        args: List of argument names.
        body: Body expression.

    Example:
        >>> stmt = DefineDef('KE', ['m', 'v'], BinaryOpExpr(...))
        >>> print(stmt)
        Define(KE(m, v) = ...)
    """

    name: str
    args: List[str]
    body: Expression

    def __repr__(self) -> str:
        return f"Define({self.name}({', '.join(self.args)}) = {self.body})"


@dataclass
class LagrangianDef(ASTNode):
    """
    Lagrangian definition statement (\\lagrangian{expr}).

    Defines the system Lagrangian L = T - V.

    Attributes:
        expr: The Lagrangian expression.

    Example:
        >>> stmt = LagrangianDef(BinaryOpExpr(T, '-', V))
        >>> print(stmt)
        Lagrangian(...)
    """

    expr: Expression

    def __repr__(self) -> str:
        return f"Lagrangian({self.expr})"


@dataclass
class HamiltonianDef(ASTNode):
    """
    Hamiltonian definition statement (\\hamiltonian{expr}).

    Defines the system Hamiltonian H = T + V.

    Attributes:
        expr: The Hamiltonian expression.
    """

    expr: Expression

    def __repr__(self) -> str:
        return f"Hamiltonian({self.expr})"


@dataclass
class TransformDef(ASTNode):
    """
    Coordinate transform definition (\\transform{type}{var = expr}).

    Defines a coordinate transformation.

    Attributes:
        coord_type: Type of coordinates (e.g., 'polar', 'spherical').
        var: Variable being defined.
        expr: Transformation expression.
    """

    coord_type: str
    var: str
    expr: Expression

    def __repr__(self) -> str:
        return f"Transform({self.coord_type}: {self.var} = {self.expr})"


@dataclass
class ConstraintDef(ASTNode):
    """
    Holonomic constraint definition (\\constraint{expr}).

    Defines a position-dependent constraint.

    Attributes:
        expr: The constraint expression (should equal zero).
        constraint_type: Type of constraint (default 'holonomic').
    """

    expr: Expression
    constraint_type: str = "holonomic"

    def __repr__(self) -> str:
        return f"Constraint({self.expr}, type={self.constraint_type})"


@dataclass
class NonHolonomicConstraintDef(ASTNode):
    """
    Non-holonomic constraint definition (\\nonholonomic{expr}).

    Defines a velocity-dependent constraint that cannot be integrated.

    Attributes:
        expr: The constraint expression.
    """

    expr: Expression

    def __repr__(self) -> str:
        return f"NonHolonomicConstraint({self.expr})"


@dataclass
class ForceDef(ASTNode):
    """
    Non-conservative force definition (\\force{expr}).

    Defines a generalized force (non-conservative).

    Attributes:
        expr: The force expression.
        force_type: Type of force ('friction', 'damping', 'drag', 'general').
    """

    expr: Expression
    force_type: str = "general"

    def __repr__(self) -> str:
        return f"Force({self.expr}, type={self.force_type})"


@dataclass
class DampingDef(ASTNode):
    """
    Damping force definition (\\damping{expr}).

    Defines a velocity-dependent damping force.

    Attributes:
        expr: The damping expression.
        damping_coefficient: Optional damping coefficient.
    """

    expr: Expression
    damping_coefficient: Optional[float] = None

    def __repr__(self) -> str:
        return f"Damping({self.expr}, coeff={self.damping_coefficient})"


@dataclass
class RayleighDef(ASTNode):
    """
    Rayleigh dissipation function definition (\\rayleigh{expr}).

    Defines the Rayleigh dissipation function F where the generalized
    dissipative forces are Q_i = -∂F/∂q̇_i.

    For velocity-dependent damping: F = ½Σ bᵢⱼ q̇ᵢ q̇ⱼ

    Attributes:
        expr: The dissipation function expression.

    Example:
        \\rayleigh{\\frac{1}{2} * b * \\dot{x}^2}
    """

    expr: Expression

    def __repr__(self) -> str:
        return f"Rayleigh({self.expr})"


@dataclass
class InitialCondition(ASTNode):
    """
    Initial conditions statement (\\initial{var1=val1, ...}).

    Specifies initial values for simulation.

    Attributes:
        conditions: Dictionary mapping variable names to initial values.
    """

    conditions: Dict[str, float]

    def __repr__(self) -> str:
        return f"Initial({self.conditions})"


@dataclass
class SolveDef(ASTNode):
    """
    Solve statement (\\solve{method}).

    Specifies the solution method.

    Attributes:
        method: Solution method name.
        options: Additional solver options.
    """

    method: str
    options: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Solve({self.method}, {self.options})"


@dataclass
class AnimateDef(ASTNode):
    """
    Animate statement (\\animate{target}).

    Specifies animation configuration.

    Attributes:
        target: Animation target.
        options: Animation options.
    """

    target: str
    options: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Animate({self.target}, {self.options})"


@dataclass
class ExportDef(ASTNode):
    """
    Export statement (\\export{filename}).

    Exports the system to a file.

    Attributes:
        filename: Output filename.
        format: Export format (default 'json').
    """

    filename: str
    format: str = "json"

    def __repr__(self) -> str:
        return f"Export({self.filename}, {self.format})"


@dataclass
class ImportDef(ASTNode):
    """
    Import statement (\\import{filename}).

    Imports a system from a file.

    Attributes:
        filename: Input filename.
    """

    filename: str

    def __repr__(self) -> str:
        return f"Import({self.filename})"


# ============================================================================
# SPH/FLUID NODES
# ============================================================================


@dataclass
class RegionDef(ASTNode):
    """
    Region definition for SPH fluids.

    Defines a geometric region with constraints.

    Attributes:
        shape: Region shape ('rectangle', 'circle', 'line').
        constraints: Coordinate constraints as (min, max) tuples.
    """

    shape: str
    constraints: Dict[str, Tuple[float, float]]

    def __repr__(self) -> str:
        return f"Region({self.shape}, {self.constraints})"


@dataclass
class FluidDef(ASTNode):
    """
    Fluid definition for SPH simulation.

    Defines a fluid region with properties.

    Attributes:
        name: Fluid name.
        region: Region definition.
        mass: Particle mass.
        eos: Equation of state (e.g., 'tait').
    """

    name: str
    region: RegionDef
    mass: float
    eos: str

    def __repr__(self) -> str:
        return f"Fluid({self.name}, {self.eos}, mass={self.mass})"


@dataclass
class BoundaryDef(ASTNode):
    """
    Boundary definition for SPH simulation.

    Defines a solid boundary.

    Attributes:
        name: Boundary name.
        region: Region definition.
    """

    name: str
    region: RegionDef

    def __repr__(self) -> str:
        return f"Boundary({self.name})"


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Base classes
    "ASTNode",
    "Expression",
    # Basic expressions
    "NumberExpr",
    "IdentExpr",
    "GreekLetterExpr",
    "DerivativeVarExpr",
    # Operations
    "BinaryOpExpr",
    "UnaryOpExpr",
    # Vectors
    "VectorExpr",
    "VectorOpExpr",
    # Calculus
    "DerivativeExpr",
    "IntegralExpr",
    # Functions
    "FunctionCallExpr",
    "FractionExpr",
    # Statements
    "SystemDef",
    "VarDef",
    "ParameterDef",
    "DefineDef",
    "LagrangianDef",
    "HamiltonianDef",
    "TransformDef",
    "ConstraintDef",
    "NonHolonomicConstraintDef",
    "ForceDef",
    "DampingDef",
    "RayleighDef",
    "InitialCondition",
    "SolveDef",
    "AnimateDef",
    "ExportDef",
    "ImportDef",
    # SPH
    "RegionDef",
    "FluidDef",
    "BoundaryDef",
]
