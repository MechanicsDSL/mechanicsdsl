"""
MechanicsDSL Parser Package

This package provides the parsing infrastructure for converting MechanicsDSL
source code into an Abstract Syntax Tree (AST) for compilation.

Modules:
    tokens: Token definitions and tokenization.
    ast_nodes: AST node dataclass definitions.
    core: MechanicsParser class implementation.

Quick Start:
    >>> from mechanics_dsl.parser import tokenize, MechanicsParser
    >>> tokens = tokenize(r"\\system{pendulum} \\lagrangian{T - V}")
    >>> parser = MechanicsParser(tokens)
    >>> ast = parser.parse()

The parser uses a recursive descent approach with operator precedence parsing
for expressions. It handles:
- System definitions and variables
- Lagrangian and Hamiltonian mechanics
- Constraints (holonomic and non-holonomic)
- Forces and damping
- Initial conditions
- Coordinate transformations
- SPH fluid definitions
"""

from .ast_nodes import (  # Base classes; Basic expressions; Operations; Vectors; Calculus; Functions; Statements; SPH
    AnimateDef,
    ASTNode,
    BinaryOpExpr,
    BoundaryDef,
    ConstraintDef,
    DampingDef,
    DefineDef,
    DerivativeExpr,
    DerivativeVarExpr,
    ExportDef,
    Expression,
    FluidDef,
    ForceDef,
    FractionExpr,
    FunctionCallExpr,
    GreekLetterExpr,
    HamiltonianDef,
    IdentExpr,
    ImportDef,
    InitialCondition,
    IntegralExpr,
    LagrangianDef,
    NonHolonomicConstraintDef,
    NumberExpr,
    ParameterDef,
    RayleighDef,
    RegionDef,
    SolveDef,
    SystemDef,
    TransformDef,
    UnaryOpExpr,
    VarDef,
    VectorExpr,
    VectorOpExpr,
)
from .core import MechanicsParser, ParserError

# Re-export from submodules for backward compatibility
from .tokens import TOKEN_TYPES, Token, token_pattern, tokenize

__all__ = [
    # Tokenization
    "Token",
    "tokenize",
    "TOKEN_TYPES",
    "token_pattern",
    # Parser
    "MechanicsParser",
    "ParserError",
    # Base classes
    "ASTNode",
    "Expression",
    # Expression nodes
    "NumberExpr",
    "IdentExpr",
    "GreekLetterExpr",
    "DerivativeVarExpr",
    "BinaryOpExpr",
    "UnaryOpExpr",
    "VectorExpr",
    "VectorOpExpr",
    "DerivativeExpr",
    "IntegralExpr",
    "FunctionCallExpr",
    "FractionExpr",
    # Statement nodes
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
