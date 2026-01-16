"""
MechanicsDSL Core Package

Core compiler infrastructure including parser, symbolic engine, and solver.

This package re-exports from the main modules for backward compatibility.
New code should import directly from mechanics_dsl instead.
"""

# Re-export from main modules (not duplicates)
from ..compiler import ParticleGenerator, PhysicsCompiler, SystemSerializer
from ..parser import (  # AST nodes (backward compatibility)
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
    MechanicsParser,
    NonHolonomicConstraintDef,
    NumberExpr,
    ParameterDef,
    ParserError,
    RegionDef,
    SolveDef,
    SystemDef,
    Token,
    TransformDef,
    UnaryOpExpr,
    VarDef,
    VectorExpr,
    VectorOpExpr,
    tokenize,
)
from ..solver import NumericalSimulator
from ..symbolic import SymbolicEngine

__all__ = [
    # Main classes
    "PhysicsCompiler",
    "SystemSerializer",
    "ParticleGenerator",
    "MechanicsParser",
    "ParserError",
    "Token",
    "tokenize",
    "SymbolicEngine",
    "NumericalSimulator",
    # AST nodes
    "ASTNode",
    "Expression",
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
    "InitialCondition",
    "SolveDef",
    "AnimateDef",
    "ExportDef",
    "ImportDef",
    "RegionDef",
    "FluidDef",
    "BoundaryDef",
]
