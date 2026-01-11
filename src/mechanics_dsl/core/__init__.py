"""
MechanicsDSL Core Package

Core compiler infrastructure including parser, symbolic engine, and solver.

This package re-exports from the main modules for backward compatibility.
New code should import directly from mechanics_dsl instead.
"""

# Re-export from main modules (not duplicates)
from ..compiler import PhysicsCompiler, SystemSerializer, ParticleGenerator
from ..parser import (
    tokenize, Token, MechanicsParser, ParserError,
    # AST nodes (backward compatibility)
    ASTNode, Expression, NumberExpr, IdentExpr, GreekLetterExpr,
    DerivativeVarExpr, BinaryOpExpr, UnaryOpExpr, VectorExpr, VectorOpExpr,
    DerivativeExpr, IntegralExpr, FunctionCallExpr, FractionExpr,
    SystemDef, VarDef, ParameterDef, DefineDef, LagrangianDef, HamiltonianDef,
    TransformDef, ConstraintDef, NonHolonomicConstraintDef, ForceDef, DampingDef,
    InitialCondition, SolveDef, AnimateDef, ExportDef, ImportDef,
    RegionDef, FluidDef, BoundaryDef
)
from ..symbolic import SymbolicEngine
from ..solver import NumericalSimulator

__all__ = [
    # Main classes
    'PhysicsCompiler', 'SystemSerializer', 'ParticleGenerator',
    'MechanicsParser', 'ParserError', 'Token', 'tokenize',
    'SymbolicEngine', 'NumericalSimulator',
    # AST nodes
    'ASTNode', 'Expression', 'NumberExpr', 'IdentExpr', 'GreekLetterExpr',
    'DerivativeVarExpr', 'BinaryOpExpr', 'UnaryOpExpr', 'VectorExpr', 'VectorOpExpr',
    'DerivativeExpr', 'IntegralExpr', 'FunctionCallExpr', 'FractionExpr',
    'SystemDef', 'VarDef', 'ParameterDef', 'DefineDef', 'LagrangianDef', 'HamiltonianDef',
    'TransformDef', 'ConstraintDef', 'NonHolonomicConstraintDef', 'ForceDef', 'DampingDef',
    'InitialCondition', 'SolveDef', 'AnimateDef', 'ExportDef', 'ImportDef',
    'RegionDef', 'FluidDef', 'BoundaryDef',
]
