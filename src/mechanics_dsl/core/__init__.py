"""
MechanicsDSL Core Package

Core compiler infrastructure including parser, symbolic engine, and solver.
"""

from .compiler import PhysicsCompiler, SystemSerializer, ParticleGenerator
from .parser import (
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
from .symbolic import SymbolicEngine
from .solver import NumericalSimulator

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
