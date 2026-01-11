"""
MechanicsDSL: A Domain-Specific Language for Classical Mechanics

A comprehensive framework for symbolic and numerical analysis of classical 
mechanical systems using LaTeX-inspired notation.
"""

# Core imports from main modules
from .compiler import PhysicsCompiler
from .parser import tokenize, MechanicsParser
from .symbolic import SymbolicEngine
from .solver import NumericalSimulator  # Now imports from solver package

# Utils imports
from .utils import setup_logging, logger, config

# Analysis imports
from .energy import PotentialEnergyCalculator

# Exceptions with actionable error messages
from .exceptions import (
    MechanicsDSLError,
    ParseError,
    TokenizationError,
    SemanticError,
    NoLagrangianError,
    NoCoordinatesError,
    SimulationError,
    IntegrationError,
    InitialConditionError,
    ParameterError,
)

__version__ = "1.5.1"
__author__ = "Noah Parsons"
__license__ = "MIT"

__all__ = [
    # Core
    'PhysicsCompiler', 'MechanicsParser', 'SymbolicEngine', 'NumericalSimulator',
    'tokenize',
    # Utils
    'setup_logging', 'logger', 'config',
    # Analysis
    'PotentialEnergyCalculator',
    # Exceptions
    'MechanicsDSLError', 'ParseError', 'TokenizationError', 'SemanticError',
    'NoLagrangianError', 'NoCoordinatesError', 'SimulationError',
    'IntegrationError', 'InitialConditionError', 'ParameterError',
]

