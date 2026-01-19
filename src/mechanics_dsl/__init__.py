"""
MechanicsDSL: A Domain-Specific Language for Classical Mechanics

A comprehensive framework for symbolic and numerical analysis of classical
mechanical systems using LaTeX-inspired notation.
"""

# Core imports from main modules
from .compiler import PhysicsCompiler

# Analysis imports
from .energy import PotentialEnergyCalculator

# Exceptions with actionable error messages
from .exceptions import (
    InitialConditionError,
    IntegrationError,
    MechanicsDSLError,
    NoCoordinatesError,
    NoLagrangianError,
    ParameterError,
    ParseError,
    SemanticError,
    SimulationError,
    TokenizationError,
)
from .parser import MechanicsParser, tokenize
from .solver import NumericalSimulator  # Now imports from solver package
from .symbolic import SymbolicEngine

# Utils imports
from .utils import config, logger, setup_logging

__version__ = "2.0.6"
__author__ = "Noah Parsons"
__license__ = "MIT"

# Lazy import for presets
def get_preset(name):
    """Get a built-in preset by name. See `list_presets()` for available options."""
    from .presets import get_preset as _get_preset
    return _get_preset(name)

def list_presets():
    """List available built-in presets."""
    from .presets import list_presets as _list_presets
    return _list_presets()

__all__ = [
    # Core
    "PhysicsCompiler",
    "MechanicsParser",
    "SymbolicEngine",
    "NumericalSimulator",
    "tokenize",
    # Utils
    "setup_logging",
    "logger",
    "config",
    # Analysis
    "PotentialEnergyCalculator",
    # Presets
    "get_preset",
    "list_presets",
    # Exceptions
    "MechanicsDSLError",
    "ParseError",
    "TokenizationError",
    "SemanticError",
    "NoLagrangianError",
    "NoCoordinatesError",
    "SimulationError",
    "IntegrationError",
    "InitialConditionError",
    "ParameterError",
]
