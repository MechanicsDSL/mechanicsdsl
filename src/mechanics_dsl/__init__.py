"""
MechanicsDSL: A Domain-Specific Language for Classical Mechanics

A comprehensive framework for symbolic and numerical analysis of classical 
mechanical systems using LaTeX-inspired notation.
"""

# Core imports - use new package structure
from .core.compiler import PhysicsCompiler
from .core.parser import tokenize, MechanicsParser
from .core.symbolic import SymbolicEngine
from .core.solver import NumericalSimulator

# Utils imports
from .utils import setup_logging, logger, config

# Analysis imports
from .energy import PotentialEnergyCalculator

# Backward compatibility: also expose from old locations
# These will be deprecated in a future version
from .compiler import PhysicsCompiler as _PhysicsCompiler
from .parser import MechanicsParser as _MechanicsParser
from .symbolic import SymbolicEngine as _SymbolicEngine
from .solver import NumericalSimulator as _NumericalSimulator

__version__ = "1.3.0"
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
]
