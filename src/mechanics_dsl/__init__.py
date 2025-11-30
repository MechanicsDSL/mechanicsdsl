"""
MechanicsDSL: A Domain-Specific Language for Classical Mechanics

A comprehensive framework for symbolic and numerical analysis of classical 
mechanical systems using LaTeX-inspired notation.
"""

from .compiler import PhysicsCompiler
from .utils import setup_logging
from .energy import PotentialEnergyCalculator

__version__ = "0.6.1"
__author__ = "Noah Parsons"
__license__ = "MIT"

__all__ = ['PhysicsCompiler', 'setup_logging']
