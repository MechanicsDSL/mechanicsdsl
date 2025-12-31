"""
MechanicsDSL Domains Package

Domain-specific physics implementations following a common interface.
"""

from .base import PhysicsDomain

# New physics domains
from . import electromagnetic
from . import relativistic
from . import quantum

__all__ = [
    'PhysicsDomain',
    'electromagnetic',
    'relativistic', 
    'quantum',
]
