"""
MechanicsDSL Domains Package

Domain-specific physics implementations following a common interface.
"""

from .base import PhysicsDomain

# Physics domains
from . import electromagnetic
from . import relativistic
from . import quantum
from . import general_relativity
from . import statistical
from . import thermodynamics
from . import kinematics

__all__ = [
    'PhysicsDomain',
    'electromagnetic',
    'relativistic', 
    'quantum',
    'general_relativity',
    'statistical',
    'thermodynamics',
    'kinematics',
]


