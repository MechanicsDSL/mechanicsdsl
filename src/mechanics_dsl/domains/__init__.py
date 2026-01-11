"""
MechanicsDSL Domains Package

Domain-specific physics implementations following a common interface.

The domains have been reorganized from single files to packages for better
maintainability. All imports remain backward compatible.
"""

from .base import PhysicsDomain

# Physics domain packages
from . import classical
from . import fluids
from . import kinematics

# Domain packages (reorganized from single files)
from . import electromagnetic
from . import relativistic
from . import quantum
from . import general_relativity
from . import statistical
from . import thermodynamics

__all__ = [
    'PhysicsDomain',
    # Packages
    'classical',
    'fluids',
    'kinematics',
    'electromagnetic',
    'relativistic', 
    'quantum',
    'general_relativity',
    'statistical',
    'thermodynamics',
]
