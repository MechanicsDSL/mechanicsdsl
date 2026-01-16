"""
MechanicsDSL Domains Package

Domain-specific physics implementations following a common interface.

The domains have been reorganized from single files to packages for better
maintainability. All imports remain backward compatible.
"""

# Domain packages (reorganized from single files)
# Physics domain packages
from . import (
    classical,
    electromagnetic,
    fluids,
    general_relativity,
    kinematics,
    quantum,
    relativistic,
    statistical,
    thermodynamics,
)
from .base import PhysicsDomain

__all__ = [
    "PhysicsDomain",
    # Packages
    "classical",
    "fluids",
    "kinematics",
    "electromagnetic",
    "relativistic",
    "quantum",
    "general_relativity",
    "statistical",
    "thermodynamics",
]
