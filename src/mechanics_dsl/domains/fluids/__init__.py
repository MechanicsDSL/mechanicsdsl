"""
Fluids Domain Package

Smoothed Particle Hydrodynamics (SPH) and fluid dynamics implementations.
"""

from .boundary import BoundaryConditions
from .sph import SPHFluid

__all__ = ["SPHFluid", "BoundaryConditions"]
