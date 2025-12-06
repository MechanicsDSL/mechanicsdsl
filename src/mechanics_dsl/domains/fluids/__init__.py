"""
Fluids Domain Package

Smoothed Particle Hydrodynamics (SPH) and fluid dynamics implementations.
"""

from .sph import SPHFluid
from .boundary import BoundaryConditions

__all__ = ['SPHFluid', 'BoundaryConditions']
