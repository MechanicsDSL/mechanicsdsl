"""
General Relativity Domain for MechanicsDSL

Provides tools for general relativistic calculations.
"""

from .core import (
    GeodesicSolver,
    SchwarzschildMetric,
)

__all__ = [
    "GeodesicSolver",
    "SchwarzschildMetric",
]
