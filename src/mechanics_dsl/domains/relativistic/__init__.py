"""
Special Relativistic Mechanics Domain for MechanicsDSL

Provides tools for relativistic particle dynamics.
"""

from .core import (
    SPEED_OF_LIGHT,
    DopplerEffect,
    FourVector,
    LorentzTransform,
    RelativisticCollision,
    RelativisticParticle,
    beta,
    gamma,
    rapidity,
)

__all__ = [
    "DopplerEffect",
    "FourVector",
    "LorentzTransform",
    "RelativisticCollision",
    "RelativisticParticle",
    "SPEED_OF_LIGHT",
    "beta",
    "gamma",
    "rapidity",
]
