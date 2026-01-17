"""
Electromagnetic Domain for MechanicsDSL

Provides tools for charged particle dynamics in electromagnetic fields.
"""

from .core import (
    ChargedParticle,
    CyclotronMotion,
    DipoleTrap,
    ElectromagneticField,
    ElectromagneticWave,
    FieldType,
)

__all__ = [
    "ChargedParticle",
    "CyclotronMotion",
    "DipoleTrap",
    "ElectromagneticField",
    "ElectromagneticWave",
    "FieldType",
]
