"""
Electromagnetic Domain for MechanicsDSL

Provides tools for charged particle dynamics in electromagnetic fields.
"""

from .core import (
    EPSILON_0,
    IMPEDANCE_FREE_SPACE,
    MU_0,
    SPEED_OF_LIGHT,
    Antenna,
    ChargedParticle,
    CyclotronMotion,
    DipoleTrap,
    ElectromagneticField,
    ElectromagneticWave,
    FieldType,
    GradientDrift,
    PenningTrap,
    SkinEffect,
    Waveguide,
    biot_savart_field,
    calculate_drift_velocity,
    cyclotron_resonance_frequency,
    debye_length,
    magnetic_moment,
    plasma_frequency,
    uniform_crossed_fields,
)

__all__ = [
    "Antenna",
    "ChargedParticle",
    "CyclotronMotion",
    "DipoleTrap",
    "ElectromagneticField",
    "ElectromagneticWave",
    "EPSILON_0",
    "FieldType",
    "GradientDrift",
    "IMPEDANCE_FREE_SPACE",
    "MU_0",
    "PenningTrap",
    "SkinEffect",
    "SPEED_OF_LIGHT",
    "Waveguide",
    "biot_savart_field",
    "calculate_drift_velocity",
    "cyclotron_resonance_frequency",
    "debye_length",
    "magnetic_moment",
    "plasma_frequency",
    "uniform_crossed_fields",
]
