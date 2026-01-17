"""
Quantum Mechanics Domain for MechanicsDSL

This package provides tools for semiclassical quantum mechanics, including:
- WKB approximation
- Bohr-Sommerfeld quantization
- Ehrenfest theorem (quantum-classical correspondence)
- Quantum harmonic oscillator
- Quantum wells and barriers
- Hydrogen atom

All classes and functions are re-exported from the core module.
"""

from .core import (
    HBAR,
    PLANCK_H,
    EhrenfestDynamics,
    EnergyLevel,
    FiniteSquareWell,
    HydrogenAtom,
    InfiniteSquareWell,
    QuantumHarmonicOscillator,
    QuantumState,
    QuantumTunneling,
    WKBApproximation,
)

__all__ = [
    "EhrenfestDynamics",
    "EnergyLevel",
    "FiniteSquareWell",
    "HBAR",
    "HydrogenAtom",
    "InfiniteSquareWell",
    "PLANCK_H",
    "QuantumHarmonicOscillator",
    "QuantumState",
    "QuantumTunneling",
    "WKBApproximation",
]
