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

# Import all symbols from core module
try:
    from .core import (
    HBAR,
    PLANCK_H,
    DeltaFunctionBarrier,
    EhrenfestDynamics,
    EnergyLevel,
    FiniteSquareWell,
    HydrogenAtom,
    InfiniteSquareWell,
    QuantumHarmonicOscillator,
    QuantumState,
    QuantumTunneling,
    StepPotential,
    WKBApproximation,
    alpha_decay_rate,
    compton_wavelength,
    de_broglie_wavelength,
    heisenberg_minimum,
    tunneling_probability_rectangular,
    )
except ImportError as e:
    # Re-raise with more context
    raise ImportError(f"Failed to import from quantum.core: {e}") from e

__all__ = [
    "DeltaFunctionBarrier",
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
    "StepPotential",
    "WKBApproximation",
    "alpha_decay_rate",
    "compton_wavelength",
    "de_broglie_wavelength",
    "heisenberg_minimum",
    "tunneling_probability_rectangular",
]
