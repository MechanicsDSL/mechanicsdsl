"""
Thermodynamics Domain for MechanicsDSL

Provides tools for thermodynamic processes and heat engines.
"""

from .core import (
    BOLTZMANN,
    R_GAS,
    CarnotEngine,
    DieselCycle,
    HeatCapacity,
    MaxwellRelations,
    OttoCycle,
    PhaseTransition,
    ProcessType,
    ThermodynamicProcess,
    VanDerWaalsGas,
)

__all__ = [
    "BOLTZMANN",
    "CarnotEngine",
    "DieselCycle",
    "HeatCapacity",
    "MaxwellRelations",
    "OttoCycle",
    "PhaseTransition",
    "ProcessType",
    "R_GAS",
    "ThermodynamicProcess",
    "VanDerWaalsGas",
]
