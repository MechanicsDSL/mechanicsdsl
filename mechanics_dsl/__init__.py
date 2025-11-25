"""
MechanicsDSL: A Domain-Specific Language for Classical Mechanics.

This package provides a compiler and simulation engine for Lagrangian and 
Hamiltonian mechanics, using a LaTeX-inspired DSL.
"""

# -----------------------------------------------------------------------------
# CRITICAL: This assumes you have renamed 'mechanics_dsl_v5.py' to 'core.py'
# inside the mechanics_dsl/ folder.
# -----------------------------------------------------------------------------

from .core import (
    PhysicsCompiler,
    SymbolicEngine,
    NumericalSimulator,
    MechanicsVisualizer,
    run_example,
    config,
    __version__ as core_version
)

# -----------------------------------------------------------------------------
# CRITICAL: This assumes you have renamed 'mechanics_dsl_3d.py' to 'rotation_3d.py'
# inside the mechanics_dsl/ folder.
# -----------------------------------------------------------------------------

from .rotation_3d import (
    EulerAngles,
    Quaternion,
    NonConservativeForces,
    NonHolonomicConstraints
)

# This list determines what is available when a user types:
# from mechanics_dsl import *
__all__ = [
    "PhysicsCompiler",
    "SymbolicEngine",
    "NumericalSimulator",
    "MechanicsVisualizer",
    "run_example",
    "config",
    "EulerAngles",
    "Quaternion",
    "NonConservativeForces",
    "NonHolonomicConstraints",
]

__version__ = "0.5.0"
__author__ = "Noah Parsons"
