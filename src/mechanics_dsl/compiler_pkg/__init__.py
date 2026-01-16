"""
MechanicsDSL Compiler Package

This package provides the main compilation infrastructure for converting
MechanicsDSL source code into executable physics simulations.

Modules:
    serializer: System state serialization and deserialization.
    particles: Particle generation for SPH simulations.

The main PhysicsCompiler class is imported from the parent module for
backward compatibility. All submodules are documented with comprehensive
docstrings.

Quick Start:
    >>> from mechanics_dsl import PhysicsCompiler
    >>> compiler = PhysicsCompiler()
    >>> result = compiler.compile_dsl(source)
"""

from .particles import ParticleGenerator
from .serializer import SystemSerializer

__all__ = [
    "SystemSerializer",
    "ParticleGenerator",
]
