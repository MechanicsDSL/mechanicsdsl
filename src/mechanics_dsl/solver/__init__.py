"""
MechanicsDSL Solver Package

This package provides numerical simulation capabilities for physics systems
defined in MechanicsDSL.

Modules:
    core: Main NumericalSimulator class
    integrators: Custom integration methods (future)
    adaptive: Adaptive step-size control (future)

Quick Start:
    >>> from mechanics_dsl.solver import NumericalSimulator
    >>> from mechanics_dsl.symbolic import SymbolicEngine
    >>> symbolic = SymbolicEngine()
    >>> simulator = NumericalSimulator(symbolic)
    >>> simulator.set_parameters({'m': 1.0, 'g': 9.81})
    >>> solution = simulator.simulate((0, 10))
"""

from .core import NumericalSimulator

__all__ = ['NumericalSimulator']
