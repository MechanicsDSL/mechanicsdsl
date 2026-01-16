"""
MechanicsDSL Solver Package

This package provides numerical simulation capabilities for physics systems
defined in MechanicsDSL.

Modules:
    core: Main NumericalSimulator class
    symplectic: Structure-preserving symplectic integrators for Hamiltonian systems
    variational: Discrete variational integrators from Hamilton's principle

Symplectic Integrators:
    StormerVerlet, Leapfrog, Yoshida4, Ruth3, McLachlan4
    - Preserve symplectic structure and bounded energy error
    - Ideal for long-time integration of Hamiltonian systems

Variational Integrators:
    MidpointVariational, TrapezoidalVariational, GalerkinVariational
    - Derived from discrete Hamilton's principle
    - Exactly preserve momentum maps (Noether's theorem)

Quick Start:
    >>> from mechanics_dsl.solver import NumericalSimulator
    >>> from mechanics_dsl.solver.symplectic import StormerVerlet
    >>> verlet = StormerVerlet()
    >>> result = verlet.integrate(t_span, q0, p0, h, grad_T, grad_V)
"""

from .core import NumericalSimulator

# Symplectic integrators
from .symplectic import (
    Leapfrog,
    McLachlan4,
    Ruth3,
    StormerVerlet,
    SymplecticIntegrator,
    Yoshida4,
    get_symplectic_integrator,
)

# Variational integrators
from .variational import (
    GalerkinVariational,
    MidpointVariational,
    TrapezoidalVariational,
    VariationalIntegrator,
    get_variational_integrator,
)

__all__ = [
    # Core
    "NumericalSimulator",
    # Symplectic integrators
    "SymplecticIntegrator",
    "StormerVerlet",
    "Leapfrog",
    "Yoshida4",
    "Ruth3",
    "McLachlan4",
    "get_symplectic_integrator",
    # Variational integrators
    "VariationalIntegrator",
    "MidpointVariational",
    "TrapezoidalVariational",
    "GalerkinVariational",
    "get_variational_integrator",
]
