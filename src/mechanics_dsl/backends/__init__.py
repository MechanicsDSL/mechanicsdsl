"""
MechanicsDSL Backends Package

Alternative simulation backends including JAX for GPU acceleration.

Example:
    from mechanics_dsl.backends import JAXBackend

    backend = JAXBackend(use_gpu=True)
    result = backend.simulate(equations, t_span, y0)
"""

from .base import Backend, BackendCapabilities
from .jax_backend import JAXBackend, JAXSolver

__all__ = [
    "Backend",
    "BackendCapabilities",
    "JAXBackend",
    "JAXSolver",
]
