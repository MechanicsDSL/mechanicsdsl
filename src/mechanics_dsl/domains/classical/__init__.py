"""
Classical Mechanics Domain

Implements Lagrangian and Hamiltonian mechanics for point particles and rigid bodies.
"""

from .lagrangian import LagrangianMechanics
from .hamiltonian import HamiltonianMechanics
from .constraints import ConstraintHandler
from .rigidbody import RigidBodyDynamics

__all__ = [
    'LagrangianMechanics',
    'HamiltonianMechanics',
    'ConstraintHandler',
    'RigidBodyDynamics',
]
