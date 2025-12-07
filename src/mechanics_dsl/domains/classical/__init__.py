"""
Classical Mechanics Domain

Implements Lagrangian and Hamiltonian mechanics for point particles and rigid bodies.

This package provides comprehensive tools for classical mechanics:
- Lagrangian mechanics with Euler-Lagrange equations
- Hamiltonian mechanics with Hamilton's equations
- Constraint handling (holonomic and non-holonomic)
- Rigid body dynamics with Euler angles and quaternions
- Dissipation and non-conservative forces
- Stability analysis and normal modes
- Noether's theorem and conservation laws
- Central force problems and orbital mechanics
- Canonical transformations
- Perturbation theory
- Collision dynamics
- Scattering theory
- Variable mass systems
- Continuous systems (strings, membranes)
"""

from .lagrangian import LagrangianMechanics
from .hamiltonian import HamiltonianMechanics
from .constraints import ConstraintHandler, BaumgarteStabilization, ConstrainedLagrangianSystem
from .rigidbody import RigidBodyDynamics, SymmetricTop, Gyroscope

# New modules for bulletproof classical mechanics
from .dissipation import (
    RayleighDissipation,
    FrictionModel,
    FrictionType,
    GeneralizedForce,
    DissipativeLagrangianMechanics
)

from .stability import (
    StabilityAnalyzer,
    StabilityResult,
    StabilityType,
    EquilibriumPoint,
    find_equilibria,
    analyze_stability
)

from .symmetry import (
    NoetherAnalyzer,
    ConservedQuantity,
    SymmetryType,
    SymmetryInfo,
    detect_cyclic_coordinates,
    get_conserved_quantities
)

from .central_forces import (
    CentralForceAnalyzer,
    EffectivePotential,
    KeplerProblem,
    OrbitalElements,
    OrbitType,
    TurningPoints
)

from .canonical import (
    CanonicalTransformation,
    GeneratingFunction,
    GeneratingFunctionType,
    ActionAngleVariables,
    HamiltonJacobi
)

from .oscillations import (
    NormalModeAnalyzer,
    NormalMode,
    ModalAnalysisResult,
    CoupledOscillatorSystem,
    extract_mass_matrix,
    extract_stiffness_matrix,
    compute_normal_modes
)

# Additional modules completing the classical mechanics domain
from .perturbation import (
    PerturbationExpander,
    PerturbationResult,
    PerturbationType,
    AveragingMethod,
    MultiScaleAnalysis,
    perturbation_expand,
    average_over_angle
)

from .nonholonomic import (
    NonholonomicSystem,
    NonholonomicConstraint,
    ConstraintType,
    AppellEquations,
    MaggiEquations,
    rolling_constraint,
    knife_edge_constraint
)

from .collisions import (
    CollisionSolver,
    CollisionResult,
    CollisionType,
    Particle,
    SymbolicCollisionSolver,
    ImpulseCalculator,
    elastic_collision_1d,
    inelastic_collision_1d,
    perfectly_inelastic_1d
)

from .scattering import (
    ScatteringAnalyzer,
    ScatteringResult,
    SymbolicScattering,
    rutherford_angle,
    rutherford_cross_section
)

from .variable_mass import (
    RocketEquation,
    RocketParameters,
    RocketState,
    VariableMassSystem,
    SymbolicVariableMass,
    tsiolkovsky_delta_v,
    required_mass_ratio,
    specific_impulse_to_exhaust_velocity
)

from .continuum import (
    LagrangianDensity,
    FieldEulerLagrange,
    VibratingString,
    VibratingMembrane,
    StressEnergyTensor,
    WaveMode,
    FieldConfiguration,
    string_mode_frequencies,
    wave_speed
)

__all__ = [
    # Core mechanics
    'LagrangianMechanics',
    'HamiltonianMechanics',
    'ConstraintHandler',
    'BaumgarteStabilization',
    'ConstrainedLagrangianSystem',
    'RigidBodyDynamics',
    'SymmetricTop',
    'Gyroscope',
    
    # Dissipation
    'RayleighDissipation',
    'FrictionModel',
    'FrictionType',
    'GeneralizedForce',
    'DissipativeLagrangianMechanics',
    
    # Stability
    'StabilityAnalyzer',
    'StabilityResult',
    'StabilityType',
    'EquilibriumPoint',
    'find_equilibria',
    'analyze_stability',
    
    # Symmetry / Noether
    'NoetherAnalyzer',
    'ConservedQuantity',
    'SymmetryType',
    'SymmetryInfo',
    'detect_cyclic_coordinates',
    'get_conserved_quantities',
    
    # Central forces
    'CentralForceAnalyzer',
    'EffectivePotential',
    'KeplerProblem',
    'OrbitalElements',
    'OrbitType',
    'TurningPoints',
    
    # Canonical transformations
    'CanonicalTransformation',
    'GeneratingFunction',
    'GeneratingFunctionType',
    'ActionAngleVariables',
    'HamiltonJacobi',
    
    # Oscillations
    'NormalModeAnalyzer',
    'NormalMode',
    'ModalAnalysisResult',
    'CoupledOscillatorSystem',
    'extract_mass_matrix',
    'extract_stiffness_matrix',
    'compute_normal_modes',
    
    # Perturbation theory
    'PerturbationExpander',
    'PerturbationResult',
    'PerturbationType',
    'AveragingMethod',
    'MultiScaleAnalysis',
    'perturbation_expand',
    'average_over_angle',
    
    # Non-holonomic constraints
    'NonholonomicSystem',
    'NonholonomicConstraint',
    'ConstraintType',
    'AppellEquations',
    'MaggiEquations',
    'rolling_constraint',
    'knife_edge_constraint',
    
    # Collisions
    'CollisionSolver',
    'CollisionResult',
    'CollisionType',
    'Particle',
    'SymbolicCollisionSolver',
    'ImpulseCalculator',
    'elastic_collision_1d',
    'inelastic_collision_1d',
    'perfectly_inelastic_1d',
    
    # Scattering
    'ScatteringAnalyzer',
    'ScatteringResult',
    'SymbolicScattering',
    'rutherford_angle',
    'rutherford_cross_section',
    
    # Variable mass
    'RocketEquation',
    'RocketParameters',
    'RocketState',
    'VariableMassSystem',
    'SymbolicVariableMass',
    'tsiolkovsky_delta_v',
    'required_mass_ratio',
    'specific_impulse_to_exhaust_velocity',
    
    # Continuum mechanics
    'LagrangianDensity',
    'FieldEulerLagrange',
    'VibratingString',
    'VibratingMembrane',
    'StressEnergyTensor',
    'WaveMode',
    'FieldConfiguration',
    'string_mode_frequencies',
    'wave_speed',
]

