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

from .canonical import (
    ActionAngleVariables,
    CanonicalTransformation,
    GeneratingFunction,
    GeneratingFunctionType,
    HamiltonJacobi,
)
from .central_forces import (
    CentralForceAnalyzer,
    EffectivePotential,
    KeplerProblem,
    OrbitalElements,
    OrbitType,
    TurningPoints,
)
from .collisions import (
    CollisionResult,
    CollisionSolver,
    CollisionType,
    ImpulseCalculator,
    Particle,
    SymbolicCollisionSolver,
    elastic_collision_1d,
    inelastic_collision_1d,
    perfectly_inelastic_1d,
)
from .constraints import BaumgarteStabilization, ConstrainedLagrangianSystem, ConstraintHandler
from .continuum import (
    FieldConfiguration,
    FieldEulerLagrange,
    LagrangianDensity,
    StressEnergyTensor,
    VibratingMembrane,
    VibratingString,
    WaveMode,
    string_mode_frequencies,
    wave_speed,
)

# New modules for bulletproof classical mechanics
from .dissipation import (
    DissipativeLagrangianMechanics,
    FrictionModel,
    FrictionType,
    GeneralizedForce,
    RayleighDissipation,
)
from .hamiltonian import HamiltonianMechanics
from .lagrangian import LagrangianMechanics
from .nonholonomic import (
    AppellEquations,
    ConstraintType,
    MaggiEquations,
    NonholonomicConstraint,
    NonholonomicSystem,
    knife_edge_constraint,
    rolling_constraint,
)
from .oscillations import (
    CoupledOscillatorSystem,
    ModalAnalysisResult,
    NormalMode,
    NormalModeAnalyzer,
    compute_normal_modes,
    extract_mass_matrix,
    extract_stiffness_matrix,
)

# Additional modules completing the classical mechanics domain
from .perturbation import (
    AveragingMethod,
    MultiScaleAnalysis,
    PerturbationExpander,
    PerturbationResult,
    PerturbationType,
    average_over_angle,
    perturbation_expand,
)
from .rigidbody import Gyroscope, RigidBodyDynamics, SymmetricTop
from .scattering import (
    ScatteringAnalyzer,
    ScatteringResult,
    SymbolicScattering,
    rutherford_angle,
    rutherford_cross_section,
)
from .stability import (
    EquilibriumPoint,
    StabilityAnalyzer,
    StabilityResult,
    StabilityType,
    analyze_stability,
    find_equilibria,
)
from .symmetry import (
    ConservedQuantity,
    NoetherAnalyzer,
    SymmetryInfo,
    SymmetryType,
    detect_cyclic_coordinates,
    get_conserved_quantities,
)
from .variable_mass import (
    RocketEquation,
    RocketParameters,
    RocketState,
    SymbolicVariableMass,
    VariableMassSystem,
    required_mass_ratio,
    specific_impulse_to_exhaust_velocity,
    tsiolkovsky_delta_v,
)

__all__ = [
    # Core mechanics
    "LagrangianMechanics",
    "HamiltonianMechanics",
    "ConstraintHandler",
    "BaumgarteStabilization",
    "ConstrainedLagrangianSystem",
    "RigidBodyDynamics",
    "SymmetricTop",
    "Gyroscope",
    # Dissipation
    "RayleighDissipation",
    "FrictionModel",
    "FrictionType",
    "GeneralizedForce",
    "DissipativeLagrangianMechanics",
    # Stability
    "StabilityAnalyzer",
    "StabilityResult",
    "StabilityType",
    "EquilibriumPoint",
    "find_equilibria",
    "analyze_stability",
    # Symmetry / Noether
    "NoetherAnalyzer",
    "ConservedQuantity",
    "SymmetryType",
    "SymmetryInfo",
    "detect_cyclic_coordinates",
    "get_conserved_quantities",
    # Central forces
    "CentralForceAnalyzer",
    "EffectivePotential",
    "KeplerProblem",
    "OrbitalElements",
    "OrbitType",
    "TurningPoints",
    # Canonical transformations
    "CanonicalTransformation",
    "GeneratingFunction",
    "GeneratingFunctionType",
    "ActionAngleVariables",
    "HamiltonJacobi",
    # Oscillations
    "NormalModeAnalyzer",
    "NormalMode",
    "ModalAnalysisResult",
    "CoupledOscillatorSystem",
    "extract_mass_matrix",
    "extract_stiffness_matrix",
    "compute_normal_modes",
    # Perturbation theory
    "PerturbationExpander",
    "PerturbationResult",
    "PerturbationType",
    "AveragingMethod",
    "MultiScaleAnalysis",
    "perturbation_expand",
    "average_over_angle",
    # Non-holonomic constraints
    "NonholonomicSystem",
    "NonholonomicConstraint",
    "ConstraintType",
    "AppellEquations",
    "MaggiEquations",
    "rolling_constraint",
    "knife_edge_constraint",
    # Collisions
    "CollisionSolver",
    "CollisionResult",
    "CollisionType",
    "Particle",
    "SymbolicCollisionSolver",
    "ImpulseCalculator",
    "elastic_collision_1d",
    "inelastic_collision_1d",
    "perfectly_inelastic_1d",
    # Scattering
    "ScatteringAnalyzer",
    "ScatteringResult",
    "SymbolicScattering",
    "rutherford_angle",
    "rutherford_cross_section",
    # Variable mass
    "RocketEquation",
    "RocketParameters",
    "RocketState",
    "VariableMassSystem",
    "SymbolicVariableMass",
    "tsiolkovsky_delta_v",
    "required_mass_ratio",
    "specific_impulse_to_exhaust_velocity",
    # Continuum mechanics
    "LagrangianDensity",
    "FieldEulerLagrange",
    "VibratingString",
    "VibratingMembrane",
    "StressEnergyTensor",
    "WaveMode",
    "FieldConfiguration",
    "string_mode_frequencies",
    "wave_speed",
]
