"""
Linear Elasticity Module for Solid Mechanics Domain

Implements Hooke's law, stress-strain relationships, and elastic material models
for isotropic, orthotropic, and transversely isotropic materials.

Security: All inputs validated for finite values and proper tensor symmetry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import sympy as sp

from ...utils import logger, validate_finite


class MaterialSymmetry(Enum):
    """Classification of material symmetry types."""

    ISOTROPIC = auto()
    TRANSVERSELY_ISOTROPIC = auto()
    ORTHOTROPIC = auto()
    MONOCLINIC = auto()
    TRICLINIC = auto()


@dataclass(frozen=True)
class LameConstants:
    """
    Lamé constants (λ, μ) for isotropic elastic materials.

    Attributes:
        lambda_: First Lamé constant (λ) in Pa
        mu: Second Lamé constant / shear modulus (μ) in Pa
    """

    lambda_: float
    mu: float

    def __post_init__(self):
        if not math.isfinite(self.lambda_):
            raise ValueError("Lamé constant λ must be finite")
        if not math.isfinite(self.mu) or self.mu <= 0:
            raise ValueError("Shear modulus μ must be positive and finite")


@dataclass
class ElasticConstants:
    """
    Complete set of elastic constants for an isotropic material.

    Any two independent constants can be used to derive all others.

    Attributes:
        E: Young's modulus (Pa)
        nu: Poisson's ratio (dimensionless, -1 < ν < 0.5)
        G: Shear modulus (Pa)
        K: Bulk modulus (Pa)
        lambda_: First Lamé constant (Pa)
        M: P-wave modulus (Pa)
    """

    E: float
    nu: float
    G: Optional[float] = None
    K: Optional[float] = None
    lambda_: Optional[float] = None
    M: Optional[float] = None

    def __post_init__(self):
        # Validate primary inputs
        if not math.isfinite(self.E) or self.E <= 0:
            raise ValueError(f"Young's modulus must be positive finite, got {self.E}")
        if not math.isfinite(self.nu):
            raise ValueError(f"Poisson's ratio must be finite, got {self.nu}")
        if self.nu <= -1.0 or self.nu >= 0.5:
            raise ValueError(f"Poisson's ratio must be in (-1, 0.5), got {self.nu}")

        # Derive remaining constants
        if self.G is None:
            self.G = self.E / (2 * (1 + self.nu))
        if self.K is None:
            self.K = self.E / (3 * (1 - 2 * self.nu))
        if self.lambda_ is None:
            self.lambda_ = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        if self.M is None:
            self.M = self.E * (1 - self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))

    def to_lame(self) -> LameConstants:
        """Convert to Lamé constants."""
        return LameConstants(lambda_=self.lambda_, mu=self.G)


@dataclass
class ElasticMaterial:
    """
    Represents an elastic material with its properties.

    Attributes:
        name: Material identifier
        constants: Elastic constants
        density: Mass density (kg/m³)
        symmetry: Material symmetry type
    """

    name: str
    constants: ElasticConstants
    density: float
    symmetry: MaterialSymmetry = MaterialSymmetry.ISOTROPIC

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Material name must be a non-empty string")
        if not math.isfinite(self.density) or self.density <= 0:
            raise ValueError(f"Density must be positive finite, got {self.density}")

    @classmethod
    def steel(cls) -> ElasticMaterial:
        """Standard structural steel properties."""
        return cls(name="Steel", constants=ElasticConstants(E=200e9, nu=0.3), density=7850.0)

    @classmethod
    def aluminum(cls) -> ElasticMaterial:
        """6061-T6 Aluminum alloy properties."""
        return cls(
            name="Aluminum_6061", constants=ElasticConstants(E=68.9e9, nu=0.33), density=2700.0
        )

    @classmethod
    def titanium(cls) -> ElasticMaterial:
        """Ti-6Al-4V titanium alloy properties."""
        return cls(
            name="Titanium_Ti6Al4V", constants=ElasticConstants(E=113.8e9, nu=0.342), density=4430.0
        )

    @classmethod
    def copper(cls) -> ElasticMaterial:
        """Pure copper properties."""
        return cls(name="Copper", constants=ElasticConstants(E=117e9, nu=0.35), density=8960.0)


class VoigtNotation:
    """
    Converts between tensor and Voigt notation for stress and strain.

    Voigt ordering: [11, 22, 33, 23, 13, 12] (engineering notation)
    """

    # Index mapping: Voigt index -> (i, j) tensor indices
    VOIGT_TO_TENSOR = {0: (0, 0), 1: (1, 1), 2: (2, 2), 3: (1, 2), 4: (0, 2), 5: (0, 1)}

    # Reverse mapping
    TENSOR_TO_VOIGT = {v: k for k, v in VOIGT_TO_TENSOR.items()}

    @staticmethod
    def stress_tensor_to_voigt(tensor: np.ndarray) -> np.ndarray:
        """
        Convert 3x3 stress tensor to 6-element Voigt vector.

        Args:
            tensor: 3x3 symmetric stress tensor

        Returns:
            6-element array [σ11, σ22, σ33, σ23, σ13, σ12]
        """
        if tensor.shape != (3, 3):
            raise ValueError(f"Expected 3x3 tensor, got shape {tensor.shape}")

        return np.array(
            [tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[1, 2], tensor[0, 2], tensor[0, 1]]
        )

    @staticmethod
    def voigt_to_stress_tensor(voigt: np.ndarray) -> np.ndarray:
        """
        Convert 6-element Voigt vector to 3x3 stress tensor.

        Args:
            voigt: 6-element stress vector

        Returns:
            3x3 symmetric stress tensor
        """
        if voigt.shape != (6,):
            raise ValueError(f"Expected 6-element vector, got shape {voigt.shape}")

        return np.array(
            [
                [voigt[0], voigt[5], voigt[4]],
                [voigt[5], voigt[1], voigt[3]],
                [voigt[4], voigt[3], voigt[2]],
            ]
        )

    @staticmethod
    def strain_tensor_to_voigt(tensor: np.ndarray) -> np.ndarray:
        """
        Convert 3x3 strain tensor to 6-element Voigt vector.

        Note: Engineering shear strains (γ = 2ε) are used for off-diagonal terms.

        Args:
            tensor: 3x3 symmetric strain tensor

        Returns:
            6-element array [ε11, ε22, ε33, γ23, γ13, γ12]
        """
        if tensor.shape != (3, 3):
            raise ValueError(f"Expected 3x3 tensor, got shape {tensor.shape}")

        return np.array(
            [
                tensor[0, 0],
                tensor[1, 1],
                tensor[2, 2],
                2 * tensor[1, 2],
                2 * tensor[0, 2],
                2 * tensor[0, 1],
            ]
        )

    @staticmethod
    def voigt_to_strain_tensor(voigt: np.ndarray) -> np.ndarray:
        """
        Convert 6-element Voigt strain vector to 3x3 tensor.

        Args:
            voigt: 6-element strain vector with engineering shear strains

        Returns:
            3x3 symmetric strain tensor
        """
        if voigt.shape != (6,):
            raise ValueError(f"Expected 6-element vector, got shape {voigt.shape}")

        return np.array(
            [
                [voigt[0], voigt[5] / 2, voigt[4] / 2],
                [voigt[5] / 2, voigt[1], voigt[3] / 2],
                [voigt[4] / 2, voigt[3] / 2, voigt[2]],
            ]
        )


class StiffnessMatrix:
    """
    6x6 stiffness matrix (C) relating stress to strain in Voigt notation.

    σ = C · ε
    """

    def __init__(self, matrix: np.ndarray):
        """
        Initialize stiffness matrix with validation.

        Args:
            matrix: 6x6 stiffness matrix
        """
        if matrix.shape != (6, 6):
            raise ValueError(f"Stiffness matrix must be 6x6, got {matrix.shape}")
        if not np.allclose(matrix, matrix.T, rtol=1e-10):
            raise ValueError("Stiffness matrix must be symmetric")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Stiffness matrix contains non-finite values")

        # Check positive definiteness
        eigenvalues = np.linalg.eigvalsh(matrix)
        if np.any(eigenvalues <= 0):
            raise ValueError("Stiffness matrix must be positive definite")

        self._matrix = matrix.copy()
        self._matrix.flags.writeable = False

    @property
    def matrix(self) -> np.ndarray:
        """Return immutable view of stiffness matrix."""
        return self._matrix

    def apply(self, strain_voigt: np.ndarray) -> np.ndarray:
        """
        Compute stress from strain using σ = C · ε.

        Args:
            strain_voigt: 6-element engineering strain vector

        Returns:
            6-element stress vector
        """
        if strain_voigt.shape != (6,):
            raise ValueError(f"Expected 6-element strain, got {strain_voigt.shape}")
        return self._matrix @ strain_voigt

    def to_compliance(self) -> "ComplianceMatrix":
        """Convert to compliance matrix S = C⁻¹."""
        return ComplianceMatrix(np.linalg.inv(self._matrix))

    @classmethod
    def isotropic(cls, E: float, nu: float) -> "StiffnessMatrix":
        """
        Create isotropic stiffness matrix from Young's modulus and Poisson's ratio.

        Args:
            E: Young's modulus (Pa)
            nu: Poisson's ratio

        Returns:
            Isotropic stiffness matrix
        """
        if not math.isfinite(E) or E <= 0:
            raise ValueError(f"Young's modulus must be positive, got {E}")
        if not (-1 < nu < 0.5):
            raise ValueError(f"Poisson's ratio must be in (-1, 0.5), got {nu}")

        factor = E / ((1 + nu) * (1 - 2 * nu))
        G = E / (2 * (1 + nu))

        C = (
            np.array(
                [
                    [1 - nu, nu, nu, 0, 0, 0],
                    [nu, 1 - nu, nu, 0, 0, 0],
                    [nu, nu, 1 - nu, 0, 0, 0],
                    [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
                    [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
                    [0, 0, 0, 0, 0, (1 - 2 * nu) / 2],
                ]
            )
            * factor
        )

        return cls(C)


class ComplianceMatrix:
    """
    6x6 compliance matrix (S) relating strain to stress in Voigt notation.

    ε = S · σ
    """

    def __init__(self, matrix: np.ndarray):
        """
        Initialize compliance matrix with validation.

        Args:
            matrix: 6x6 compliance matrix
        """
        if matrix.shape != (6, 6):
            raise ValueError(f"Compliance matrix must be 6x6, got {matrix.shape}")
        if not np.allclose(matrix, matrix.T, rtol=1e-10):
            raise ValueError("Compliance matrix must be symmetric")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Compliance matrix contains non-finite values")

        self._matrix = matrix.copy()
        self._matrix.flags.writeable = False

    @property
    def matrix(self) -> np.ndarray:
        """Return immutable view of compliance matrix."""
        return self._matrix

    def apply(self, stress_voigt: np.ndarray) -> np.ndarray:
        """
        Compute strain from stress using ε = S · σ.

        Args:
            stress_voigt: 6-element stress vector

        Returns:
            6-element engineering strain vector
        """
        if stress_voigt.shape != (6,):
            raise ValueError(f"Expected 6-element stress, got {stress_voigt.shape}")
        return self._matrix @ stress_voigt

    def to_stiffness(self) -> StiffnessMatrix:
        """Convert to stiffness matrix C = S⁻¹."""
        return StiffnessMatrix(np.linalg.inv(self._matrix))

    @classmethod
    def isotropic(cls, E: float, nu: float) -> "ComplianceMatrix":
        """
        Create isotropic compliance matrix.

        Args:
            E: Young's modulus (Pa)
            nu: Poisson's ratio

        Returns:
            Isotropic compliance matrix
        """
        if not math.isfinite(E) or E <= 0:
            raise ValueError(f"Young's modulus must be positive, got {E}")
        if not (-1 < nu < 0.5):
            raise ValueError(f"Poisson's ratio must be in (-1, 0.5), got {nu}")

        G = E / (2 * (1 + nu))

        S = np.array(
            [
                [1 / E, -nu / E, -nu / E, 0, 0, 0],
                [-nu / E, 1 / E, -nu / E, 0, 0, 0],
                [-nu / E, -nu / E, 1 / E, 0, 0, 0],
                [0, 0, 0, 1 / G, 0, 0],
                [0, 0, 0, 0, 1 / G, 0],
                [0, 0, 0, 0, 0, 1 / G],
            ]
        )

        return cls(S)


class HookesLaw:
    """
    Implementation of Hooke's law for linear elastic materials.

    Provides methods for computing stress from strain and vice versa
    for various material types and loading conditions.
    """

    def __init__(self, material: ElasticMaterial):
        """
        Initialize Hooke's law calculator.

        Args:
            material: Elastic material properties
        """
        self.material = material
        self._stiffness = StiffnessMatrix.isotropic(material.constants.E, material.constants.nu)
        self._compliance = self._stiffness.to_compliance()

    def stress_from_strain(self, strain: np.ndarray) -> np.ndarray:
        """
        Compute stress tensor from strain tensor.

        Args:
            strain: 3x3 strain tensor or 6-element Voigt strain

        Returns:
            Stress in same format as input
        """
        if strain.shape == (3, 3):
            strain_v = VoigtNotation.strain_tensor_to_voigt(strain)
            stress_v = self._stiffness.apply(strain_v)
            return VoigtNotation.voigt_to_stress_tensor(stress_v)
        elif strain.shape == (6,):
            return self._stiffness.apply(strain)
        else:
            raise ValueError(f"Invalid strain shape: {strain.shape}")

    def strain_from_stress(self, stress: np.ndarray) -> np.ndarray:
        """
        Compute strain tensor from stress tensor.

        Args:
            stress: 3x3 stress tensor or 6-element Voigt stress

        Returns:
            Strain in same format as input
        """
        if stress.shape == (3, 3):
            stress_v = VoigtNotation.stress_tensor_to_voigt(stress)
            strain_v = self._compliance.apply(stress_v)
            return VoigtNotation.voigt_to_strain_tensor(strain_v)
        elif stress.shape == (6,):
            return self._compliance.apply(stress)
        else:
            raise ValueError(f"Invalid stress shape: {stress.shape}")

    @property
    def stiffness(self) -> StiffnessMatrix:
        """Get stiffness matrix."""
        return self._stiffness

    @property
    def compliance(self) -> ComplianceMatrix:
        """Get compliance matrix."""
        return self._compliance


class IsotropicElasticity:
    """
    Isotropic linear elasticity with 2 independent constants.

    Provides symbolic and numerical computations for isotropic materials.
    """

    def __init__(self, E: Union[float, sp.Symbol], nu: Union[float, sp.Symbol]):
        """
        Initialize isotropic elasticity.

        Args:
            E: Young's modulus (numeric or symbolic)
            nu: Poisson's ratio (numeric or symbolic)
        """
        self.E = E
        self.nu = nu

        # Derived constants (symbolic-aware)
        self.G = E / (2 * (1 + nu))
        self.K = E / (3 * (1 - 2 * nu))
        self.lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))

    def constitutive_tensor(self) -> sp.Array:
        """
        Get fourth-order constitutive tensor Cijkl.

        Returns:
            SymPy 3x3x3x3 array representing Cijkl
        """
        delta = sp.eye(3)
        C = sp.MutableDenseNDimArray.zeros(3, 3, 3, 3)

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        C[i, j, k, l] = self.lambda_ * delta[i, j] * delta[k, l] + self.G * (
                            delta[i, k] * delta[j, l] + delta[i, l] * delta[j, k]
                        )

        return sp.ImmutableDenseNDimArray(C)

    def wave_speeds(self, rho: Union[float, sp.Symbol]) -> Tuple:
        """
        Compute P-wave and S-wave speeds.

        Args:
            rho: Mass density

        Returns:
            (c_p, c_s) wave speeds
        """
        c_p = sp.sqrt(self.M / rho)
        c_s = sp.sqrt(self.G / rho)
        return (c_p, c_s)


class OrthotropicElasticity:
    """
    Orthotropic linear elasticity with 9 independent constants.

    Used for materials with three mutually perpendicular planes of symmetry
    (wood, fiber-reinforced composites, rolled metals).
    """

    def __init__(
        self,
        E1: float,
        E2: float,
        E3: float,
        nu12: float,
        nu13: float,
        nu23: float,
        G12: float,
        G13: float,
        G23: float,
    ):
        """
        Initialize orthotropic elasticity.

        Args:
            E1, E2, E3: Young's moduli in principal directions
            nu12, nu13, nu23: Poisson's ratios
            G12, G13, G23: Shear moduli
        """
        # Store properties
        self.E1, self.E2, self.E3 = E1, E2, E3
        self.nu12, self.nu13, self.nu23 = nu12, nu13, nu23
        self.G12, self.G13, self.G23 = G12, G13, G23

        # Compute reciprocal Poisson's ratios
        self.nu21 = nu12 * E2 / E1
        self.nu31 = nu13 * E3 / E1
        self.nu32 = nu23 * E3 / E2

        self._validate()
        self._compliance = self._build_compliance_matrix()

    def _validate(self):
        """Validate material constants for thermodynamic consistency."""
        # All moduli must be positive
        for name, val in [
            ("E1", self.E1),
            ("E2", self.E2),
            ("E3", self.E3),
            ("G12", self.G12),
            ("G13", self.G13),
            ("G23", self.G23),
        ]:
            if not math.isfinite(val) or val <= 0:
                raise ValueError(f"{name} must be positive finite, got {val}")

        # Compliance matrix must be positive definite (checked in build)

    def _build_compliance_matrix(self) -> np.ndarray:
        """Build 6x6 compliance matrix."""
        S = np.zeros((6, 6))

        S[0, 0] = 1 / self.E1
        S[1, 1] = 1 / self.E2
        S[2, 2] = 1 / self.E3

        S[0, 1] = S[1, 0] = -self.nu12 / self.E1
        S[0, 2] = S[2, 0] = -self.nu13 / self.E1
        S[1, 2] = S[2, 1] = -self.nu23 / self.E2

        S[3, 3] = 1 / self.G23
        S[4, 4] = 1 / self.G13
        S[5, 5] = 1 / self.G12

        # Verify positive definiteness
        eigenvalues = np.linalg.eigvalsh(S)
        if np.any(eigenvalues <= 0):
            raise ValueError("Material constants violate positive definiteness")

        return S

    def compliance_matrix(self) -> ComplianceMatrix:
        """Get compliance matrix."""
        return ComplianceMatrix(self._compliance)

    def stiffness_matrix(self) -> StiffnessMatrix:
        """Get stiffness matrix."""
        return StiffnessMatrix(np.linalg.inv(self._compliance))


class TransverselyIsotropicElasticity:
    """
    Transversely isotropic elasticity with 5 independent constants.

    Used for materials with one axis of rotational symmetry
    (unidirectional composites, hexagonal crystals).
    """

    def __init__(
        self,
        E_parallel: float,
        E_perpendicular: float,
        nu_parallel: float,
        nu_perpendicular: float,
        G_parallel: float,
    ):
        """
        Initialize transversely isotropic material.

        Args:
            E_parallel: Young's modulus parallel to symmetry axis
            E_perpendicular: Young's modulus perpendicular to axis
            nu_parallel: Poisson's ratio for loading parallel to axis
            nu_perpendicular: Poisson's ratio in the plane of isotropy
            G_parallel: Shear modulus for shear parallel to axis
        """
        self.E_p = E_parallel
        self.E_t = E_perpendicular
        self.nu_p = nu_parallel
        self.nu_t = nu_perpendicular
        self.G_p = G_parallel

        # In-plane shear modulus (derived)
        self.G_t = E_perpendicular / (2 * (1 + nu_perpendicular))

        self._validate()

    def _validate(self):
        """Validate material constants."""
        if self.E_p <= 0 or self.E_t <= 0:
            raise ValueError("Young's moduli must be positive")
        if self.G_p <= 0:
            raise ValueError("Shear modulus must be positive")
        if not (-1 < self.nu_t < 0.5):
            raise ValueError("In-plane Poisson's ratio must be in (-1, 0.5)")


@dataclass
class CauchyStressTensor:
    """
    Cauchy stress tensor representation.

    The Cauchy stress tensor gives the true stress in the deformed configuration.
    """

    components: np.ndarray

    def __post_init__(self):
        if self.components.shape != (3, 3):
            raise ValueError("Stress tensor must be 3x3")
        if not np.allclose(self.components, self.components.T):
            raise ValueError("Stress tensor must be symmetric")

    def hydrostatic(self) -> float:
        """Compute hydrostatic (mean) stress."""
        return np.trace(self.components) / 3

    def deviatoric(self) -> np.ndarray:
        """Compute deviatoric stress tensor."""
        return self.components - self.hydrostatic() * np.eye(3)

    def von_mises(self) -> float:
        """Compute von Mises equivalent stress."""
        s = self.deviatoric()
        return np.sqrt(1.5 * np.sum(s * s))

    def principal_stresses(self) -> np.ndarray:
        """Compute principal stresses (eigenvalues, sorted descending)."""
        eigenvalues = np.linalg.eigvalsh(self.components)
        return np.sort(eigenvalues)[::-1]


@dataclass
class InfinitesimalStrainTensor:
    """
    Infinitesimal (engineering) strain tensor for small deformations.

    ε = (1/2)(∇u + ∇uᵀ)
    """

    components: np.ndarray

    def __post_init__(self):
        if self.components.shape != (3, 3):
            raise ValueError("Strain tensor must be 3x3")
        if not np.allclose(self.components, self.components.T):
            raise ValueError("Strain tensor must be symmetric")

    def volumetric(self) -> float:
        """Compute volumetric strain (trace)."""
        return np.trace(self.components)

    def deviatoric(self) -> np.ndarray:
        """Compute deviatoric strain tensor."""
        return self.components - (self.volumetric() / 3) * np.eye(3)

    def principal_strains(self) -> np.ndarray:
        """Compute principal strains (sorted descending)."""
        eigenvalues = np.linalg.eigvalsh(self.components)
        return np.sort(eigenvalues)[::-1]


@dataclass
class GreenStrainTensor:
    """
    Green-Lagrange strain tensor for large deformations.

    E = (1/2)(FᵀF - I)

    where F is the deformation gradient.
    """

    components: np.ndarray

    def __post_init__(self):
        if self.components.shape != (3, 3):
            raise ValueError("Strain tensor must be 3x3")

    @classmethod
    def from_deformation_gradient(cls, F: np.ndarray) -> "GreenStrainTensor":
        """
        Compute Green strain from deformation gradient.

        Args:
            F: 3x3 deformation gradient tensor

        Returns:
            Green-Lagrange strain tensor
        """
        if F.shape != (3, 3):
            raise ValueError("Deformation gradient must be 3x3")

        C = F.T @ F  # Right Cauchy-Green tensor
        E = 0.5 * (C - np.eye(3))
        return cls(components=E)


class PlaneStress:
    """
    Plane stress condition (σ33 = σ13 = σ23 = 0).

    Used for thin plates loaded in their plane.
    """

    def __init__(self, E: float, nu: float):
        """
        Initialize plane stress condition.

        Args:
            E: Young's modulus
            nu: Poisson's ratio
        """
        if not math.isfinite(E) or E <= 0:
            raise ValueError("Young's modulus must be positive")
        if not (-1 < nu < 0.5):
            raise ValueError("Poisson's ratio must be in (-1, 0.5)")

        self.E = E
        self.nu = nu

        # 3x3 reduced stiffness matrix
        factor = E / (1 - nu**2)
        self.D = np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]]) * factor

    def stress_from_strain(self, strain: np.ndarray) -> np.ndarray:
        """
        Compute in-plane stresses from strains.

        Args:
            strain: [ε11, ε22, γ12]

        Returns:
            [σ11, σ22, σ12]
        """
        if strain.shape != (3,):
            raise ValueError("Expected 3-element strain vector")
        return self.D @ strain

    def out_of_plane_strain(self, strain_11: float, strain_22: float) -> float:
        """
        Compute ε33 from in-plane strains (satisfies σ33 = 0).

        Returns:
            ε33 = -ν/(1-ν) * (ε11 + ε22)
        """
        return -self.nu / (1 - self.nu) * (strain_11 + strain_22)


class PlaneStrain:
    """
    Plane strain condition (ε33 = ε13 = ε23 = 0).

    Used for long prismatic bodies under uniform loading.
    """

    def __init__(self, E: float, nu: float):
        """
        Initialize plane strain condition.

        Args:
            E: Young's modulus
            nu: Poisson's ratio
        """
        if not math.isfinite(E) or E <= 0:
            raise ValueError("Young's modulus must be positive")
        if not (-1 < nu < 0.5):
            raise ValueError("Poisson's ratio must be in (-1, 0.5)")

        self.E = E
        self.nu = nu

        # 3x3 reduced stiffness matrix for plane strain
        factor = E / ((1 + nu) * (1 - 2 * nu))
        self.D = np.array([[1 - nu, nu, 0], [nu, 1 - nu, 0], [0, 0, (1 - 2 * nu) / 2]]) * factor

    def stress_from_strain(self, strain: np.ndarray) -> np.ndarray:
        """
        Compute in-plane stresses from strains.

        Args:
            strain: [ε11, ε22, γ12]

        Returns:
            [σ11, σ22, σ12]
        """
        if strain.shape != (3,):
            raise ValueError("Expected 3-element strain vector")
        return self.D @ strain

    def out_of_plane_stress(self, sigma_11: float, sigma_22: float) -> float:
        """
        Compute σ33 from in-plane stresses (to satisfy ε33 = 0).

        Returns:
            σ33 = ν(σ11 + σ22)
        """
        return self.nu * (sigma_11 + sigma_22)


class ConstitutiveModel:
    """Base class for constitutive material models."""

    def stress(self, strain: np.ndarray) -> np.ndarray:
        """Compute stress from strain."""
        raise NotImplementedError

    def tangent_modulus(self, strain: np.ndarray) -> np.ndarray:
        """Compute tangent stiffness at given strain."""
        raise NotImplementedError


class StrainEnergyDensity:
    """
    Compute strain energy density for elastic materials.

    W = (1/2) σ : ε = (1/2) ε : C : ε
    """

    @staticmethod
    def from_stress_strain(stress: np.ndarray, strain: np.ndarray) -> float:
        """
        Compute strain energy density from stress and strain tensors.

        Args:
            stress: 3x3 stress tensor or 6-element Voigt
            strain: 3x3 strain tensor or 6-element Voigt

        Returns:
            Strain energy density (J/m³)
        """
        if stress.shape == (3, 3) and strain.shape == (3, 3):
            return 0.5 * np.sum(stress * strain)
        elif stress.shape == (6,) and strain.shape == (6,):
            # Account for engineering shear strain factor
            factor = np.array([1, 1, 1, 0.5, 0.5, 0.5])
            return 0.5 * np.sum(stress * strain * factor)
        else:
            raise ValueError("Stress and strain must have matching shapes")

    @staticmethod
    def isotropic(E: float, nu: float, strain: np.ndarray) -> float:
        """
        Compute strain energy density for isotropic material.

        Args:
            E: Young's modulus
            nu: Poisson's ratio
            strain: 3x3 strain tensor

        Returns:
            Strain energy density
        """
        eps_v = np.trace(strain)
        eps_d = strain - (eps_v / 3) * np.eye(3)

        G = E / (2 * (1 + nu))
        K = E / (3 * (1 - 2 * nu))

        return 0.5 * K * eps_v**2 + G * np.sum(eps_d * eps_d)


class PoissonEffect:
    """Calculate Poisson effect (lateral contraction under axial load)."""

    @staticmethod
    def lateral_strain(axial_strain: float, nu: float) -> float:
        """
        Compute lateral strain from axial strain.

        Args:
            axial_strain: Applied axial strain
            nu: Poisson's ratio

        Returns:
            Lateral strain (negative for tensile axial strain)
        """
        return -nu * axial_strain

    @staticmethod
    def volume_change(axial_strain: float, nu: float) -> float:
        """
        Compute relative volume change under uniaxial strain.

        Args:
            axial_strain: Applied axial strain
            nu: Poisson's ratio

        Returns:
            Relative volume change ΔV/V
        """
        return axial_strain * (1 - 2 * nu)


# Convenience functions


def compute_youngs_modulus(K: float, G: float) -> float:
    """
    Compute Young's modulus from bulk and shear moduli.

    E = 9KG / (3K + G)
    """
    if K <= 0 or G <= 0:
        raise ValueError("Moduli must be positive")
    return 9 * K * G / (3 * K + G)


def compute_shear_modulus(E: float, nu: float) -> float:
    """
    Compute shear modulus from Young's modulus and Poisson's ratio.

    G = E / (2(1 + ν))
    """
    if E <= 0:
        raise ValueError("Young's modulus must be positive")
    return E / (2 * (1 + nu))


def compute_bulk_modulus(E: float, nu: float) -> float:
    """
    Compute bulk modulus from Young's modulus and Poisson's ratio.

    K = E / (3(1 - 2ν))
    """
    if E <= 0:
        raise ValueError("Young's modulus must be positive")
    if nu >= 0.5:
        raise ValueError("Poisson's ratio must be less than 0.5 for finite bulk modulus")
    return E / (3 * (1 - 2 * nu))


def compute_lame_constants(E: float, nu: float) -> LameConstants:
    """
    Compute Lamé constants from Young's modulus and Poisson's ratio.

    λ = Eν / ((1+ν)(1-2ν))
    μ = E / (2(1+ν))
    """
    if E <= 0:
        raise ValueError("Young's modulus must be positive")

    lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    return LameConstants(lambda_=lambda_, mu=mu)


def compute_wave_speeds(E: float, nu: float, rho: float) -> Tuple[float, float]:
    """
    Compute P-wave and S-wave speeds in an elastic solid.

    Args:
        E: Young's modulus (Pa)
        nu: Poisson's ratio
        rho: Mass density (kg/m³)

    Returns:
        (c_p, c_s): P-wave and S-wave speeds (m/s)
    """
    if E <= 0 or rho <= 0:
        raise ValueError("Modulus and density must be positive")

    G = E / (2 * (1 + nu))
    M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))

    c_p = math.sqrt(M / rho)
    c_s = math.sqrt(G / rho)

    return (c_p, c_s)
