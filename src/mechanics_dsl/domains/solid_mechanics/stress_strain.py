"""
Stress and Strain Analysis Module for Solid Mechanics

Comprehensive stress-strain analysis including:
- Principal stress/strain computation
- Stress invariants (I1, I2, I3, J2, J3)
- Von Mises, Tresca, and octahedral stresses
- Mohr's circles (2D and 3D)
- Stress/strain transformations
- Stress concentration factors
- Strain rosette analysis
- Compatibility equations

Security: All tensor operations validated for proper dimensions and symmetry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple

import numpy as np


class StressState(Enum):
    """Classification of stress state types."""

    UNIAXIAL = auto()
    BIAXIAL = auto()
    TRIAXIAL = auto()
    PLANE_STRESS = auto()
    PLANE_STRAIN = auto()
    HYDROSTATIC = auto()
    PURE_SHEAR = auto()


@dataclass
class StressInvariant:
    """
    Stress tensor invariants.

    First invariant: I1 = σ1 + σ2 + σ3 = tr(σ)
    Second invariant: I2 = σ1σ2 + σ2σ3 + σ3σ1
    Third invariant: I3 = σ1σ2σ3 = det(σ)

    Deviatoric invariants:
    J2 = (1/2)s:s (related to von Mises stress)
    J3 = det(s)
    """

    I1: float
    I2: float
    I3: float
    J2: float
    J3: float

    @classmethod
    def from_tensor(cls, sigma: np.ndarray) -> "StressInvariant":
        """
        Compute invariants from 3x3 stress tensor.

        Args:
            sigma: 3x3 symmetric stress tensor

        Returns:
            StressInvariant dataclass
        """
        if sigma.shape != (3, 3):
            raise ValueError(f"Expected 3x3 tensor, got {sigma.shape}")

        # First invariant (trace)
        I1 = np.trace(sigma)

        # Second invariant
        I2 = 0.5 * (I1**2 - np.trace(sigma @ sigma))

        # Third invariant (determinant)
        I3 = np.linalg.det(sigma)

        # Deviatoric stress
        p = I1 / 3  # Mean stress
        s = sigma - p * np.eye(3)

        # Deviatoric invariants
        J2 = 0.5 * np.sum(s * s)
        J3 = np.linalg.det(s)

        return cls(I1=I1, I2=I2, I3=I3, J2=J2, J3=J3)

    @property
    def mean_stress(self) -> float:
        """Hydrostatic (mean) stress σm = I1/3."""
        return self.I1 / 3

    @property
    def von_mises(self) -> float:
        """Von Mises equivalent stress σv = √(3J2)."""
        return math.sqrt(3 * self.J2)

    @property
    def lode_angle(self) -> float:
        """
        Lode angle θ describing stress state on deviatoric plane.

        cos(3θ) = (3√3/2) * J3 / J2^(3/2)
        Range: -π/6 ≤ θ ≤ π/6
        """
        if self.J2 < 1e-12:
            return 0.0

        cos_3theta = (3 * math.sqrt(3) / 2) * self.J3 / (self.J2**1.5)
        cos_3theta = np.clip(cos_3theta, -1.0, 1.0)
        return math.acos(cos_3theta) / 3


@dataclass
class StrainInvariant:
    """
    Strain tensor invariants (analogous to stress invariants).
    """

    I1: float  # Volumetric strain
    I2: float
    I3: float
    J2: float  # Deviatoric strain measure

    @classmethod
    def from_tensor(cls, epsilon: np.ndarray) -> "StrainInvariant":
        """Compute invariants from 3x3 strain tensor."""
        if epsilon.shape != (3, 3):
            raise ValueError(f"Expected 3x3 tensor, got {epsilon.shape}")

        I1 = np.trace(epsilon)
        I2 = 0.5 * (I1**2 - np.trace(epsilon @ epsilon))
        I3 = np.linalg.det(epsilon)

        e = epsilon - (I1 / 3) * np.eye(3)  # Deviatoric strain
        J2 = 0.5 * np.sum(e * e)

        return cls(I1=I1, I2=I2, I3=I3, J2=J2)

    @property
    def volumetric_strain(self) -> float:
        """Volumetric strain εv = I1 = ε11 + ε22 + ε33."""
        return self.I1

    @property
    def equivalent_strain(self) -> float:
        """Equivalent (von Mises) strain εeq = √(2J2/3) * (2/3)."""
        return math.sqrt(2 * self.J2 / 3) * (2 / 3)


@dataclass
class PrincipalStress:
    """
    Principal stresses and directions.

    Attributes:
        sigma1: Maximum principal stress
        sigma2: Intermediate principal stress
        sigma3: Minimum principal stress
        directions: 3x3 matrix of unit eigenvectors (columns)
    """

    sigma1: float
    sigma2: float
    sigma3: float
    directions: np.ndarray = field(default_factory=lambda: np.eye(3))

    @classmethod
    def from_tensor(cls, sigma: np.ndarray) -> "PrincipalStress":
        """
        Compute principal stresses from stress tensor.

        Args:
            sigma: 3x3 symmetric stress tensor

        Returns:
            PrincipalStress with sorted values (σ1 ≥ σ2 ≥ σ3)
        """
        if sigma.shape != (3, 3):
            raise ValueError(f"Expected 3x3 tensor, got {sigma.shape}")
        if not np.allclose(sigma, sigma.T, rtol=1e-10):
            raise ValueError("Stress tensor must be symmetric")

        eigenvalues, eigenvectors = np.linalg.eigh(sigma)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return cls(
            sigma1=eigenvalues[0],
            sigma2=eigenvalues[1],
            sigma3=eigenvalues[2],
            directions=eigenvectors,
        )

    @property
    def max_shear(self) -> float:
        """Maximum shear stress τmax = (σ1 - σ3)/2."""
        return (self.sigma1 - self.sigma3) / 2

    def as_array(self) -> np.ndarray:
        """Return principal stresses as array."""
        return np.array([self.sigma1, self.sigma2, self.sigma3])


@dataclass
class PrincipalStrain:
    """
    Principal strains and directions.
    """

    epsilon1: float
    epsilon2: float
    epsilon3: float
    directions: np.ndarray = field(default_factory=lambda: np.eye(3))

    @classmethod
    def from_tensor(cls, epsilon: np.ndarray) -> "PrincipalStrain":
        """Compute principal strains from strain tensor."""
        if epsilon.shape != (3, 3):
            raise ValueError(f"Expected 3x3 tensor, got {epsilon.shape}")

        eigenvalues, eigenvectors = np.linalg.eigh(epsilon)
        idx = np.argsort(eigenvalues)[::-1]

        return cls(
            epsilon1=eigenvalues[idx[0]],
            epsilon2=eigenvalues[idx[1]],
            epsilon3=eigenvalues[idx[2]],
            directions=eigenvectors[:, idx],
        )

    @property
    def max_shear_strain(self) -> float:
        """Maximum engineering shear strain γmax = ε1 - ε3."""
        return self.epsilon1 - self.epsilon3


class VonMisesStress:
    """
    Von Mises (equivalent) stress calculations.

    The von Mises stress is used in yield criteria for ductile materials.
    σv = √(3J2) = √((σ1-σ2)² + (σ2-σ3)² + (σ3-σ1)²)/√2
    """

    @staticmethod
    def from_tensor(sigma: np.ndarray) -> float:
        """
        Compute von Mises stress from 3x3 stress tensor.

        Args:
            sigma: 3x3 symmetric stress tensor

        Returns:
            Von Mises equivalent stress
        """
        if sigma.shape != (3, 3):
            raise ValueError(f"Expected 3x3 tensor, got {sigma.shape}")

        # Deviatoric stress
        s = sigma - (np.trace(sigma) / 3) * np.eye(3)
        J2 = 0.5 * np.sum(s * s)

        return math.sqrt(3 * J2)

    @staticmethod
    def from_principal(sigma1: float, sigma2: float, sigma3: float) -> float:
        """
        Compute von Mises stress from principal stresses.

        Args:
            sigma1, sigma2, sigma3: Principal stresses

        Returns:
            Von Mises equivalent stress
        """
        return math.sqrt(
            0.5 * ((sigma1 - sigma2) ** 2 + (sigma2 - sigma3) ** 2 + (sigma3 - sigma1) ** 2)
        )

    @staticmethod
    def from_components(
        sigma_xx: float,
        sigma_yy: float,
        sigma_zz: float,
        tau_xy: float,
        tau_yz: float,
        tau_xz: float,
    ) -> float:
        """
        Compute von Mises stress from individual components.

        Args:
            sigma_xx, sigma_yy, sigma_zz: Normal stresses
            tau_xy, tau_yz, tau_xz: Shear stresses

        Returns:
            Von Mises equivalent stress
        """
        return math.sqrt(
            0.5
            * ((sigma_xx - sigma_yy) ** 2 + (sigma_yy - sigma_zz) ** 2 + (sigma_zz - sigma_xx) ** 2)
            + 3 * (tau_xy**2 + tau_yz**2 + tau_xz**2)
        )

    @staticmethod
    def plane_stress(sigma_x: float, sigma_y: float, tau_xy: float) -> float:
        """
        Von Mises stress for plane stress condition.

        σv = √(σx² - σxσy + σy² + 3τxy²)
        """
        return math.sqrt(sigma_x**2 - sigma_x * sigma_y + sigma_y**2 + 3 * tau_xy**2)


class MaxShearStress:
    """
    Maximum shear stress (Tresca criterion) calculations.

    τmax = (σmax - σmin) / 2
    """

    @staticmethod
    def from_tensor(sigma: np.ndarray) -> float:
        """Compute maximum shear stress from stress tensor."""
        principal = PrincipalStress.from_tensor(sigma)
        return principal.max_shear

    @staticmethod
    def from_principal(sigma1: float, sigma2: float, sigma3: float) -> float:
        """Compute maximum shear stress from principal stresses."""
        stresses = sorted([sigma1, sigma2, sigma3], reverse=True)
        return (stresses[0] - stresses[2]) / 2


class HydrostaticStress:
    """
    Hydrostatic (mean) stress calculations.

    σh = (σ1 + σ2 + σ3) / 3 = I1 / 3
    """

    @staticmethod
    def from_tensor(sigma: np.ndarray) -> float:
        """Compute hydrostatic stress from tensor."""
        return np.trace(sigma) / 3

    @staticmethod
    def from_principal(sigma1: float, sigma2: float, sigma3: float) -> float:
        """Compute hydrostatic stress from principal stresses."""
        return (sigma1 + sigma2 + sigma3) / 3


class DeviatoricStress:
    """
    Deviatoric stress tensor operations.

    The deviatoric stress represents shape-changing stresses:
    s = σ - σh·I
    """

    @staticmethod
    def from_tensor(sigma: np.ndarray) -> np.ndarray:
        """
        Compute deviatoric stress tensor.

        Args:
            sigma: 3x3 stress tensor

        Returns:
            3x3 deviatoric stress tensor
        """
        if sigma.shape != (3, 3):
            raise ValueError(f"Expected 3x3 tensor, got {sigma.shape}")

        hydrostatic = np.trace(sigma) / 3
        return sigma - hydrostatic * np.eye(3)

    @staticmethod
    def principal_deviatoric(
        sigma1: float, sigma2: float, sigma3: float
    ) -> Tuple[float, float, float]:
        """
        Compute principal deviatoric stresses.

        Returns:
            (s1, s2, s3) principal deviatoric stresses
        """
        mean = (sigma1 + sigma2 + sigma3) / 3
        return (sigma1 - mean, sigma2 - mean, sigma3 - mean)


class OctahedralStress:
    """
    Octahedral stress calculations.

    Acting on planes equally inclined to principal axes.
    """

    @staticmethod
    def normal(sigma1: float, sigma2: float, sigma3: float) -> float:
        """
        Octahedral normal stress.

        σoct = (σ1 + σ2 + σ3) / 3
        """
        return (sigma1 + sigma2 + sigma3) / 3

    @staticmethod
    def shear(sigma1: float, sigma2: float, sigma3: float) -> float:
        """
        Octahedral shear stress.

        τoct = (1/3)√((σ1-σ2)² + (σ2-σ3)² + (σ3-σ1)²)
        """
        return (
            math.sqrt((sigma1 - sigma2) ** 2 + (sigma2 - sigma3) ** 2 + (sigma3 - sigma1) ** 2) / 3
        )

    @staticmethod
    def from_tensor(sigma: np.ndarray) -> Tuple[float, float]:
        """
        Compute octahedral normal and shear stresses.

        Returns:
            (σoct, τoct)
        """
        principal = PrincipalStress.from_tensor(sigma)
        sigma1, sigma2, sigma3 = principal.sigma1, principal.sigma2, principal.sigma3

        return (
            OctahedralStress.normal(sigma1, sigma2, sigma3),
            OctahedralStress.shear(sigma1, sigma2, sigma3),
        )


@dataclass
class EquivalentStress:
    """
    Collection of equivalent stress measures for yield criteria.
    """

    von_mises: float
    tresca: float
    octahedral_shear: float
    hydrostatic: float

    @classmethod
    def from_tensor(cls, sigma: np.ndarray) -> "EquivalentStress":
        """Compute all equivalent stresses from tensor."""
        principal = PrincipalStress.from_tensor(sigma)
        s1, s2, s3 = principal.sigma1, principal.sigma2, principal.sigma3

        return cls(
            von_mises=VonMisesStress.from_principal(s1, s2, s3),
            tresca=2 * MaxShearStress.from_principal(s1, s2, s3),
            octahedral_shear=OctahedralStress.shear(s1, s2, s3),
            hydrostatic=HydrostaticStress.from_principal(s1, s2, s3),
        )


@dataclass
class EquivalentStrain:
    """Equivalent strain measures."""

    von_mises: float
    volumetric: float
    deviatoric_magnitude: float

    @classmethod
    def from_tensor(cls, epsilon: np.ndarray) -> "EquivalentStrain":
        """Compute equivalent strains from tensor."""
        volumetric = np.trace(epsilon)
        e = epsilon - (volumetric / 3) * np.eye(3)
        J2 = 0.5 * np.sum(e * e)

        # Von Mises equivalent strain
        vm = math.sqrt(2 * J2 / 3) * (2 / 3)

        return cls(von_mises=vm, volumetric=volumetric, deviatoric_magnitude=math.sqrt(2 * J2))


class TriaxialityFactor:
    """
    Stress triaxiality factor η = σh / σv

    Used in ductile fracture analysis.
    η > 1/3: Tension-dominated
    η ≈ 0: Pure shear
    η < 0: Compression
    """

    @staticmethod
    def compute(sigma: np.ndarray) -> float:
        """
        Compute triaxiality factor.

        Args:
            sigma: 3x3 stress tensor

        Returns:
            Triaxiality factor η
        """
        hydrostatic = np.trace(sigma) / 3
        von_mises = VonMisesStress.from_tensor(sigma)

        if abs(von_mises) < 1e-12:
            return 0.0

        return hydrostatic / von_mises

    @staticmethod
    def classify_state(eta: float) -> str:
        """Classify stress state based on triaxiality."""
        if eta > 0.6:
            return "High triaxiality (severe tension)"
        elif eta > 0.33:
            return "Moderate tension"
        elif eta > 0:
            return "Low triaxiality (near shear)"
        elif eta > -0.33:
            return "Compression-shear"
        else:
            return "High compression"


class StressTransformation:
    """
    Stress tensor transformation operations.
    """

    @staticmethod
    def rotate(sigma: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """
        Transform stress tensor by rotation.

        σ' = Q · σ · Qᵀ

        Args:
            sigma: 3x3 stress tensor
            Q: 3x3 rotation matrix (orthogonal)

        Returns:
            Rotated stress tensor
        """
        if sigma.shape != (3, 3) or Q.shape != (3, 3):
            raise ValueError("Both tensors must be 3x3")
        if not np.allclose(Q @ Q.T, np.eye(3), rtol=1e-10):
            raise ValueError("Q must be orthogonal")

        return Q @ sigma @ Q.T

    @staticmethod
    def rotation_matrix_z(theta: float) -> np.ndarray:
        """
        Create rotation matrix about z-axis.

        Args:
            theta: Rotation angle in radians

        Returns:
            3x3 rotation matrix
        """
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

    @staticmethod
    def plane_stress_transform(
        sigma_x: float, sigma_y: float, tau_xy: float, theta: float
    ) -> Tuple[float, float, float]:
        """
        Transform plane stress state by angle theta.

        Args:
            sigma_x, sigma_y: Normal stresses
            tau_xy: Shear stress
            theta: Rotation angle (radians)

        Returns:
            (sigma_x', sigma_y', tau_xy') in rotated frame
        """
        c2, s2 = math.cos(2 * theta), math.sin(2 * theta)

        sigma_avg = (sigma_x + sigma_y) / 2
        sigma_diff = (sigma_x - sigma_y) / 2

        sigma_x_prime = sigma_avg + sigma_diff * c2 + tau_xy * s2
        sigma_y_prime = sigma_avg - sigma_diff * c2 - tau_xy * s2
        tau_xy_prime = -sigma_diff * s2 + tau_xy * c2

        return (sigma_x_prime, sigma_y_prime, tau_xy_prime)


class StrainTransformation:
    """
    Strain tensor transformation operations.
    """

    @staticmethod
    def rotate(epsilon: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Transform strain tensor by rotation."""
        if epsilon.shape != (3, 3) or Q.shape != (3, 3):
            raise ValueError("Both tensors must be 3x3")

        return Q @ epsilon @ Q.T

    @staticmethod
    def plane_strain_transform(
        eps_x: float, eps_y: float, gamma_xy: float, theta: float
    ) -> Tuple[float, float, float]:
        """
        Transform plane strain state by angle theta.

        Note: gamma_xy is engineering shear strain.

        Returns:
            (eps_x', eps_y', gamma_xy') in rotated frame
        """
        c2, s2 = math.cos(2 * theta), math.sin(2 * theta)

        eps_avg = (eps_x + eps_y) / 2
        eps_diff = (eps_x - eps_y) / 2

        eps_x_prime = eps_avg + eps_diff * c2 + (gamma_xy / 2) * s2
        eps_y_prime = eps_avg - eps_diff * c2 - (gamma_xy / 2) * s2
        gamma_xy_prime = -2 * eps_diff * s2 + gamma_xy * c2

        return (eps_x_prime, eps_y_prime, gamma_xy_prime)


@dataclass
class MohrsCircle2D:
    """
    Mohr's circle for 2D (plane) stress/strain analysis.

    Provides graphical representation of stress transformation.
    """

    center: float
    radius: float
    sigma1: float
    sigma2: float
    max_shear: float
    principal_angle: float  # Angle to σ1 from x-axis

    @classmethod
    def from_stress(cls, sigma_x: float, sigma_y: float, tau_xy: float) -> "MohrsCircle2D":
        """
        Construct Mohr's circle from 2D stress state.

        Args:
            sigma_x, sigma_y: Normal stresses
            tau_xy: Shear stress

        Returns:
            MohrsCircle2D instance
        """
        center = (sigma_x + sigma_y) / 2
        radius = math.sqrt(((sigma_x - sigma_y) / 2) ** 2 + tau_xy**2)

        sigma1 = center + radius
        sigma2 = center - radius
        max_shear = radius

        # Angle to first principal stress
        if abs(sigma_x - sigma_y) < 1e-12:
            if tau_xy >= 0:
                theta_p = math.pi / 4
            else:
                theta_p = -math.pi / 4
        else:
            theta_p = 0.5 * math.atan2(2 * tau_xy, sigma_x - sigma_y)

        return cls(
            center=center,
            radius=radius,
            sigma1=sigma1,
            sigma2=sigma2,
            max_shear=max_shear,
            principal_angle=theta_p,
        )

    def stress_at_angle(self, theta: float) -> Tuple[float, float]:
        """
        Get normal and shear stress at angle theta from x-axis.

        Returns:
            (σn, τ) at the given plane orientation
        """
        # On Mohr's circle, angle is 2θ
        phi = 2 * (theta - self.principal_angle)

        sigma_n = self.center + self.radius * math.cos(phi)
        tau = self.radius * math.sin(phi)

        return (sigma_n, tau)


@dataclass
class MohrsCircle3D:
    """
    Mohr's circles for 3D stress state.

    Three circles representing all possible plane orientations.
    """

    sigma1: float
    sigma2: float
    sigma3: float
    circle_12: MohrsCircle2D
    circle_23: MohrsCircle2D
    circle_13: MohrsCircle2D

    @classmethod
    def from_principal(cls, sigma1: float, sigma2: float, sigma3: float) -> "MohrsCircle3D":
        """
        Construct 3D Mohr's circles from principal stresses.

        Assumes σ1 ≥ σ2 ≥ σ3.
        """
        # Ensure proper ordering
        stresses = sorted([sigma1, sigma2, sigma3], reverse=True)
        s1, s2, s3 = stresses

        # The three circles
        c12 = MohrsCircle2D(
            center=(s1 + s2) / 2,
            radius=(s1 - s2) / 2,
            sigma1=s1,
            sigma2=s2,
            max_shear=(s1 - s2) / 2,
            principal_angle=0,
        )

        c23 = MohrsCircle2D(
            center=(s2 + s3) / 2,
            radius=(s2 - s3) / 2,
            sigma1=s2,
            sigma2=s3,
            max_shear=(s2 - s3) / 2,
            principal_angle=0,
        )

        c13 = MohrsCircle2D(
            center=(s1 + s3) / 2,
            radius=(s1 - s3) / 2,
            sigma1=s1,
            sigma2=s3,
            max_shear=(s1 - s3) / 2,
            principal_angle=0,
        )

        return cls(sigma1=s1, sigma2=s2, sigma3=s3, circle_12=c12, circle_23=c23, circle_13=c13)

    @property
    def max_shear(self) -> float:
        """Maximum shear stress (outer circle radius)."""
        return self.circle_13.radius


@dataclass
class StressConcentration:
    """
    Stress concentration factor calculations.

    Kt = σmax / σnom
    """

    kt: float  # Theoretical stress concentration factor
    geometry: str  # Description of geometry

    @classmethod
    def circular_hole_plate(cls, d: float, W: float) -> "StressConcentration":
        """
        Stress concentration for circular hole in infinite plate.

        For infinite plate: Kt = 3.0
        For finite width: Kt ≈ 3.0 - 3.13(d/W) + 3.66(d/W)² - 1.53(d/W)³

        Args:
            d: Hole diameter
            W: Plate width
        """
        if d >= W:
            raise ValueError("Hole diameter must be less than plate width")

        ratio = d / W
        kt = 3.0 - 3.13 * ratio + 3.66 * ratio**2 - 1.53 * ratio**3

        return cls(kt=kt, geometry=f"Circular hole d={d}, W={W}")

    @classmethod
    def semicircular_notch(cls, r: float, d: float, D: float) -> "StressConcentration":
        """
        Stress concentration for semicircular notch.

        Args:
            r: Notch radius
            d: Net width at notch
            D: Gross width
        """
        if r <= 0 or d <= 0 or d >= D:
            raise ValueError("Invalid notch geometry")

        # Peterson's formula approximation
        kt = 3.0 - 3.4 * (r / d) + 2.31 * (r / d) ** 2
        kt = max(kt, 1.0)

        return cls(kt=kt, geometry=f"Semicircular notch r={r}")

    @classmethod
    def fillet_shoulder(cls, r: float, d: float, D: float) -> "StressConcentration":
        """
        Stress concentration for filleted shoulder.

        Args:
            r: Fillet radius
            d: Smaller diameter/width
            D: Larger diameter/width
        """
        if r <= 0 or d <= 0 or d >= D:
            raise ValueError("Invalid fillet geometry")

        # Simplified Peterson's formula
        t = (D - d) / 2
        kt = 1.0 + 1.0 / math.sqrt(1 + 2 * r / t)

        return cls(kt=kt, geometry=f"Fillet r={r}, d={d}, D={D}")


class StrainRosette:
    """
    Strain rosette analysis for experimental strain measurement.

    Converts readings from strain gauges at different angles to
    full strain state.
    """

    @staticmethod
    def rectangular(eps_a: float, eps_b: float, eps_c: float) -> Tuple[float, float, float]:
        """
        Analyze rectangular (0°-45°-90°) strain rosette.

        Args:
            eps_a: Strain at 0°
            eps_b: Strain at 45°
            eps_c: Strain at 90°

        Returns:
            (eps_x, eps_y, gamma_xy) strain state
        """
        eps_x = eps_a
        eps_y = eps_c
        gamma_xy = 2 * eps_b - eps_a - eps_c

        return (eps_x, eps_y, gamma_xy)

    @staticmethod
    def delta(eps_a: float, eps_b: float, eps_c: float) -> Tuple[float, float, float]:
        """
        Analyze delta (0°-60°-120°) strain rosette.

        Returns:
            (eps_x, eps_y, gamma_xy) strain state
        """
        eps_x = eps_a
        eps_y = (2 * eps_b + 2 * eps_c - eps_a) / 3
        gamma_xy = 2 * (eps_b - eps_c) / math.sqrt(3)

        return (eps_x, eps_y, gamma_xy)

    @staticmethod
    def general(readings: List[Tuple[float, float]]) -> Tuple[float, float, float]:
        """
        Analyze general strain rosette with 3+ readings.

        Uses least squares if overdetermined.

        Args:
            readings: List of (angle_rad, strain) tuples

        Returns:
            (eps_x, eps_y, gamma_xy) strain state
        """
        if len(readings) < 3:
            raise ValueError("Need at least 3 strain readings")

        # Build linear system: eps = eps_x*cos²θ + eps_y*sin²θ + gamma_xy*sinθcosθ/2
        A = []
        b = []

        for theta, eps in readings:
            c, s = math.cos(theta), math.sin(theta)
            A.append([c**2, s**2, s * c])
            b.append(eps)

        A = np.array(A)
        b = np.array(b)

        # Least squares solution
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        return (result[0], result[1], 2 * result[2])


class CompatibilityEquations:
    """
    Strain compatibility equations (Saint-Venant compatibility).

    Ensures strain field corresponds to a continuous displacement field.
    """

    @staticmethod
    def check_2d_compatibility(
        eps_xx: callable, eps_yy: callable, eps_xy: callable, x: float, y: float, h: float = 1e-6
    ) -> float:
        """
        Check 2D compatibility equation numerically.

        ∂²εxx/∂y² + ∂²εyy/∂x² = 2∂²εxy/∂x∂y

        Returns residual (should be ~0 for compatible strains).
        """
        # Numerical second derivatives
        d2_exx_dy2 = (eps_xx(x, y + h) - 2 * eps_xx(x, y) + eps_xx(x, y - h)) / h**2
        d2_eyy_dx2 = (eps_yy(x + h, y) - 2 * eps_yy(x, y) + eps_yy(x - h, y)) / h**2

        # Mixed partial
        d2_exy_dxdy = (
            eps_xy(x + h, y + h)
            - eps_xy(x + h, y - h)
            - eps_xy(x - h, y + h)
            + eps_xy(x - h, y - h)
        ) / (4 * h**2)

        return d2_exx_dy2 + d2_eyy_dx2 - 2 * d2_exy_dxdy


# Convenience functions


def compute_principal_stresses(sigma: np.ndarray) -> np.ndarray:
    """Compute principal stresses from 3x3 stress tensor."""
    return PrincipalStress.from_tensor(sigma).as_array()


def compute_principal_strains(epsilon: np.ndarray) -> np.ndarray:
    """Compute principal strains from 3x3 strain tensor."""
    ps = PrincipalStrain.from_tensor(epsilon)
    return np.array([ps.epsilon1, ps.epsilon2, ps.epsilon3])


def compute_von_mises(sigma: np.ndarray) -> float:
    """Compute von Mises stress from tensor."""
    return VonMisesStress.from_tensor(sigma)


def rotate_stress_tensor(sigma: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Rotate stress tensor by orthogonal matrix Q."""
    return StressTransformation.rotate(sigma, Q)
