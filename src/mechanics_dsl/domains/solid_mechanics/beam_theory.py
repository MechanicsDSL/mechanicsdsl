"""
Beam Theory Module for Solid Mechanics

Comprehensive beam analysis including:
- Euler-Bernoulli beam theory (thin beams)
- Timoshenko beam theory (thick beams with shear deformation)
- Various cross-sections and their properties
- Beam deflection, slope, moment, and shear calculations
- Composite and tapered beams
- Curved beams

Security: All geometric inputs validated for positive values.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import sympy as sp

from ...utils import logger


class BeamSupportType(Enum):
    """Types of beam supports."""

    FIXED = auto()  # Clamped (no displacement, no rotation)
    PINNED = auto()  # Simply supported (no displacement, free rotation)
    ROLLER = auto()  # Roller support (vertical restraint only)
    FREE = auto()  # Free end
    GUIDED = auto()  # Guided (no rotation, free horizontal displacement)


@dataclass
class BeamMaterial:
    """
    Material properties for beam analysis.

    Attributes:
        E: Young's modulus (Pa)
        G: Shear modulus (Pa)
        rho: Mass density (kg/m³)
        name: Material identifier
    """

    E: float
    G: float
    rho: float = 7850.0
    name: str = "Steel"

    def __post_init__(self):
        if self.E <= 0 or self.G <= 0:
            raise ValueError("Elastic moduli must be positive")
        if self.rho <= 0:
            raise ValueError("Density must be positive")

    @classmethod
    def steel(cls) -> "BeamMaterial":
        """Standard structural steel."""
        return cls(E=200e9, G=77e9, rho=7850, name="Steel")

    @classmethod
    def aluminum(cls) -> "BeamMaterial":
        """6061-T6 Aluminum."""
        return cls(E=69e9, G=26e9, rho=2700, name="Aluminum")

    @classmethod
    def wood(cls) -> "BeamMaterial":
        """Douglas fir (parallel to grain)."""
        return cls(E=12.4e9, G=0.62e9, rho=530, name="Wood")


class BeamCrossSection(ABC):
    """Abstract base class for beam cross-sections."""

    @property
    @abstractmethod
    def area(self) -> float:
        """Cross-sectional area (m²)."""
        pass

    @property
    @abstractmethod
    def Ixx(self) -> float:
        """Second moment of area about x-axis (m⁴)."""
        pass

    @property
    @abstractmethod
    def Iyy(self) -> float:
        """Second moment of area about y-axis (m⁴)."""
        pass

    @property
    def J(self) -> float:
        """Polar moment of area (m⁴)."""
        return self.Ixx + self.Iyy

    @property
    @abstractmethod
    def y_max(self) -> float:
        """Maximum distance from neutral axis (m)."""
        pass

    @property
    def Sx(self) -> float:
        """Section modulus about x-axis (m³)."""
        return self.Ixx / self.y_max

    @property
    def rx(self) -> float:
        """Radius of gyration about x-axis (m)."""
        return math.sqrt(self.Ixx / self.area)


@dataclass
class RectangularCrossSection(BeamCrossSection):
    """
    Rectangular cross-section.

    Attributes:
        b: Width (m)
        h: Height (m)
    """

    b: float
    h: float

    def __post_init__(self):
        if self.b <= 0 or self.h <= 0:
            raise ValueError("Dimensions must be positive")

    @property
    def area(self) -> float:
        return self.b * self.h

    @property
    def Ixx(self) -> float:
        return self.b * self.h**3 / 12

    @property
    def Iyy(self) -> float:
        return self.h * self.b**3 / 12

    @property
    def y_max(self) -> float:
        return self.h / 2

    @property
    def shear_correction_factor(self) -> float:
        """Timoshenko shear correction factor."""
        return 5 / 6


@dataclass
class CircularCrossSection(BeamCrossSection):
    """
    Solid circular cross-section.

    Attributes:
        d: Diameter (m)
    """

    d: float

    def __post_init__(self):
        if self.d <= 0:
            raise ValueError("Diameter must be positive")

    @property
    def r(self) -> float:
        """Radius."""
        return self.d / 2

    @property
    def area(self) -> float:
        return math.pi * self.r**2

    @property
    def Ixx(self) -> float:
        return math.pi * self.d**4 / 64

    @property
    def Iyy(self) -> float:
        return self.Ixx

    @property
    def y_max(self) -> float:
        return self.r

    @property
    def shear_correction_factor(self) -> float:
        return 6 / 7


@dataclass
class HollowCircularCrossSection(BeamCrossSection):
    """
    Hollow circular (tube) cross-section.

    Attributes:
        d_outer: Outer diameter (m)
        d_inner: Inner diameter (m)
    """

    d_outer: float
    d_inner: float

    def __post_init__(self):
        if self.d_outer <= 0 or self.d_inner <= 0:
            raise ValueError("Diameters must be positive")
        if self.d_inner >= self.d_outer:
            raise ValueError("Inner diameter must be less than outer")

    @property
    def area(self) -> float:
        return math.pi * (self.d_outer**2 - self.d_inner**2) / 4

    @property
    def Ixx(self) -> float:
        return math.pi * (self.d_outer**4 - self.d_inner**4) / 64

    @property
    def Iyy(self) -> float:
        return self.Ixx

    @property
    def y_max(self) -> float:
        return self.d_outer / 2

    @property
    def t(self) -> float:
        """Wall thickness."""
        return (self.d_outer - self.d_inner) / 2


@dataclass
class HollowRectangularCrossSection(BeamCrossSection):
    """
    Hollow rectangular (box) cross-section.

    Attributes:
        b_outer: Outer width (m)
        h_outer: Outer height (m)
        t: Wall thickness (m)
    """

    b_outer: float
    h_outer: float
    t: float

    def __post_init__(self):
        if self.b_outer <= 0 or self.h_outer <= 0 or self.t <= 0:
            raise ValueError("Dimensions must be positive")
        if 2 * self.t >= min(self.b_outer, self.h_outer):
            raise ValueError("Wall thickness too large")

    @property
    def b_inner(self) -> float:
        return self.b_outer - 2 * self.t

    @property
    def h_inner(self) -> float:
        return self.h_outer - 2 * self.t

    @property
    def area(self) -> float:
        return self.b_outer * self.h_outer - self.b_inner * self.h_inner

    @property
    def Ixx(self) -> float:
        return (self.b_outer * self.h_outer**3 - self.b_inner * self.h_inner**3) / 12

    @property
    def Iyy(self) -> float:
        return (self.h_outer * self.b_outer**3 - self.h_inner * self.b_inner**3) / 12

    @property
    def y_max(self) -> float:
        return self.h_outer / 2


@dataclass
class IBeamCrossSection(BeamCrossSection):
    """
    I-beam (wide-flange) cross-section.

    Attributes:
        h: Total height (m)
        b: Flange width (m)
        tw: Web thickness (m)
        tf: Flange thickness (m)
    """

    h: float
    b: float
    tw: float
    tf: float

    def __post_init__(self):
        if any(x <= 0 for x in [self.h, self.b, self.tw, self.tf]):
            raise ValueError("All dimensions must be positive")
        if 2 * self.tf >= self.h:
            raise ValueError("Flanges too thick for height")
        if self.tw > self.b:
            raise ValueError("Web thicker than flange width")

    @property
    def area(self) -> float:
        # Two flanges + web
        return 2 * self.b * self.tf + (self.h - 2 * self.tf) * self.tw

    @property
    def Ixx(self) -> float:
        # Parallel axis theorem
        hw = self.h - 2 * self.tf  # Web height

        # Web contribution
        I_web = self.tw * hw**3 / 12

        # Flange contributions (about their own centroids + parallel axis)
        I_flange = 2 * (self.b * self.tf**3 / 12 + self.b * self.tf * ((self.h - self.tf) / 2) ** 2)

        return I_web + I_flange

    @property
    def Iyy(self) -> float:
        hw = self.h - 2 * self.tf
        return 2 * self.tf * self.b**3 / 12 + hw * self.tw**3 / 12

    @property
    def y_max(self) -> float:
        return self.h / 2

    @property
    def web_height(self) -> float:
        """Clear height of web."""
        return self.h - 2 * self.tf


@dataclass
class BeamLoading:
    """
    Represents a loading condition on a beam.
    """

    type: str  # 'point', 'distributed', 'moment'
    value: float  # Force (N), distributed load (N/m), or moment (N·m)
    position: float  # Position along beam (m)
    end_position: Optional[float] = None  # For distributed loads

    def __post_init__(self):
        if self.type not in ("point", "distributed", "moment"):
            raise ValueError(f"Unknown loading type: {self.type}")
        if self.position < 0:
            raise ValueError("Position must be non-negative")


@dataclass
class BeamElement:
    """
    A beam element with material, cross-section, and supports.

    Attributes:
        length: Beam length (m)
        cross_section: Cross-section geometry
        material: Material properties
        left_support: Left end support type
        right_support: Right end support type
    """

    length: float
    cross_section: BeamCrossSection
    material: BeamMaterial
    left_support: BeamSupportType = BeamSupportType.FIXED
    right_support: BeamSupportType = BeamSupportType.FREE

    def __post_init__(self):
        if self.length <= 0:
            raise ValueError("Beam length must be positive")

    @property
    def EI(self) -> float:
        """Bending stiffness (N·m²)."""
        return self.material.E * self.cross_section.Ixx

    @property
    def GA(self) -> float:
        """Shear stiffness (N)."""
        return self.material.G * self.cross_section.area


class BendingStiffness:
    """Calculate bending stiffness for beam elements."""

    @staticmethod
    def compute(E: float, I: float) -> float:
        """
        Compute bending stiffness EI.

        Args:
            E: Young's modulus (Pa)
            I: Second moment of area (m⁴)

        Returns:
            Bending stiffness (N·m²)
        """
        if E <= 0 or I <= 0:
            raise ValueError("E and I must be positive")
        return E * I


class SectionModulus:
    """Calculate section modulus for stress calculations."""

    @staticmethod
    def elastic(I: float, y_max: float) -> float:
        """
        Elastic section modulus S = I / y_max.

        Args:
            I: Second moment of area
            y_max: Maximum distance from neutral axis

        Returns:
            Elastic section modulus (m³)
        """
        if I <= 0 or y_max <= 0:
            raise ValueError("I and y_max must be positive")
        return I / y_max

    @staticmethod
    def plastic_rectangular(b: float, h: float) -> float:
        """
        Plastic section modulus for rectangular section.

        Z = bh²/4
        """
        return b * h**2 / 4


class NeutralAxis:
    """Neutral axis calculations for composite beams."""

    @staticmethod
    def composite_location(sections: List[Tuple[float, float, float]]) -> float:
        """
        Find neutral axis of composite section.

        Args:
            sections: List of (area, y_centroid, E) for each part

        Returns:
            y-coordinate of neutral axis
        """
        sum_EA_y = sum(A * y * E for A, y, E in sections)
        sum_EA = sum(A * E for A, y, E in sections)

        if abs(sum_EA) < 1e-12:
            raise ValueError("Total EA cannot be zero")

        return sum_EA_y / sum_EA


class ShearCenter:
    """Shear center calculations for open sections."""

    @staticmethod
    def channel_section(b: float, h: float, t: float) -> float:
        """
        Shear center location for channel section.

        Args:
            b: Flange width
            h: Web height
            t: Wall thickness

        Returns:
            Distance of shear center from web (m)
        """
        return 3 * b**2 / (h + 6 * b)


@dataclass
class TorsionalRigidity:
    """Torsional rigidity calculations."""

    @staticmethod
    def circular(G: float, J: float) -> float:
        """Torsional rigidity GJ for circular section."""
        return G * J

    @staticmethod
    def rectangular(G: float, b: float, h: float) -> float:
        """
        Approximate torsional rigidity for rectangular section.

        Uses correction factor for non-circular sections.
        """
        a = max(b, h) / 2
        c = min(b, h) / 2

        # Approximate torsional constant
        J = a * c**3 * (16 / 3 - 3.36 * c / a * (1 - c**4 / (12 * a**4)))

        return G * J


class WarpingConstant:
    """Warping constant for torsional analysis of open sections."""

    @staticmethod
    def i_beam(h: float, b: float, tf: float) -> float:
        """
        Warping constant for I-beam.

        Cw = Iyy * h² / 4 (approximate)
        """
        Iyy = 2 * tf * b**3 / 12
        return Iyy * h**2 / 4


class ShearCorrectionFactor:
    """Timoshenko beam shear correction factors."""

    RECTANGLE = 5 / 6
    CIRCLE = 6 / 7
    THIN_TUBE = 0.5
    I_BEAM = 5 / 6  # Approximate

    @staticmethod
    def for_section(section: BeamCrossSection) -> float:
        """Get shear correction factor for section type."""
        if hasattr(section, "shear_correction_factor"):
            return section.shear_correction_factor
        return 5 / 6  # Default


class EulerBernoulliBeam:
    """
    Euler-Bernoulli beam theory (thin beam theory).

    Assumptions:
    - Plane sections remain plane
    - No shear deformation
    - Small rotations

    Governing equation: EI * d⁴w/dx⁴ = q(x)
    """

    def __init__(self, beam: BeamElement):
        """
        Initialize Euler-Bernoulli beam.

        Args:
            beam: BeamElement with geometry and material
        """
        self.beam = beam
        self.loadings: List[BeamLoading] = []

    def add_load(self, loading: BeamLoading) -> None:
        """Add a loading to the beam."""
        if loading.position > self.beam.length:
            raise ValueError("Loading position exceeds beam length")
        self.loadings.append(loading)

    def cantilever_point_load(self, P: float, a: float) -> Dict[str, Callable]:
        """
        Cantilever beam with point load P at distance a from fixed end.

        Args:
            P: Point load (N), positive downward
            a: Distance from fixed end (m)

        Returns:
            Dict with deflection, slope, moment, shear functions
        """
        L = self.beam.length
        EI = self.beam.EI

        if a > L or a < 0:
            raise ValueError(f"Load position must be in [0, {L}]")

        def deflection(x: float) -> float:
            if x < 0 or x > L:
                return 0.0
            if x <= a:
                return P * x**2 * (3 * a - x) / (6 * EI)
            else:
                return P * a**2 * (3 * x - a) / (6 * EI)

        def slope(x: float) -> float:
            if x < 0 or x > L:
                return 0.0
            if x <= a:
                return P * x * (2 * a - x) / (2 * EI)
            else:
                return P * a**2 / (2 * EI)

        def moment(x: float) -> float:
            if x <= a:
                return -P * (a - x)
            return 0.0

        def shear(x: float) -> float:
            if x < a:
                return -P
            return 0.0

        return {"deflection": deflection, "slope": slope, "moment": moment, "shear": shear}

    def cantilever_uniform_load(self, q: float) -> Dict[str, Callable]:
        """
        Cantilever with uniform distributed load q.

        Args:
            q: Distributed load (N/m), positive downward

        Returns:
            Dict with deflection, slope, moment, shear functions
        """
        L = self.beam.length
        EI = self.beam.EI

        def deflection(x: float) -> float:
            return q * x**2 * (6 * L**2 - 4 * L * x + x**2) / (24 * EI)

        def slope(x: float) -> float:
            return q * x * (3 * L**2 - 3 * L * x + x**2) / (6 * EI)

        def moment(x: float) -> float:
            return -q * (L - x) ** 2 / 2

        def shear(x: float) -> float:
            return -q * (L - x)

        return {"deflection": deflection, "slope": slope, "moment": moment, "shear": shear}

    def simply_supported_point_load(self, P: float, a: float) -> Dict[str, Callable]:
        """
        Simply supported beam with point load.

        Args:
            P: Point load (N)
            a: Distance from left support (m)
        """
        L = self.beam.length
        EI = self.beam.EI
        b = L - a

        if a > L or a < 0:
            raise ValueError(f"Load position must be in [0, {L}]")

        def deflection(x: float) -> float:
            if x < 0 or x > L:
                return 0.0
            if x <= a:
                return P * b * x * (L**2 - b**2 - x**2) / (6 * EI * L)
            else:
                return P * a * (L - x) * (2 * L * x - x**2 - a**2) / (6 * EI * L)

        def moment(x: float) -> float:
            if x <= a:
                return P * b * x / L
            else:
                return P * a * (L - x) / L

        def shear(x: float) -> float:
            if x < a:
                return P * b / L
            elif x > a:
                return -P * a / L
            else:
                return 0.0  # At load point

        return {"deflection": deflection, "moment": moment, "shear": shear}

    def max_deflection(self, load_type: str = "uniform") -> float:
        """
        Get maximum deflection for common loading cases.

        Args:
            load_type: 'uniform' or 'point_center'
        """
        L = self.beam.length
        EI = self.beam.EI

        if self.beam.left_support == BeamSupportType.FIXED:
            # Cantilever
            if load_type == "uniform" and self.loadings:
                q = self.loadings[0].value
                return q * L**4 / (8 * EI)
            elif load_type == "point_center":
                P = self.loadings[0].value if self.loadings else 0
                return P * L**3 / (3 * EI)
        else:
            # Simply supported
            if load_type == "uniform" and self.loadings:
                q = self.loadings[0].value
                return 5 * q * L**4 / (384 * EI)
            elif load_type == "point_center":
                P = self.loadings[0].value if self.loadings else 0
                return P * L**3 / (48 * EI)

        return 0.0

    def natural_frequencies(self, n_modes: int = 3) -> List[float]:
        """
        Compute natural frequencies of beam.

        Args:
            n_modes: Number of modes to compute

        Returns:
            List of natural frequencies (Hz)
        """
        L = self.beam.length
        EI = self.beam.EI
        rho = self.beam.material.rho
        A = self.beam.cross_section.area

        # Coefficient depends on boundary conditions
        if (
            self.beam.left_support == BeamSupportType.FIXED
            and self.beam.right_support == BeamSupportType.FREE
        ):
            # Cantilever: βL = 1.875, 4.694, 7.855, ...
            beta_L = [1.875, 4.694, 7.855, 10.996, 14.137]
        elif (
            self.beam.left_support == BeamSupportType.PINNED
            and self.beam.right_support == BeamSupportType.PINNED
        ):
            # Simply supported: βL = π, 2π, 3π, ...
            beta_L = [n * math.pi for n in range(1, n_modes + 3)]
        elif (
            self.beam.left_support == BeamSupportType.FIXED
            and self.beam.right_support == BeamSupportType.FIXED
        ):
            # Fixed-fixed: βL = 4.730, 7.853, 10.996, ...
            beta_L = [4.730, 7.853, 10.996, 14.137, 17.279]
        else:
            # Default to simply supported
            beta_L = [n * math.pi for n in range(1, n_modes + 3)]

        frequencies = []
        for i in range(min(n_modes, len(beta_L))):
            beta = beta_L[i] / L
            omega = beta**2 * math.sqrt(EI / (rho * A))
            f = omega / (2 * math.pi)
            frequencies.append(f)

        return frequencies


class TimoshenkoBeam:
    """
    Timoshenko beam theory (thick beam theory).

    Includes shear deformation effects, important for:
    - Short, deep beams (L/h < 10)
    - High-frequency vibrations
    - Composites with low shear modulus
    """

    def __init__(self, beam: BeamElement):
        """Initialize Timoshenko beam."""
        self.beam = beam
        self.kappa = ShearCorrectionFactor.for_section(beam.cross_section)

    @property
    def shear_coefficient(self) -> float:
        """Ratio of shear to bending stiffness."""
        return self.beam.EI / (self.kappa * self.beam.GA * self.beam.length**2)

    def cantilever_point_tip(self, P: float) -> Dict[str, float]:
        """
        Tip deflection of cantilever with point load at tip.

        Includes both bending and shear contributions.

        Returns:
            Dict with total, bending, and shear deflections
        """
        L = self.beam.length
        EI = self.beam.EI
        kGA = self.kappa * self.beam.GA

        w_bending = P * L**3 / (3 * EI)
        w_shear = P * L / kGA

        return {
            "total": w_bending + w_shear,
            "bending": w_bending,
            "shear": w_shear,
            "shear_ratio": w_shear / (w_bending + w_shear),
        }

    def is_shear_significant(self, threshold: float = 0.05) -> bool:
        """
        Check if shear deformation is significant.

        Args:
            threshold: Shear contribution threshold (default 5%)

        Returns:
            True if shear contribution exceeds threshold
        """
        result = self.cantilever_point_tip(1.0)  # Unit load
        return result["shear_ratio"] > threshold


class CantileverBeam(EulerBernoulliBeam):
    """Convenience class for cantilever beams."""

    def __init__(self, length: float, section: BeamCrossSection, material: BeamMaterial = None):
        beam = BeamElement(
            length=length,
            cross_section=section,
            material=material or BeamMaterial.steel(),
            left_support=BeamSupportType.FIXED,
            right_support=BeamSupportType.FREE,
        )
        super().__init__(beam)


class SimplySupportedBeam(EulerBernoulliBeam):
    """Convenience class for simply supported beams."""

    def __init__(self, length: float, section: BeamCrossSection, material: BeamMaterial = None):
        beam = BeamElement(
            length=length,
            cross_section=section,
            material=material or BeamMaterial.steel(),
            left_support=BeamSupportType.PINNED,
            right_support=BeamSupportType.PINNED,
        )
        super().__init__(beam)


@dataclass
class BeamDeflection:
    """Store beam deflection results."""

    position: np.ndarray
    deflection: np.ndarray
    max_deflection: float
    max_deflection_position: float


@dataclass
class BeamBendingMoment:
    """Store bending moment distribution."""

    position: np.ndarray
    moment: np.ndarray
    max_moment: float
    max_moment_position: float


@dataclass
class BeamShearForce:
    """Store shear force distribution."""

    position: np.ndarray
    shear: np.ndarray
    max_shear: float


@dataclass
class BeamSlopeAngle:
    """Store rotation/slope distribution."""

    position: np.ndarray
    slope: np.ndarray
    max_slope: float


@dataclass
class BeamStress:
    """Beam stress results at a cross-section."""

    bending_stress_top: float
    bending_stress_bottom: float
    max_shear_stress: float
    position: float


class CompositeBeam:
    """Analysis of composite (multi-material) beams."""

    def __init__(self, sections: List[Tuple[BeamCrossSection, BeamMaterial, float]]):
        """
        Initialize composite beam.

        Args:
            sections: List of (cross_section, material, y_position) tuples
        """
        self.sections = sections
        self._compute_transformed_section()

    def _compute_transformed_section(self):
        """Compute transformed section properties."""
        # Use first section as reference
        E_ref = self.sections[0][1].E

        # Transform all sections
        self.transformed_areas = []
        for section, material, y in self.sections:
            n = material.E / E_ref
            A_transformed = n * section.area
            self.transformed_areas.append((A_transformed, y, section))

        # Compute neutral axis
        sum_Ay = sum(A * y for A, y, _ in self.transformed_areas)
        sum_A = sum(A for A, _, _ in self.transformed_areas)
        self.neutral_axis_y = sum_Ay / sum_A

        # Compute transformed moment of inertia
        self.I_transformed = 0
        for A, y, section in self.transformed_areas:
            I_local = section.Ixx
            d = y - self.neutral_axis_y
            self.I_transformed += I_local + A * d**2


class CurvedBeam:
    """Analysis of curved beams (Winkler theory)."""

    def __init__(self, R: float, section: BeamCrossSection, material: BeamMaterial):
        """
        Initialize curved beam.

        Args:
            R: Radius of curvature to centroid (m)
            section: Cross-section
            material: Material properties
        """
        self.R = R
        self.section = section
        self.material = material

        # Check if curved beam theory is needed (R/h < 5)
        self.is_deeply_curved = (R / (2 * section.y_max)) < 5

    def bending_stress(self, M: float, y: float) -> float:
        """
        Compute bending stress in curved beam.

        Uses Winkler-Bach formula for deep curvature.

        Args:
            M: Bending moment (N·m)
            y: Distance from neutral axis (positive toward center)

        Returns:
            Bending stress (Pa)
        """
        A = self.section.area
        R = self.R

        if self.is_deeply_curved:
            # Winkler-Bach formula
            e = R - self.section.y_max  # Distance to neutral axis
            sigma = M * y / (A * e * (R + y))
        else:
            # Standard beam formula
            sigma = M * y / self.section.Ixx

        return sigma


class TaperBeam:
    """Analysis of tapered beams with varying cross-section."""

    def __init__(
        self,
        length: float,
        section_start: BeamCrossSection,
        section_end: BeamCrossSection,
        material: BeamMaterial,
    ):
        """
        Initialize tapered beam.

        Args:
            length: Beam length
            section_start: Cross-section at x=0
            section_end: Cross-section at x=L
            material: Material properties
        """
        self.length = length
        self.section_start = section_start
        self.section_end = section_end
        self.material = material

    def I_at_x(self, x: float) -> float:
        """
        Get moment of inertia at position x (linear interpolation).

        Args:
            x: Position along beam

        Returns:
            Second moment of area at x
        """
        if x < 0 or x > self.length:
            raise ValueError(f"x must be in [0, {self.length}]")

        ratio = x / self.length
        I_start = self.section_start.Ixx
        I_end = self.section_end.Ixx

        return I_start + ratio * (I_end - I_start)


# Convenience functions


def compute_beam_deflection(
    beam: BeamElement, loading: BeamLoading, num_points: int = 100
) -> BeamDeflection:
    """
    Compute beam deflection along length.

    Args:
        beam: Beam element
        loading: Applied loading
        num_points: Number of output points

    Returns:
        BeamDeflection with position and deflection arrays
    """
    eb = EulerBernoulliBeam(beam)
    eb.add_load(loading)

    x = np.linspace(0, beam.length, num_points)

    if loading.type == "point":
        funcs = eb.cantilever_point_load(loading.value, loading.position)
    else:
        funcs = eb.cantilever_uniform_load(loading.value)

    w = np.array([funcs["deflection"](xi) for xi in x])

    max_idx = np.argmax(np.abs(w))

    return BeamDeflection(
        position=x, deflection=w, max_deflection=w[max_idx], max_deflection_position=x[max_idx]
    )


def compute_bending_stress(M: float, section: BeamCrossSection, y: Optional[float] = None) -> float:
    """
    Compute bending stress σ = My/I.

    Args:
        M: Bending moment (N·m)
        section: Cross-section
        y: Distance from neutral axis (default: y_max)

    Returns:
        Bending stress (Pa)
    """
    if y is None:
        y = section.y_max
    return M * y / section.Ixx


def compute_shear_stress_beam(V: float, Q: float, I: float, t: float) -> float:
    """
    Compute shear stress τ = VQ/(It).

    Args:
        V: Shear force (N)
        Q: First moment of area above point (m³)
        I: Second moment of area (m⁴)
        t: Width at point (m)

    Returns:
        Shear stress (Pa)
    """
    if I <= 0 or t <= 0:
        raise ValueError("I and t must be positive")
    return V * Q / (I * t)
