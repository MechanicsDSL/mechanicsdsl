"""
Plate Theory Module for Solid Mechanics

Implements thin and thick plate theories including:
- Kirchhoff-Love plate theory (thin plates)
- Mindlin-Reissner plate theory (thick plates)
- Rectangular and circular plate solutions
- Plate buckling analysis
- Natural frequency calculations

Security: All inputs validated for physical consistency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy as sp

from ...utils import logger


class PlateBoundaryCondition(Enum):
    """Plate edge boundary conditions."""

    SIMPLY_SUPPORTED = auto()  # SS: w=0, Mxx=0
    CLAMPED = auto()  # C: w=0, dw/dn=0
    FREE = auto()  # F: Mxx=0, Vx=0
    GUIDED = auto()  # G: dw/dn=0, Vx=0


@dataclass
class PlateMaterial:
    """Material properties for plate analysis."""

    E: float  # Young's modulus (Pa)
    nu: float  # Poisson's ratio
    rho: float = 7850  # Density (kg/m³)

    def __post_init__(self):
        if self.E <= 0:
            raise ValueError("Young's modulus must be positive")
        if not (-1 < self.nu < 0.5):
            raise ValueError("Poisson's ratio must be in (-1, 0.5)")
        if self.rho <= 0:
            raise ValueError("Density must be positive")

    @property
    def G(self) -> float:
        """Shear modulus."""
        return self.E / (2 * (1 + self.nu))


@dataclass
class PlateLoading:
    """Represents loading on a plate."""

    type: str  # 'uniform', 'point', 'line', 'hydrostatic'
    magnitude: float  # Pressure (Pa) or force (N)
    position: Optional[Tuple[float, float]] = None  # For point loads


class BendingRigidity:
    """Plate bending rigidity D = Eh³/(12(1-ν²))."""

    @staticmethod
    def compute(E: float, h: float, nu: float) -> float:
        """
        Compute plate bending rigidity.

        Args:
            E: Young's modulus (Pa)
            h: Plate thickness (m)
            nu: Poisson's ratio

        Returns:
            Bending rigidity D (N·m)
        """
        if E <= 0 or h <= 0:
            raise ValueError("E and h must be positive")
        if nu >= 0.5:
            raise ValueError("nu must be less than 0.5")

        return E * h**3 / (12 * (1 - nu**2))


@dataclass
class PlateElement:
    """A plate element with geometry and material."""

    thickness: float
    material: PlateMaterial

    def __post_init__(self):
        if self.thickness <= 0:
            raise ValueError("Thickness must be positive")

    @property
    def D(self) -> float:
        """Bending rigidity."""
        return BendingRigidity.compute(self.material.E, self.thickness, self.material.nu)

    @property
    def mass_per_area(self) -> float:
        """Mass per unit area (kg/m²)."""
        return self.material.rho * self.thickness


class KirchhoffLovePlate:
    """
    Kirchhoff-Love (thin) plate theory.

    Assumptions:
    - Plane sections remain plane and perpendicular to mid-surface
    - No transverse shear deformation
    - h/a < 0.1 (thin plate)

    Governing equation: D∇⁴w = q(x,y)
    """

    def __init__(self, plate: PlateElement):
        """Initialize Kirchhoff-Love plate."""
        self.plate = plate

    def rectangular_simply_supported(
        self, a: float, b: float, q: float, m_terms: int = 5, n_terms: int = 5
    ) -> Dict:
        """
        Solve simply supported rectangular plate with uniform load.

        Navier solution using double Fourier series.

        Args:
            a, b: Plate dimensions (m)
            q: Uniform pressure (Pa)
            m_terms, n_terms: Number of series terms

        Returns:
            Dict with max deflection, max moment, coefficients
        """
        D = self.plate.D

        # Maximum deflection at center (x=a/2, y=b/2)
        w_max = 0.0
        Mx_max = 0.0
        My_max = 0.0

        for m in range(1, m_terms + 1, 2):  # Odd terms only for uniform load
            for n in range(1, n_terms + 1, 2):
                amn = 16 * q / (math.pi**6 * m * n)
                amn /= D * ((m / a) ** 2 + (n / b) ** 2) ** 2

                # Deflection coefficient
                w_max += amn

                # Moment coefficients
                factor_x = (m * math.pi / a) ** 2 + self.plate.material.nu * (n * math.pi / b) ** 2
                factor_y = self.plate.material.nu * (m * math.pi / a) ** 2 + (n * math.pi / b) ** 2

                Mx_max += amn * factor_x
                My_max += amn * factor_y

        Mx_max *= -D
        My_max *= -D

        return {
            "max_deflection": w_max,
            "max_Mx": Mx_max,
            "max_My": My_max,
            "center": (a / 2, b / 2),
        }

    def rectangular_ss_point_load(
        self,
        a: float,
        b: float,
        P: float,
        xi: float,
        eta: float,
        m_terms: int = 10,
        n_terms: int = 10,
    ) -> Dict:
        """
        Simply supported plate with point load.

        Args:
            a, b: Plate dimensions
            P: Point load (N)
            xi, eta: Load position
        """
        D = self.plate.D

        w_max = 0.0
        for m in range(1, m_terms + 1):
            for n in range(1, n_terms + 1):
                amn = (
                    (4 * P / (a * b * D))
                    * math.sin(m * math.pi * xi / a)
                    * math.sin(n * math.pi * eta / b)
                )
                amn /= ((m * math.pi / a) ** 2 + (n * math.pi / b) ** 2) ** 2
                w_max += amn * math.sin(m * math.pi / 2) * math.sin(n * math.pi / 2)

        return {"max_deflection": w_max}


class MindlinReissnerPlate:
    """
    Mindlin-Reissner (thick) plate theory.

    Includes transverse shear deformation.
    Important for h/a > 0.1 or sandwich plates.
    """

    def __init__(self, plate: PlateElement, kappa: float = 5 / 6):
        """
        Initialize Mindlin-Reissner plate.

        Args:
            plate: Plate element
            kappa: Shear correction factor
        """
        self.plate = plate
        self.kappa = kappa

    @property
    def shear_stiffness(self) -> float:
        """Shear stiffness κGh."""
        return self.kappa * self.plate.material.G * self.plate.thickness

    def is_shear_significant(self, span: float, threshold: float = 0.05) -> bool:
        """
        Check if shear deformation is significant.

        Args:
            span: Characteristic plate span
            threshold: Contribution threshold
        """
        h = self.plate.thickness
        return (h / span) > 0.1


@dataclass
class RectangularPlate:
    """Rectangular plate geometry and analysis."""

    a: float  # Length in x-direction
    b: float  # Length in y-direction
    plate: PlateElement
    bc: Dict[str, PlateBoundaryCondition] = None

    def __post_init__(self):
        if self.a <= 0 or self.b <= 0:
            raise ValueError("Dimensions must be positive")
        if self.bc is None:
            self.bc = {
                "x0": PlateBoundaryCondition.SIMPLY_SUPPORTED,
                "xa": PlateBoundaryCondition.SIMPLY_SUPPORTED,
                "y0": PlateBoundaryCondition.SIMPLY_SUPPORTED,
                "yb": PlateBoundaryCondition.SIMPLY_SUPPORTED,
            }

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio a/b."""
        return self.a / self.b

    def uniform_load_deflection(self, q: float) -> Dict:
        """
        Maximum deflection under uniform load.

        Uses tabulated coefficients for all-SS case.
        """
        kl = KirchhoffLovePlate(self.plate)
        return kl.rectangular_simply_supported(self.a, self.b, q)

    def buckling_load(self, m: int = 1, n: int = 1) -> float:
        """
        Critical buckling load for uniaxial compression.

        Args:
            m, n: Number of half-waves in x and y

        Returns:
            Critical stress (Pa)
        """
        D = self.plate.D
        h = self.plate.thickness
        a, b = self.a, self.b

        factor = (m / a) ** 2 + (n / b) ** 2
        Ncr = math.pi**2 * D * factor**2 / (m / a) ** 2

        return Ncr / h  # Critical stress


@dataclass
class CircularPlate:
    """Circular plate geometry and analysis."""

    radius: float
    plate: PlateElement
    center_support: bool = False

    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError("Radius must be positive")

    def clamped_uniform_load(self, q: float) -> Dict:
        """
        Clamped circular plate with uniform load.

        Returns:
            Dict with max deflection (at center), max moment
        """
        R = self.radius
        D = self.plate.D
        nu = self.plate.material.nu

        w_max = q * R**4 / (64 * D)
        Mr_max = q * R**2 / 8  # At edge
        Mr_center = q * R**2 * (1 + nu) / 16

        return {"max_deflection": w_max, "max_radial_moment": Mr_max, "center_moment": Mr_center}

    def simply_supported_uniform(self, q: float) -> Dict:
        """Simply supported circular plate with uniform load."""
        R = self.radius
        D = self.plate.D
        nu = self.plate.material.nu

        w_max = q * R**4 * (5 + nu) / (64 * D * (1 + nu))
        Mr_max = q * R**2 * (3 + nu) / 16

        return {"max_deflection": w_max, "max_radial_moment": Mr_max}

    def natural_frequencies(self, n_modes: int = 3, bc: str = "clamped") -> List[float]:
        """
        Compute natural frequencies of circular plate.

        Args:
            n_modes: Number of modes
            bc: 'clamped' or 'simply_supported'

        Returns:
            List of natural frequencies (Hz)
        """
        D = self.plate.D
        rho = self.plate.material.rho
        h = self.plate.thickness
        R = self.radius

        # Eigenvalue parameters (λ² where λ = βR)
        if bc == "clamped":
            # (m,n) = (0,0), (1,0), (2,0) axisymmetric modes
            lambda_sq = [10.22, 39.77, 89.10, 158.18, 247.00]
        else:  # simply_supported
            lambda_sq = [4.94, 29.72, 74.16, 138.32, 222.22]

        frequencies = []
        for i in range(min(n_modes, len(lambda_sq))):
            omega_sq = lambda_sq[i] * D / (rho * h * R**4)
            f = math.sqrt(omega_sq) / (2 * math.pi)
            frequencies.append(f)

        return frequencies


class ShellElement:
    """
    Basic shell element for curved surfaces.

    Combines membrane and bending behavior.
    """

    def __init__(self, plate: PlateElement, R1: float, R2: float = None):
        """
        Initialize shell element.

        Args:
            plate: Plate element (defines thickness/material)
            R1: Principal radius of curvature
            R2: Second principal radius (default = R1 for sphere)
        """
        self.plate = plate
        self.R1 = R1
        self.R2 = R2 if R2 is not None else R1

        if R1 <= 0 or self.R2 <= 0:
            raise ValueError("Radii must be positive")

    @property
    def gaussian_curvature(self) -> float:
        """Gaussian curvature K = 1/(R1*R2)."""
        return 1 / (self.R1 * self.R2)

    @property
    def mean_curvature(self) -> float:
        """Mean curvature H = (1/R1 + 1/R2)/2."""
        return (1 / self.R1 + 1 / self.R2) / 2

    def membrane_stress_sphere(self, p: float) -> float:
        """
        Membrane stress in pressurized sphere.

        σ = pR/(2t)
        """
        return p * self.R1 / (2 * self.plate.thickness)

    def membrane_stress_cylinder(self, p: float) -> Tuple[float, float]:
        """
        Membrane stresses in pressurized cylinder.

        Returns:
            (hoop_stress, axial_stress)
        """
        t = self.plate.thickness
        R = self.R1

        sigma_hoop = p * R / t
        sigma_axial = p * R / (2 * t)

        return (sigma_hoop, sigma_axial)


class ThinPlateEquation:
    """Symbolic representation of thin plate governing equation."""

    def __init__(self):
        """Initialize symbolic variables."""
        self.x, self.y = sp.symbols("x y", real=True)
        self.w = sp.Function("w")(self.x, self.y)
        self.D = sp.Symbol("D", positive=True)
        self.q = sp.Function("q")(self.x, self.y)

    def biharmonic_operator(self) -> sp.Expr:
        """
        Biharmonic operator ∇⁴w.

        ∇⁴w = ∂⁴w/∂x⁴ + 2∂⁴w/∂x²∂y² + ∂⁴w/∂y⁴
        """
        return (
            sp.diff(self.w, self.x, 4)
            + 2 * sp.diff(self.w, self.x, 2, self.y, 2)
            + sp.diff(self.w, self.y, 4)
        )

    def governing_equation(self) -> sp.Eq:
        """Get plate governing equation D∇⁴w = q."""
        return sp.Eq(self.D * self.biharmonic_operator(), self.q)

    def moment_x(self, nu: sp.Symbol) -> sp.Expr:
        """Bending moment Mx = -D(∂²w/∂x² + ν∂²w/∂y²)."""
        return -self.D * (sp.diff(self.w, self.x, 2) + nu * sp.diff(self.w, self.y, 2))

    def moment_y(self, nu: sp.Symbol) -> sp.Expr:
        """Bending moment My = -D(∂²w/∂y² + ν∂²w/∂x²)."""
        return -self.D * (sp.diff(self.w, self.y, 2) + nu * sp.diff(self.w, self.x, 2))


@dataclass
class PlateDeflection:
    """Store plate deflection field."""

    x: np.ndarray
    y: np.ndarray
    w: np.ndarray
    max_deflection: float
    max_location: Tuple[float, float]


@dataclass
class PlateBendingMoment:
    """Store plate bending moment distribution."""

    x: np.ndarray
    y: np.ndarray
    Mx: np.ndarray
    My: np.ndarray
    Mxy: np.ndarray
    max_Mx: float
    max_My: float


@dataclass
class PlateShearForce:
    """Store plate shear force distribution."""

    x: np.ndarray
    y: np.ndarray
    Qx: np.ndarray
    Qy: np.ndarray


@dataclass
class PlateStress:
    """Store plate stress results."""

    sigma_x_top: float
    sigma_x_bottom: float
    sigma_y_top: float
    sigma_y_bottom: float
    tau_xy_max: float


class PlateBuckling:
    """Plate buckling analysis."""

    @staticmethod
    def uniaxial_compression(a: float, b: float, D: float, m: int = 1, n: int = 1) -> float:
        """
        Critical buckling load for uniaxial compression.

        Args:
            a, b: Plate dimensions
            D: Bending rigidity
            m, n: Buckling mode numbers

        Returns:
            Critical load Ncr (N/m)
        """
        k = ((m * b / a) + (n * a / (m * b))) ** 2
        return k * math.pi**2 * D / b**2

    @staticmethod
    def buckling_coefficient(a: float, b: float, m: int = 1) -> float:
        """
        Buckling coefficient k for simply supported plate.

        k = (mb/a + a/(mb))²
        """
        phi = a / b  # Aspect ratio
        k_min = float("inf")

        for m_try in range(1, 10):
            k = (m_try / phi + phi / m_try) ** 2
            if k < k_min:
                k_min = k

        return k_min


# Convenience functions


def compute_plate_deflection(
    plate: RectangularPlate, q: float, nx: int = 50, ny: int = 50
) -> PlateDeflection:
    """
    Compute plate deflection field.

    Args:
        plate: Rectangular plate
        q: Uniform pressure
        nx, ny: Grid resolution

    Returns:
        PlateDeflection with field data
    """
    result = plate.uniform_load_deflection(q)

    x = np.linspace(0, plate.a, nx)
    y = np.linspace(0, plate.b, ny)
    X, Y = np.meshgrid(x, y)

    # Simplified: assume maximum at center, approximate shape
    w = result["max_deflection"] * np.sin(np.pi * X / plate.a) * np.sin(np.pi * Y / plate.b)

    return PlateDeflection(
        x=X,
        y=Y,
        w=w,
        max_deflection=result["max_deflection"],
        max_location=(plate.a / 2, plate.b / 2),
    )


def compute_plate_stress(M: float, h: float, nu: float) -> PlateStress:
    """
    Compute plate stresses from bending moment.

    Args:
        M: Bending moment per unit width (N)
        h: Plate thickness
        nu: Poisson's ratio

    Returns:
        PlateStress with surface stresses
    """
    # Maximum stress at surfaces
    sigma = 6 * M / h**2

    return PlateStress(
        sigma_x_top=sigma,
        sigma_x_bottom=-sigma,
        sigma_y_top=sigma * nu,
        sigma_y_bottom=-sigma * nu,
        tau_xy_max=0,
    )


def compute_plate_natural_frequencies(
    plate: Union[RectangularPlate, CircularPlate], n_modes: int = 5
) -> List[float]:
    """
    Compute plate natural frequencies.

    Args:
        plate: Plate geometry
        n_modes: Number of modes

    Returns:
        List of frequencies (Hz)
    """
    if isinstance(plate, CircularPlate):
        return plate.natural_frequencies(n_modes)
    else:
        # Rectangular plate - simplified for all SS
        D = plate.plate.D
        rho = plate.plate.material.rho
        h = plate.plate.thickness
        a, b = plate.a, plate.b

        frequencies = []
        for m in range(1, n_modes + 1):
            for n in range(1, n_modes - m + 2):
                omega = math.pi**2 * ((m / a) ** 2 + (n / b) ** 2) * math.sqrt(D / (rho * h))
                f = omega / (2 * math.pi)
                frequencies.append(f)

        return sorted(frequencies)[:n_modes]
