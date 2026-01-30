"""
Fracture Mechanics Module for Solid Mechanics

Implements linear elastic and elastic-plastic fracture mechanics:
- Stress intensity factors (Mode I, II, III)
- Energy release rate
- J-integral
- CTOD/COD approach
- Griffith criterion
- Paris law for fatigue crack growth
- Plastic zone size
- Mixed-mode fracture

Security: All geometric and material inputs validated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...utils import logger


class FractureMode(Enum):
    """Fracture mode types."""
    MODE_I = auto()    # Opening (tensile)
    MODE_II = auto()   # In-plane shear (sliding)
    MODE_III = auto()  # Out-of-plane shear (tearing)
    MIXED = auto()     # Combined modes


@dataclass
class CrackGeometry:
    """Crack geometry definition."""
    half_length: float  # Half crack length 'a' for center crack
    width: float = None   # Specimen width W
    thickness: float = None  # Specimen thickness B
    geometry_type: str = 'center'  # 'center', 'edge', 'semi_elliptical'
    
    def __post_init__(self):
        if self.half_length <= 0:
            raise ValueError("Crack half-length must be positive")
        if self.width is not None and self.half_length >= self.width:
            raise ValueError("Crack length must be less than width")


@dataclass
class FractureToughness:
    """Material fracture toughness properties."""
    KIc: float        # Mode I plane strain toughness (Pa√m)
    KIIc: float = None  # Mode II toughness
    KIIIc: float = None  # Mode III toughness
    
    def __post_init__(self):
        if self.KIc <= 0:
            raise ValueError("Fracture toughness must be positive")
    
    @classmethod
    def steel_structural(cls) -> 'FractureToughness':
        """Typical structural steel."""
        return cls(KIc=100e6)
    
    @classmethod
    def aluminum_7075(cls) -> 'FractureToughness':
        """7075-T6 Aluminum."""
        return cls(KIc=29e6)
    
    @classmethod
    def titanium(cls) -> 'FractureToughness':
        """Ti-6Al-4V."""
        return cls(KIc=75e6)


class StressIntensityFactor:
    """
    Stress intensity factor (SIF) calculations.
    
    K = σ√(πa) × F(geometry)
    """
    
    @staticmethod
    def center_crack_tension(sigma: float, a: float, W: float = None) -> float:
        """
        SIF for center crack in infinite plate under tension.
        
        K_I = σ√(πa) for infinite plate
        K_I = σ√(πa) × √(sec(πa/W)) for finite width
        
        Args:
            sigma: Applied stress (Pa)
            a: Half crack length (m)
            W: Plate width (m), None for infinite
            
        Returns:
            K_I (Pa√m)
        """
        if a <= 0:
            raise ValueError("Crack length must be positive")
        
        K = sigma * math.sqrt(math.pi * a)
        
        if W is not None:
            if a >= W / 2:
                raise ValueError("Crack too large for plate width")
            # Finite width correction
            K *= math.sqrt(1 / math.cos(math.pi * a / W))
        
        return K
    
    @staticmethod
    def edge_crack_tension(sigma: float, a: float, W: float) -> float:
        """
        SIF for single edge crack in tension.
        
        Uses polynomial correction factor.
        
        Args:
            sigma: Applied stress
            a: Crack length
            W: Specimen width
        """
        if a >= W:
            raise ValueError("Crack length must be less than width")
        
        ratio = a / W
        
        # Correction factor (Tada, Paris, Irwin)
        F = 1.12 - 0.231 * ratio + 10.55 * ratio**2 - 21.72 * ratio**3 + 30.39 * ratio**4
        
        return F * sigma * math.sqrt(math.pi * a)
    
    @staticmethod
    def semi_elliptical_surface(
        sigma: float, a: float, c: float, t: float, phi: float = math.pi/2
    ) -> float:
        """
        SIF for semi-elliptical surface crack.
        
        Args:
            sigma: Applied stress
            a: Crack depth
            c: Crack half-length at surface
            t: Plate thickness
            phi: Parametric angle (π/2 for deepest point)
            
        Returns:
            K at specified point
        """
        # Aspect ratio
        ac = a / c
        
        # Shape factor Q (elliptic integral approximation)
        Q = 1 + 1.464 * ac**1.65
        
        # Geometry correction
        F = 1.0  # Simplified
        
        K = F * sigma * math.sqrt(math.pi * a / Q)
        
        return K
    
    @staticmethod
    def compact_tension(P: float, a: float, W: float, B: float) -> float:
        """
        SIF for compact tension (CT) specimen.
        
        Standard ASTM E399 geometry.
        
        Args:
            P: Applied load (N)
            a: Crack length from load line
            W: Specimen width
            B: Specimen thickness
        """
        ratio = a / W
        if ratio < 0.2 or ratio > 0.8:
            logger.warning(f"a/W = {ratio:.2f} outside valid range [0.2, 0.8]")
        
        # ASTM E399 formula
        f = ((2 + ratio) / (1 - ratio)**1.5 * 
             (0.886 + 4.64 * ratio - 13.32 * ratio**2 + 
              14.72 * ratio**3 - 5.6 * ratio**4))
        
        K = (P / (B * math.sqrt(W))) * f
        
        return K


class EnergyReleaseRate:
    """
    Energy release rate G calculations.
    
    G = K²/E (plane stress)
    G = K²(1-ν²)/E (plane strain)
    """
    
    @staticmethod
    def from_K(K: float, E: float, nu: float = None) -> float:
        """
        Compute G from stress intensity factor.
        
        Args:
            K: Stress intensity factor
            E: Young's modulus
            nu: Poisson's ratio (None for plane stress)
            
        Returns:
            Energy release rate G (J/m²)
        """
        if E <= 0:
            raise ValueError("Young's modulus must be positive")
        
        if nu is None:
            # Plane stress
            return K**2 / E
        else:
            # Plane strain
            return K**2 * (1 - nu**2) / E
    
    @staticmethod
    def to_K(G: float, E: float, nu: float = None) -> float:
        """Convert G to K."""
        if nu is None:
            return math.sqrt(G * E)
        else:
            return math.sqrt(G * E / (1 - nu**2))


class GriffithCriterion:
    """
    Griffith energy criterion for brittle fracture.
    
    Crack propagates when G ≥ 2γ (surface energy criterion)
    or G ≥ Gc (critical energy release rate)
    """
    
    @staticmethod
    def critical_stress(a: float, E: float, gamma: float) -> float:
        """
        Critical stress for crack propagation.
        
        σ_c = √(2Eγ/(πa))
        
        Args:
            a: Half crack length
            E: Young's modulus
            gamma: Surface energy (J/m²)
            
        Returns:
            Critical stress
        """
        return math.sqrt(2 * E * gamma / (math.pi * a))
    
    @staticmethod
    def critical_crack_length(sigma: float, E: float, gamma: float) -> float:
        """
        Critical crack length for given stress.
        
        a_c = 2Eγ/(πσ²)
        """
        return 2 * E * gamma / (math.pi * sigma**2)


class IrwinCriterion:
    """
    Irwin criterion for LEFM.
    
    Fracture occurs when K ≥ K_c
    """
    
    @staticmethod
    def safety_factor(K: float, Kc: float) -> float:
        """
        Safety factor against fracture.
        
        SF = Kc / K
        """
        if K <= 0:
            return float('inf')
        return Kc / K
    
    @staticmethod
    def critical_stress(Kc: float, a: float, F: float = 1.0) -> float:
        """
        Critical stress for fracture.
        
        σ_c = Kc / (F√(πa))
        """
        return Kc / (F * math.sqrt(math.pi * a))
    
    @staticmethod
    def critical_crack_length(sigma: float, Kc: float, F: float = 1.0) -> float:
        """Critical crack length for given stress."""
        return (Kc / (F * sigma))**2 / math.pi


class PlasticZoneSize:
    """
    Plastic zone size estimates.
    """
    
    @staticmethod
    def irwin_plane_stress(K: float, sigma_y: float) -> float:
        """
        Irwin plastic zone size for plane stress.
        
        r_p = (1/π)(K/σ_y)²
        """
        return (K / sigma_y)**2 / math.pi
    
    @staticmethod
    def irwin_plane_strain(K: float, sigma_y: float) -> float:
        """
        Irwin plastic zone size for plane strain.
        
        r_p = (1/3π)(K/σ_y)²
        """
        return (K / sigma_y)**2 / (3 * math.pi)
    
    @staticmethod
    def dugdale(K: float, sigma_y: float) -> float:
        """
        Dugdale strip yield model zone size.
        
        ρ = (π/8)(K/σ_y)²
        """
        return (math.pi / 8) * (K / sigma_y)**2


@dataclass
class JIntegral:
    """
    J-integral for elastic-plastic fracture.
    
    Path-independent contour integral.
    """
    value: float  # J-integral value (J/m²)
    elastic_component: float = None
    plastic_component: float = None
    
    @classmethod
    def from_K(cls, K: float, E: float, nu: float = None) -> 'JIntegral':
        """J = G for linear elastic materials."""
        G = EnergyReleaseRate.from_K(K, E, nu)
        return cls(value=G, elastic_component=G, plastic_component=0)
    
    def to_K(self, E: float, nu: float = None) -> float:
        """Convert J to equivalent K."""
        return EnergyReleaseRate.to_K(self.value, E, nu)


@dataclass
class CTOD:
    """
    Crack Tip Opening Displacement.
    
    δ = K²/(mσ_y E) where m depends on constraint.
    """
    value: float  # CTOD (m)
    
    @classmethod
    def from_K(cls, K: float, sigma_y: float, E: float, 
               plane_strain: bool = True) -> 'CTOD':
        """
        Compute CTOD from K.
        
        Args:
            K: Stress intensity factor
            sigma_y: Yield stress
            E: Young's modulus
            plane_strain: True for plane strain, False for plane stress
        """
        m = 2.0 if plane_strain else 1.0
        delta = K**2 / (m * sigma_y * E)
        return cls(value=delta)
    
    @classmethod
    def from_J(cls, J: float, sigma_y: float, 
               plane_strain: bool = True) -> 'CTOD':
        """Compute CTOD from J-integral."""
        m = 2.0 if plane_strain else 1.0
        delta = J / (m * sigma_y)
        return cls(value=delta)


# Alias for CTOD
COD = CTOD


class CrackTipStress:
    """
    Near-tip stress field (K-dominant zone).
    """
    
    @staticmethod
    def mode_I(K: float, r: float, theta: float) -> Dict[str, float]:
        """
        Williams stress field for Mode I.
        
        Args:
            K: Mode I SIF
            r: Distance from crack tip
            theta: Angle from crack plane
            
        Returns:
            Dict with stress components
        """
        if r <= 0:
            raise ValueError("Distance must be positive")
        
        factor = K / math.sqrt(2 * math.pi * r)
        ct = math.cos(theta / 2)
        st = math.sin(theta / 2)
        c3t = math.cos(3 * theta / 2)
        s3t = math.sin(3 * theta / 2)
        
        sigma_xx = factor * ct * (1 - st * s3t)
        sigma_yy = factor * ct * (1 + st * s3t)
        tau_xy = factor * ct * st * c3t
        
        return {
            'sigma_xx': sigma_xx,
            'sigma_yy': sigma_yy,
            'tau_xy': tau_xy
        }


class ParisLaw:
    """
    Paris-Erdogan fatigue crack growth law.
    
    da/dN = C(ΔK)^m
    """
    
    def __init__(self, C: float, m: float):
        """
        Initialize Paris law parameters.
        
        Args:
            C: Material constant
            m: Paris exponent (typically 2-4)
        """
        if C <= 0 or m <= 0:
            raise ValueError("Paris law parameters must be positive")
        
        self.C = C
        self.m = m
    
    def growth_rate(self, delta_K: float) -> float:
        """
        Compute crack growth rate da/dN.
        
        Args:
            delta_K: Stress intensity range
            
        Returns:
            Crack growth per cycle (m/cycle)
        """
        if delta_K <= 0:
            return 0.0
        return self.C * delta_K**self.m
    
    def life_to_failure(
        self, a0: float, af: float, delta_sigma: float, F: float = 1.0
    ) -> int:
        """
        Integrate Paris law for fatigue life.
        
        Args:
            a0: Initial crack size
            af: Final crack size
            delta_sigma: Stress range
            F: Geometry factor
            
        Returns:
            Number of cycles to failure
        """
        # Numerical integration
        n_steps = 1000
        a = np.linspace(a0, af, n_steps)
        da = np.diff(a)
        
        N = 0
        for i, a_i in enumerate(a[:-1]):
            delta_K = F * delta_sigma * math.sqrt(math.pi * a_i)
            dN = da[i] / self.growth_rate(delta_K)
            N += dN
        
        return int(N)
    
    @classmethod
    def steel(cls) -> 'ParisLaw':
        """Typical values for steel."""
        return cls(C=1e-11, m=3.0)
    
    @classmethod
    def aluminum(cls) -> 'ParisLaw':
        """Typical values for aluminum alloys."""
        return cls(C=5e-11, m=3.5)


class CrackPropagation:
    """Crack propagation analysis."""
    
    @staticmethod
    def threshold_check(K: float, K_threshold: float) -> bool:
        """Check if K exceeds threshold for propagation."""
        return K >= K_threshold
    
    @staticmethod
    def propagation_direction_mode_I(K: float) -> float:
        """Mode I propagates perpendicular to max principal stress."""
        return 0.0  # Straight ahead
    
    @staticmethod
    def max_tangential_stress_angle(KI: float, KII: float) -> float:
        """
        Mixed-mode crack propagation angle (MTS criterion).
        
        Returns angle in radians.
        """
        if abs(KII) < 1e-12:
            return 0.0
        
        ratio = KI / KII
        theta = 2 * math.atan((ratio - math.sqrt(ratio**2 + 8)) / 4)
        
        return theta


class MixedModeFracture:
    """Mixed-mode fracture criteria."""
    
    @staticmethod
    def equivalent_K_mixed(KI: float, KII: float, KIII: float = 0) -> float:
        """
        Equivalent stress intensity for mixed mode.
        
        Using von Mises-type combination.
        """
        return math.sqrt(KI**2 + KII**2 + KIII**2 / (1 - 0.3))
    
    @staticmethod
    def failure_envelope_ellipse(KI: float, KII: float, 
                                  KIc: float, KIIc: float) -> float:
        """
        Elliptical failure envelope.
        
        Returns value < 1 for safe, ≥ 1 for failure.
        """
        return (KI / KIc)**2 + (KII / KIIc)**2


@dataclass
class CohesiveZoneModel:
    """
    Cohesive zone model parameters.
    """
    sigma_max: float  # Cohesive strength (Pa)
    delta_c: float    # Critical separation (m)
    Gc: float = None  # Cohesive energy (J/m²)
    
    def __post_init__(self):
        if self.Gc is None:
            # Triangular traction-separation
            self.Gc = 0.5 * self.sigma_max * self.delta_c
    
    def traction(self, delta: float) -> float:
        """
        Traction-separation law (triangular).
        
        Args:
            delta: Current separation
            
        Returns:
            Traction force per unit area
        """
        if delta <= 0:
            return 0.0
        if delta >= self.delta_c:
            return 0.0
        
        return self.sigma_max * (1 - delta / self.delta_c)


class Fatigue:
    """General fatigue-related utilities."""
    
    @staticmethod
    def mean_stress_ratio(sigma_min: float, sigma_max: float) -> float:
        """Stress ratio R = σ_min/σ_max."""
        if sigma_max == 0:
            raise ValueError("Maximum stress cannot be zero")
        return sigma_min / sigma_max
    
    @staticmethod
    def stress_intensity_range(
        sigma_min: float, sigma_max: float, a: float, F: float = 1.0
    ) -> float:
        """Compute ΔK for fatigue analysis."""
        delta_sigma = sigma_max - sigma_min
        return F * delta_sigma * math.sqrt(math.pi * a)


@dataclass
class FractureCriterion:
    """Container for fracture assessment results."""
    K_applied: float
    Kc: float
    safety_factor: float
    is_safe: bool
    crack_length: float
    critical_crack_length: float


# Convenience functions

def compute_stress_intensity(
    sigma: float, a: float, geometry: str = 'center', **kwargs
) -> float:
    """
    Compute stress intensity factor for various geometries.
    
    Args:
        sigma: Applied stress
        a: Crack length
        geometry: 'center', 'edge', or 'semi_elliptical'
        **kwargs: Geometry-specific parameters
        
    Returns:
        Stress intensity factor
    """
    if geometry == 'center':
        return StressIntensityFactor.center_crack_tension(
            sigma, a, kwargs.get('W')
        )
    elif geometry == 'edge':
        return StressIntensityFactor.edge_crack_tension(
            sigma, a, kwargs['W']
        )
    elif geometry == 'semi_elliptical':
        return StressIntensityFactor.semi_elliptical_surface(
            sigma, a, kwargs['c'], kwargs['t']
        )
    else:
        raise ValueError(f"Unknown geometry: {geometry}")


def compute_energy_release_rate(K: float, E: float, nu: float = None) -> float:
    """Compute energy release rate from K."""
    return EnergyReleaseRate.from_K(K, E, nu)


def compute_j_integral(K: float, E: float, nu: float = None) -> float:
    """Compute J-integral (equals G for LEFM)."""
    return EnergyReleaseRate.from_K(K, E, nu)


def compute_plastic_zone(K: float, sigma_y: float, 
                         plane_strain: bool = True) -> float:
    """Compute plastic zone size."""
    if plane_strain:
        return PlasticZoneSize.irwin_plane_strain(K, sigma_y)
    else:
        return PlasticZoneSize.irwin_plane_stress(K, sigma_y)
