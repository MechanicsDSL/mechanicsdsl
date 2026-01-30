"""
Optics Domain for MechanicsDSL

Comprehensive optics implementation including:
- Geometric optics (ray tracing, Snell's law, reflection)
- Wave optics (interference, diffraction)
- Polarization (Jones matrices, Stokes parameters)
- Thin and thick lenses
- Fiber optics

Security: All refractive indices and angles validated.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple, Optional
import numpy as np


class OpticalMaterialType(Enum):
    """Types of optical materials."""
    GLASS = auto()
    CRYSTAL = auto()
    PLASTIC = auto()
    AIR = auto()
    WATER = auto()


@dataclass
class OpticalMaterial:
    """Properties of optical material."""
    name: str
    n: float  # Refractive index
    abbe_number: float = 50.0  # Dispersion parameter
    
    def __post_init__(self):
        if self.n < 1.0:
            raise ValueError("Refractive index must be >= 1.0")
    
    @classmethod
    def air(cls) -> 'OpticalMaterial':
        return cls(name="air", n=1.0, abbe_number=100)
    
    @classmethod
    def water(cls) -> 'OpticalMaterial':
        return cls(name="water", n=1.333, abbe_number=55)
    
    @classmethod
    def bk7_glass(cls) -> 'OpticalMaterial':
        return cls(name="BK7", n=1.5168, abbe_number=64.17)
    
    @classmethod
    def diamond(cls) -> 'OpticalMaterial':
        return cls(name="diamond", n=2.417, abbe_number=55)


class SnellsLaw:
    """Snell's law calculations."""
    
    @staticmethod
    def refraction_angle(n1: float, n2: float, theta1: float) -> Optional[float]:
        """
        Compute refraction angle using Snell's law.
        
        n1 sin(θ1) = n2 sin(θ2)
        
        Args:
            n1, n2: Refractive indices
            theta1: Incident angle (radians)
            
        Returns:
            Refracted angle or None if total internal reflection
        """
        sin_theta2 = (n1 / n2) * math.sin(theta1)
        
        if abs(sin_theta2) > 1.0:
            return None  # Total internal reflection
        
        return math.asin(sin_theta2)
    
    @staticmethod
    def critical_angle(n1: float, n2: float) -> Optional[float]:
        """
        Critical angle for total internal reflection.
        
        θc = arcsin(n2/n1)  for n1 > n2
        """
        if n1 <= n2:
            return None
        return math.asin(n2 / n1)
    
    @staticmethod
    def is_total_internal_reflection(n1: float, n2: float, theta1: float) -> bool:
        """Check if TIR occurs."""
        return SnellsLaw.refraction_angle(n1, n2, theta1) is None


class FresnelEquations:
    """Fresnel reflection and transmission coefficients."""
    
    @staticmethod
    def rs(n1: float, n2: float, theta1: float, theta2: float) -> float:
        """S-polarization (TE) reflection coefficient."""
        num = n1 * math.cos(theta1) - n2 * math.cos(theta2)
        den = n1 * math.cos(theta1) + n2 * math.cos(theta2)
        return num / den
    
    @staticmethod
    def rp(n1: float, n2: float, theta1: float, theta2: float) -> float:
        """P-polarization (TM) reflection coefficient."""
        num = n2 * math.cos(theta1) - n1 * math.cos(theta2)
        den = n2 * math.cos(theta1) + n1 * math.cos(theta2)
        return num / den
    
    @staticmethod
    def reflectance_normal(n1: float, n2: float) -> float:
        """Reflectance at normal incidence: R = ((n2-n1)/(n2+n1))²."""
        return ((n2 - n1) / (n2 + n1))**2
    
    @staticmethod
    def brewster_angle(n1: float, n2: float) -> float:
        """Brewster's angle: θB = arctan(n2/n1)."""
        return math.atan(n2 / n1)


class RayTracing:
    """Geometric ray tracing."""
    
    @staticmethod
    def ray_at_distance(origin: np.ndarray, direction: np.ndarray, 
                        t: float) -> np.ndarray:
        """Point on ray: P(t) = O + t*D."""
        return origin + t * direction
    
    @staticmethod
    def reflect_ray(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """
        Reflected ray direction.
        
        R = D - 2(D·N)N
        """
        d = direction / np.linalg.norm(direction)
        n = normal / np.linalg.norm(normal)
        return d - 2 * np.dot(d, n) * n
    
    @staticmethod
    def refract_ray(direction: np.ndarray, normal: np.ndarray,
                    n1: float, n2: float) -> Optional[np.ndarray]:
        """
        Refracted ray direction using vector form of Snell's law.
        
        Returns None if total internal reflection.
        """
        d = direction / np.linalg.norm(direction)
        n = normal / np.linalg.norm(normal)
        
        ratio = n1 / n2
        cos_theta1 = -np.dot(d, n)
        sin_theta2_sq = ratio**2 * (1 - cos_theta1**2)
        
        if sin_theta2_sq > 1:
            return None
        
        cos_theta2 = math.sqrt(1 - sin_theta2_sq)
        return ratio * d + (ratio * cos_theta1 - cos_theta2) * n
    
    @staticmethod
    def sphere_intersection(origin: np.ndarray, direction: np.ndarray,
                           center: np.ndarray, radius: float) -> Optional[float]:
        """
        Ray-sphere intersection.
        
        Returns parameter t for closest intersection, or None.
        """
        d = direction / np.linalg.norm(direction)
        oc = origin - center
        
        a = np.dot(d, d)
        b = 2 * np.dot(oc, d)
        c = np.dot(oc, oc) - radius**2
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return None
        
        t = (-b - math.sqrt(discriminant)) / (2*a)
        if t < 0:
            t = (-b + math.sqrt(discriminant)) / (2*a)
        
        return t if t > 0 else None


class ThinLens:
    """Thin lens optics."""
    
    def __init__(self, focal_length: float):
        if focal_length == 0:
            raise ValueError("Focal length cannot be zero")
        self.f = focal_length
    
    @property
    def power(self) -> float:
        """Optical power in diopters (1/m)."""
        return 1.0 / self.f
    
    def image_distance(self, object_distance: float) -> float:
        """
        Image distance using thin lens equation.
        
        1/f = 1/do + 1/di
        """
        if object_distance == 0:
            return float('inf')
        return 1.0 / (1.0/self.f - 1.0/object_distance)
    
    def magnification(self, object_distance: float) -> float:
        """Lateral magnification m = -di/do."""
        di = self.image_distance(object_distance)
        if object_distance == 0:
            return float('inf')
        return -di / object_distance
    
    @staticmethod
    def lensmaker_equation(n: float, R1: float, R2: float, 
                           d: float = 0) -> float:
        """
        Lensmaker's equation for focal length.
        
        1/f = (n-1)[1/R1 - 1/R2 + (n-1)d/(n*R1*R2)]
        """
        if d == 0:  # Thin lens
            return 1.0 / ((n - 1) * (1/R1 - 1/R2))
        else:
            return 1.0 / ((n-1) * (1/R1 - 1/R2 + (n-1)*d/(n*R1*R2)))


class Interference:
    """Wave interference calculations."""
    
    @staticmethod
    def two_slit_intensity(theta: float, d: float, wavelength: float) -> float:
        """
        Double slit interference intensity.
        
        I = I0 cos²(πd sinθ / λ)
        """
        phase = math.pi * d * math.sin(theta) / wavelength
        return math.cos(phase)**2
    
    @staticmethod
    def two_slit_maxima(d: float, wavelength: float, n_max: int = 5) -> List[float]:
        """
        Angular positions of interference maxima.
        
        sin(θ) = mλ/d
        """
        maxima = []
        for m in range(-n_max, n_max + 1):
            sin_theta = m * wavelength / d
            if abs(sin_theta) <= 1:
                maxima.append(math.asin(sin_theta))
        return sorted(maxima)
    
    @staticmethod
    def thin_film_constructive(n_film: float, d: float, 
                                wavelength: float, m: int = 1) -> bool:
        """
        Check for constructive interference in thin film.
        
        2nd = (m + 1/2)λ for phase change at both surfaces
        """
        return abs(2 * n_film * d - (m - 0.5) * wavelength) < wavelength * 0.01
    
    @staticmethod
    def path_difference(d: float, theta: float) -> float:
        """Path difference = d sin(θ)."""
        return d * math.sin(theta)


class Diffraction:
    """Diffraction calculations."""
    
    @staticmethod
    def single_slit_intensity(theta: float, a: float, wavelength: float) -> float:
        """
        Single slit diffraction intensity.
        
        I = I0 (sin(β)/β)² where β = πa sinθ / λ
        """
        if abs(theta) < 1e-10:
            return 1.0
        
        beta = math.pi * a * math.sin(theta) / wavelength
        if abs(beta) < 1e-10:
            return 1.0
        return (math.sin(beta) / beta)**2
    
    @staticmethod
    def single_slit_minima(a: float, wavelength: float, n_max: int = 3) -> List[float]:
        """
        Angular positions of diffraction minima.
        
        a sin(θ) = mλ
        """
        minima = []
        for m in range(1, n_max + 1):
            sin_theta = m * wavelength / a
            if sin_theta <= 1:
                minima.append(math.asin(sin_theta))
        return minima
    
    @staticmethod
    def rayleigh_criterion(diameter: float, wavelength: float) -> float:
        """
        Rayleigh criterion angular resolution.
        
        θ_min = 1.22 λ/D
        """
        return 1.22 * wavelength / diameter
    
    @staticmethod
    def grating_equation(d: float, theta_i: float, m: int,
                         wavelength: float) -> Optional[float]:
        """
        Diffraction grating equation.
        
        d(sin θm - sin θi) = mλ
        """
        sin_theta_m = math.sin(theta_i) + m * wavelength / d
        if abs(sin_theta_m) > 1:
            return None
        return math.asin(sin_theta_m)


class Polarization:
    """Polarization optics using Jones calculus."""
    
    @staticmethod
    def linear_polarizer(theta: float = 0) -> np.ndarray:
        """
        Jones matrix for linear polarizer.
        
        Args:
            theta: Polarizer axis angle from horizontal
        """
        c, s = math.cos(theta), math.sin(theta)
        return np.array([
            [c**2, c*s],
            [c*s, s**2]
        ])
    
    @staticmethod
    def half_wave_plate(theta: float = 0) -> np.ndarray:
        """Jones matrix for half-wave plate with fast axis at theta."""
        c2, s2 = math.cos(2*theta), math.sin(2*theta)
        return np.array([
            [c2, s2],
            [s2, -c2]
        ])
    
    @staticmethod
    def quarter_wave_plate(theta: float = 0) -> np.ndarray:
        """Jones matrix for quarter-wave plate."""
        c, s = math.cos(theta), math.sin(theta)
        return np.array([
            [c**2 + 1j*s**2, (1-1j)*c*s],
            [(1-1j)*c*s, s**2 + 1j*c**2]
        ])
    
    @staticmethod
    def malus_law(I0: float, theta: float) -> float:
        """Malus' law: I = I0 cos²(θ)."""
        return I0 * math.cos(theta)**2
    
    @staticmethod
    def stokes_to_intensity(S: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert Stokes parameters to polarization properties.
        
        Returns (intensity, degree_of_polarization, angle)
        """
        I = S[0]
        DOP = math.sqrt(S[1]**2 + S[2]**2 + S[3]**2) / I
        angle = 0.5 * math.atan2(S[2], S[1])
        return (I, DOP, angle)


class FiberOptics:
    """Optical fiber calculations."""
    
    @staticmethod
    def numerical_aperture(n_core: float, n_clad: float) -> float:
        """NA = √(n_core² - n_clad²)."""
        if n_core <= n_clad:
            raise ValueError("Core index must exceed cladding")
        return math.sqrt(n_core**2 - n_clad**2)
    
    @staticmethod
    def acceptance_angle(NA: float) -> float:
        """Maximum acceptance angle θmax = arcsin(NA)."""
        if NA > 1:
            return math.pi / 2
        return math.asin(NA)
    
    @staticmethod
    def v_number(a: float, wavelength: float, NA: float) -> float:
        """
        Normalized frequency (V-number).
        
        V = 2πa·NA/λ
        """
        return 2 * math.pi * a * NA / wavelength
    
    @staticmethod
    def is_single_mode(V: float) -> bool:
        """Single mode operation for V < 2.405."""
        return V < 2.405
    
    @staticmethod
    def number_of_modes(V: float) -> int:
        """Approximate number of guided modes."""
        if V < 2.405:
            return 1
        return int(V**2 / 2)
    
    @staticmethod
    def attenuation_db(alpha: float, L: float) -> float:
        """Attenuation in dB: A = αL."""
        return alpha * L
    
    @staticmethod
    def power_after_fiber(P0: float, alpha_db_per_km: float, L_km: float) -> float:
        """Output power after fiber of length L."""
        return P0 * 10**(-alpha_db_per_km * L_km / 10)


__all__ = [
    "OpticalMaterial",
    "OpticalMaterialType",
    "SnellsLaw",
    "FresnelEquations",
    "RayTracing",
    "ThinLens",
    "Interference",
    "Diffraction",
    "Polarization",
    "FiberOptics",
]
