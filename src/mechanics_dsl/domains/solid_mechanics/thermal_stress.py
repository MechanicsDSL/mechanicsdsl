"""
Thermal Stress Module for Solid Mechanics

Thermal expansion, thermal stresses, and thermoelastic coupling.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class ThermalExpansion:
    """Thermal expansion coefficient."""
    alpha: float  # Linear CTE (1/K)
    
    def linear_strain(self, delta_T: float) -> float:
        """ε_thermal = α·ΔT."""
        return self.alpha * delta_T
    
    def volumetric_strain(self, delta_T: float) -> float:
        """3α·ΔT for isotropic material."""
        return 3 * self.alpha * delta_T
    
    @classmethod
    def steel(cls) -> 'ThermalExpansion':
        return cls(alpha=12e-6)
    
    @classmethod
    def aluminum(cls) -> 'ThermalExpansion':
        return cls(alpha=23e-6)


@dataclass
class ThermalStrain:
    """Thermal strain tensor."""
    components: np.ndarray
    
    @classmethod
    def isotropic(cls, alpha: float, delta_T: float) -> 'ThermalStrain':
        strain = alpha * delta_T * np.eye(3)
        return cls(components=strain)


class ThermalStress:
    """Thermal stress calculations."""
    
    @staticmethod
    def constrained_bar(E: float, alpha: float, delta_T: float) -> float:
        """Stress in fully constrained bar: σ = -E·α·ΔT."""
        return -E * alpha * delta_T
    
    @staticmethod
    def biaxial_plate(E: float, nu: float, alpha: float, delta_T: float) -> float:
        """Biaxial thermal stress in constrained plate."""
        return -E * alpha * delta_T / (1 - nu)
    
    @staticmethod
    def thick_cylinder(E: float, nu: float, alpha: float, 
                       T_inner: float, T_outer: float,
                       r_inner: float, r_outer: float, r: float) -> Tuple[float, float]:
        """
        Thermal stresses in thick-walled cylinder.
        
        Returns (sigma_r, sigma_theta) at radius r.
        """
        k = r_outer / r_inner
        dT = T_inner - T_outer
        
        factor = E * alpha * dT / (2 * (1 - nu) * math.log(k))
        
        sigma_r = factor * (-math.log(r_outer/r) - r_inner**2/(r**2 - r_inner**2) * 
                           (1 - r_outer**2/r**2) * math.log(k))
        sigma_theta = factor * (1 - math.log(r_outer/r) - r_inner**2/(r**2 - r_inner**2) * 
                               (1 + r_outer**2/r**2) * math.log(k))
        
        return (sigma_r, sigma_theta)


@dataclass
class ThermalDeformation:
    """Thermal deformation results."""
    delta_length: float
    delta_volume: float
    strain: float


class ThermalGradient:
    """Temperature gradient effects."""
    
    @staticmethod
    def linear_gradient_stress(E: float, alpha: float, 
                               dT_dy: float, y: float, h: float) -> float:
        """
        Stress from linear temperature gradient through thickness.
        
        σ = E·α·(T_avg - T(y))
        """
        T_deviation = dT_dy * (y - h/2)
        return -E * alpha * T_deviation


@dataclass
class ThermalMismatch:
    """Thermal mismatch stress between materials."""
    alpha1: float
    alpha2: float
    E1: float
    E2: float
    
    def interface_stress(self, delta_T: float, A1: float, A2: float) -> Tuple[float, float]:
        """
        Stresses at interface of bonded dissimilar materials.
        
        Returns (stress1, stress2).
        """
        d_alpha = self.alpha1 - self.alpha2
        denom = 1/(self.E1 * A1) + 1/(self.E2 * A2)
        
        F = d_alpha * delta_T / denom
        
        sigma1 = -F / A1
        sigma2 = F / A2
        
        return (sigma1, sigma2)


class ThermalShock:
    """Thermal shock resistance."""
    
    @staticmethod
    def first_parameter(sigma_f: float, nu: float, 
                        E: float, alpha: float) -> float:
        """First thermal shock parameter R = σf(1-ν)/(Eα)."""
        return sigma_f * (1 - nu) / (E * alpha)
    
    @staticmethod
    def second_parameter(k: float, sigma_f: float, nu: float,
                         E: float, alpha: float) -> float:
        """Second thermal shock parameter R' = k·R."""
        R = ThermalShock.first_parameter(sigma_f, nu, E, alpha)
        return k * R


class ThermalBuckling:
    """Thermal buckling analysis."""
    
    @staticmethod
    def critical_temperature_plate(
        a: float, b: float, h: float,
        E: float, nu: float, alpha: float
    ) -> float:
        """
        Critical temperature rise for plate buckling.
        
        Simply supported, uniaxial constraint.
        """
        D = E * h**3 / (12 * (1 - nu**2))
        Ncr = 4 * math.pi**2 * D / b**2
        
        delta_T_cr = Ncr / (E * h * alpha)
        return delta_T_cr
    
    @staticmethod
    def critical_temperature_column(L: float, A: float, I: float,
                                    E: float, alpha: float) -> float:
        """Critical temperature for Euler column buckling."""
        Pcr = math.pi**2 * E * I / L**2
        return Pcr / (E * A * alpha)


class ThermoelasticCoupling:
    """Coupled thermoelastic effects."""
    
    def __init__(self, E: float, alpha: float, rho: float, 
                 c_p: float, T_0: float):
        """
        Initialize thermoelastic coupling.
        
        Args:
            E: Young's modulus
            alpha: CTE
            rho: Density  
            c_p: Specific heat
            T_0: Reference temperature
        """
        self.E = E
        self.alpha = alpha
        self.rho = rho
        self.c_p = c_p
        self.T_0 = T_0
    
    @property
    def coupling_parameter(self) -> float:
        """Thermoelastic coupling parameter."""
        return self.E * self.alpha**2 * self.T_0 / (self.rho * self.c_p)


class HeatConduction:
    """Basic heat conduction for thermal analysis."""
    
    @staticmethod
    def steady_state_1d(k: float, T1: float, T2: float, L: float) -> float:
        """Heat flux q = -k·dT/dx."""
        return k * (T1 - T2) / L
    
    @staticmethod
    def temperature_distribution_plate(
        T_hot: float, T_cold: float, x: float, L: float
    ) -> float:
        """Linear temperature distribution."""
        return T_hot - (T_hot - T_cold) * x / L


class TransientThermal:
    """Transient thermal analysis utilities."""
    
    @staticmethod
    def biot_number(h: float, L: float, k: float) -> float:
        """Biot number Bi = hL/k."""
        return h * L / k
    
    @staticmethod
    def fourier_number(alpha_d: float, t: float, L: float) -> float:
        """Fourier number Fo = αt/L²."""
        return alpha_d * t / L**2
    
    @staticmethod
    def lumped_capacitance(
        T_initial: float, T_ambient: float,
        h: float, A: float, rho: float, V: float, c_p: float, t: float
    ) -> float:
        """Temperature using lumped capacitance method."""
        tau = rho * V * c_p / (h * A)
        return T_ambient + (T_initial - T_ambient) * math.exp(-t / tau)


# Convenience functions

def compute_thermal_strain(alpha: float, delta_T: float) -> float:
    """Compute thermal strain."""
    return alpha * delta_T


def compute_thermal_stress(E: float, alpha: float, delta_T: float, 
                           constrained: bool = True) -> float:
    """Compute thermal stress."""
    if constrained:
        return -E * alpha * delta_T
    return 0.0
