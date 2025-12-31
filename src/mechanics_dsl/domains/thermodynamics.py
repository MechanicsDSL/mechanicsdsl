"""
Thermodynamics Domain for MechanicsDSL

Provides tools for thermodynamic calculations, including:
- Heat engines and cycles (Carnot, Otto, Diesel, Rankine)
- Equations of state (ideal gas, van der Waals, Redlich-Kwong)
- Phase transitions
- Maxwell relations
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable, Union
import numpy as np
from enum import Enum


# Physical constants
R_GAS = 8.314462  # J/(mol·K)
BOLTZMANN = 1.380649e-23  # J/K


class ProcessType(Enum):
    """Thermodynamic process types."""
    ISOTHERMAL = "isothermal"       # Constant T
    ADIABATIC = "adiabatic"         # No heat transfer
    ISOBARIC = "isobaric"           # Constant P
    ISOCHORIC = "isochoric"         # Constant V
    POLYTROPIC = "polytropic"       # PV^n = const


@dataclass
class ThermodynamicProcess:
    """
    Represents a thermodynamic process.
    
    Attributes:
        process_type: Type of process
        initial_state: (P, V, T) initial
        final_state: (P, V, T) final
        heat: Heat transferred Q
        work: Work done W
    """
    process_type: ProcessType
    initial_state: Tuple[float, float, float]  # (P, V, T)
    final_state: Tuple[float, float, float]
    heat: float = 0.0
    work: float = 0.0


class CarnotEngine:
    """
    Carnot heat engine (ideal maximum efficiency).
    
    Efficiency η = 1 - T_cold/T_hot
    
    Consists of:
    1. Isothermal expansion at T_hot
    2. Adiabatic expansion
    3. Isothermal compression at T_cold
    4. Adiabatic compression
    
    Example:
        >>> engine = CarnotEngine(T_hot=500, T_cold=300)
        >>> eta = engine.efficiency()  # 0.4 (40%)
    """
    
    def __init__(self, T_hot: float, T_cold: float):
        """
        Initialize Carnot engine.
        
        Args:
            T_hot: Hot reservoir temperature (K)
            T_cold: Cold reservoir temperature (K)
        """
        if T_cold >= T_hot:
            raise ValueError("T_cold must be < T_hot")
        self.T_hot = T_hot
        self.T_cold = T_cold
    
    def efficiency(self) -> float:
        """Carnot efficiency η = 1 - T_cold/T_hot."""
        return 1 - self.T_cold / self.T_hot
    
    def work_output(self, Q_hot: float) -> float:
        """Work output W = η × Q_hot."""
        return self.efficiency() * Q_hot
    
    def heat_rejected(self, Q_hot: float) -> float:
        """Heat rejected Q_cold = Q_hot - W."""
        return Q_hot - self.work_output(Q_hot)
    
    def cop_refrigerator(self) -> float:
        """COP as refrigerator = T_cold/(T_hot - T_cold)."""
        return self.T_cold / (self.T_hot - self.T_cold)
    
    def cop_heat_pump(self) -> float:
        """COP as heat pump = T_hot/(T_hot - T_cold)."""
        return self.T_hot / (self.T_hot - self.T_cold)


class OttoCycle:
    """
    Otto cycle (gasoline engine).
    
    1-2: Adiabatic compression
    2-3: Isochoric heat addition
    3-4: Adiabatic expansion
    4-1: Isochoric heat rejection
    
    Efficiency η = 1 - 1/r^(γ-1)
    
    where r = V_max/V_min is compression ratio.
    """
    
    def __init__(self, compression_ratio: float, gamma: float = 1.4):
        """
        Initialize Otto cycle.
        
        Args:
            compression_ratio: V1/V2
            gamma: Heat capacity ratio (1.4 for air)
        """
        self.r = compression_ratio
        self.gamma = gamma
    
    def efficiency(self) -> float:
        """Otto efficiency η = 1 - 1/r^(γ-1)."""
        return 1 - 1 / self.r**(self.gamma - 1)
    
    def pressure_ratio(self, T3_over_T2: float) -> float:
        """Pressure ratio P3/P2 for heat addition."""
        return T3_over_T2
    
    def work_output(self, n: float, T1: float, T3: float, Cv: float = None) -> float:
        """
        Net work output.
        
        W = n × Cv × [(T3 - T2) - (T4 - T1)]
        """
        if Cv is None:
            Cv = R_GAS / (self.gamma - 1)
        
        T2 = T1 * self.r**(self.gamma - 1)
        T4 = T3 / self.r**(self.gamma - 1)
        
        return n * Cv * ((T3 - T2) - (T4 - T1))


class DieselCycle:
    """
    Diesel cycle (compression ignition engine).
    
    1-2: Adiabatic compression
    2-3: Isobaric heat addition
    3-4: Adiabatic expansion
    4-1: Isochoric heat rejection
    
    Efficiency η = 1 - (1/r^(γ-1)) × (ρ^γ - 1)/(γ(ρ - 1))
    
    where r is compression ratio, ρ is cutoff ratio.
    """
    
    def __init__(self, compression_ratio: float, cutoff_ratio: float,
                 gamma: float = 1.4):
        """
        Initialize Diesel cycle.
        
        Args:
            compression_ratio: V1/V2
            cutoff_ratio: V3/V2 (volume at end of combustion / start)
            gamma: Heat capacity ratio
        """
        self.r = compression_ratio
        self.rho = cutoff_ratio
        self.gamma = gamma
    
    def efficiency(self) -> float:
        """Diesel efficiency."""
        r = self.r
        rho = self.rho
        gamma = self.gamma
        
        return 1 - (1/r**(gamma-1)) * (rho**gamma - 1) / (gamma * (rho - 1))


class VanDerWaalsGas:
    """
    Van der Waals equation of state.
    
    (P + a/V²)(V - b) = RT
    
    where a accounts for intermolecular attraction,
    b accounts for molecular volume.
    
    Example:
        >>> co2 = VanDerWaalsGas(a=0.364, b=4.27e-5)  # CO2 parameters
        >>> P = co2.pressure(V=0.001, T=300)
    """
    
    def __init__(self, a: float, b: float, R: float = R_GAS):
        """
        Initialize van der Waals gas.
        
        Args:
            a: Attraction parameter (Pa·m⁶/mol²)
            b: Volume parameter (m³/mol)
            R: Gas constant
        """
        self.a = a
        self.b = b
        self.R = R
    
    def pressure(self, V: float, T: float, n: float = 1.0) -> float:
        """
        Calculate pressure from van der Waals equation.
        
        P = nRT/(V - nb) - a(n/V)²
        """
        return n * self.R * T / (V - n * self.b) - self.a * (n / V)**2
    
    def critical_point(self) -> Tuple[float, float, float]:
        """
        Critical point (P_c, V_c, T_c).
        
        V_c = 3b, P_c = a/(27b²), T_c = 8a/(27Rb)
        """
        V_c = 3 * self.b
        P_c = self.a / (27 * self.b**2)
        T_c = 8 * self.a / (27 * self.R * self.b)
        return P_c, V_c, T_c
    
    def compressibility_factor(self, V: float, T: float, n: float = 1.0) -> float:
        """
        Compressibility factor Z = PV/(nRT).
        
        Z = 1 for ideal gas, deviates for real gas.
        """
        P = self.pressure(V, T, n)
        return P * V / (n * self.R * T)
    
    def reduced_equation(self, P_r: float, V_r: float) -> float:
        """
        Reduced temperature from reduced P and V.
        
        T_r = (8/3) × (P_r + 3/V_r²)(V_r - 1/3)
        """
        return (8/3) * (P_r + 3/V_r**2) * (V_r - 1/3)


class PhaseTransition:
    """
    Phase transition calculations.
    
    Supports solid-liquid, liquid-gas, and solid-gas transitions.
    
    Clausius-Clapeyron equation:
    dP/dT = ΔH / (T × ΔV)
    """
    
    def __init__(self, T_transition: float, P_transition: float,
                 latent_heat: float, delta_V: float):
        """
        Initialize phase transition.
        
        Args:
            T_transition: Transition temperature (K)
            P_transition: Transition pressure (Pa)
            latent_heat: Latent heat of transition (J/mol)
            delta_V: Molar volume change (m³/mol)
        """
        self.T0 = T_transition
        self.P0 = P_transition
        self.L = latent_heat
        self.dV = delta_V
    
    def clausius_clapeyron_slope(self) -> float:
        """dP/dT = L/(T × ΔV)."""
        return self.L / (self.T0 * self.dV)
    
    def transition_pressure(self, T: float) -> float:
        """
        Pressure at temperature T using Clausius-Clapeyron.
        
        P(T) ≈ P0 × exp(L/R × (1/T0 - 1/T))
        """
        return self.P0 * np.exp(self.L / R_GAS * (1/self.T0 - 1/T))
    
    def transition_temperature(self, P: float) -> float:
        """Temperature at pressure P."""
        return 1 / (1/self.T0 - R_GAS/self.L * np.log(P/self.P0))
    
    def entropy_change(self) -> float:
        """Entropy change ΔS = L/T."""
        return self.L / self.T0


class HeatCapacity:
    """
    Heat capacity calculations for various substances.
    """
    
    @staticmethod
    def debye_heat_capacity(T: float, theta_D: float, n: int = 1) -> float:
        """
        Debye model heat capacity.
        
        C_V = 9nR (T/θ_D)³ ∫₀^(θ_D/T) x⁴e^x/(e^x-1)² dx
        
        Args:
            T: Temperature (K)
            theta_D: Debye temperature (K)
            n: Number of atoms per formula unit
        """
        from scipy.integrate import quad
        
        if T >= theta_D:
            return 3 * n * R_GAS  # Classical limit
        
        x_max = theta_D / T
        
        def integrand(x):
            if x < 1e-10:
                return 0
            ex = np.exp(x)
            return x**4 * ex / (ex - 1)**2
        
        integral, _ = quad(integrand, 0, x_max)
        
        return 9 * n * R_GAS * (T / theta_D)**3 * integral
    
    @staticmethod
    def einstein_heat_capacity(T: float, theta_E: float, n: int = 1) -> float:
        """
        Einstein model heat capacity.
        
        C_V = 3nR (θ_E/T)² exp(θ_E/T)/(exp(θ_E/T)-1)²
        """
        x = theta_E / T
        return 3 * n * R_GAS * x**2 * np.exp(x) / (np.exp(x) - 1)**2


class MaxwellRelations:
    """
    Maxwell relations for thermodynamic potentials.
    
    Derived from the equality of mixed partial derivatives.
    """
    
    @staticmethod
    def relation_1() -> str:
        """(∂T/∂V)_S = -(∂P/∂S)_V from dU = TdS - PdV."""
        return "(∂T/∂V)_S = -(∂P/∂S)_V"
    
    @staticmethod
    def relation_2() -> str:
        """(∂T/∂P)_S = (∂V/∂S)_P from dH = TdS + VdP."""
        return "(∂T/∂P)_S = (∂V/∂S)_P"
    
    @staticmethod
    def relation_3() -> str:
        """(∂S/∂V)_T = (∂P/∂T)_V from dF = -SdT - PdV."""
        return "(∂S/∂V)_T = (∂P/∂T)_V"
    
    @staticmethod
    def relation_4() -> str:
        """(∂S/∂P)_T = -(∂V/∂T)_P from dG = -SdT + VdP."""
        return "(∂S/∂P)_T = -(∂V/∂T)_P"
    
    @staticmethod
    def thermal_expansion(dV_dT: float, V: float) -> float:
        """Thermal expansion coefficient α = (1/V)(∂V/∂T)_P."""
        return dV_dT / V
    
    @staticmethod
    def isothermal_compressibility(dV_dP: float, V: float) -> float:
        """Isothermal compressibility κ_T = -(1/V)(∂V/∂P)_T."""
        return -dV_dP / V


__all__ = [
    'R_GAS',
    'BOLTZMANN',
    'ProcessType',
    'ThermodynamicProcess',
    'CarnotEngine',
    'OttoCycle',
    'DieselCycle',
    'VanDerWaalsGas',
    'PhaseTransition',
    'HeatCapacity',
    'MaxwellRelations',
]
