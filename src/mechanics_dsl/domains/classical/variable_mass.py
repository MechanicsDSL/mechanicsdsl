"""
Variable Mass Systems for Classical Mechanics

This module implements:
- Tsiolkovsky rocket equation
- Generalized EOM for dm/dt ≠ 0
- Conveyor belt problems
- Rain/hail collection
- Rope/chain dynamics

For a system with variable mass:
F_ext = d(mv)/dt = m(dv/dt) + v_rel(dm/dt)

where v_rel is the velocity of mass entering/leaving the system.
"""
from typing import Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp

from ...utils import logger


class MassFlowType(Enum):
    """Types of mass flow."""
    EJECTION = "ejection"    # Mass leaving (rocket exhaust)
    ACCRETION = "accretion"  # Mass entering (rain collection)
    TRANSFER = "transfer"    # Mass transfer between bodies


@dataclass
class RocketState:
    """
    State of a rocket at a given time.
    
    Attributes:
        time: Current time
        velocity: Current velocity
        mass: Current total mass
        altitude: Current altitude (if applicable)
        fuel_remaining: Fuel mass remaining
    """
    time: float
    velocity: float
    mass: float
    altitude: float = 0.0
    fuel_remaining: float = 0.0
    
    @property
    def kinetic_energy(self) -> float:
        """Get kinetic energy."""
        return 0.5 * self.mass * self.velocity**2


@dataclass 
class RocketParameters:
    """
    Parameters for rocket simulation.
    
    Attributes:
        initial_mass: Total initial mass (rocket + fuel)
        fuel_mass: Initial fuel mass
        exhaust_velocity: Exhaust velocity relative to rocket
        mass_flow_rate: Rate of mass ejection (dm/dt)
        thrust: Thrust force (alternative to exhaust_velocity)
        external_force: External force function F(t, state)
    """
    initial_mass: float
    fuel_mass: float
    exhaust_velocity: float
    mass_flow_rate: float
    thrust: Optional[float] = None
    external_force: Optional[Callable[[float, 'RocketState'], float]] = None
    
    @property
    def dry_mass(self) -> float:
        """Mass without fuel."""
        return self.initial_mass - self.fuel_mass
    
    @property
    def mass_ratio(self) -> float:
        """Initial to final mass ratio."""
        return self.initial_mass / self.dry_mass


class RocketEquation:
    """
    Tsiolkovsky rocket equation solver.
    
    The ideal rocket equation (no external forces, no gravity):
    Δv = v_e * ln(m_0 / m_f)
    
    With gravity:
    Δv = v_e * ln(m_0 / m_f) - g * t_burn
    
    Example:
        >>> rocket = RocketEquation()
        >>> params = RocketParameters(
        ...     initial_mass=1000, fuel_mass=800,
        ...     exhaust_velocity=3000, mass_flow_rate=10
        ... )
        >>> result = rocket.ideal_delta_v(params)
    """
    
    def ideal_delta_v(self, params: RocketParameters) -> float:
        """
        Compute ideal delta-v from Tsiolkovsky equation.
        
        Δv = v_e * ln(m_0 / m_f)
        
        Args:
            params: Rocket parameters
            
        Returns:
            Ideal velocity change
        """
        return params.exhaust_velocity * np.log(params.mass_ratio)
    
    def burn_time(self, params: RocketParameters) -> float:
        """
        Compute total burn time.
        
        t_burn = m_fuel / ṁ
        
        Args:
            params: Rocket parameters
            
        Returns:
            Burn time
        """
        return params.fuel_mass / params.mass_flow_rate
    
    def delta_v_with_gravity(self, params: RocketParameters, 
                              g: float = 9.81) -> float:
        """
        Compute delta-v accounting for gravity losses.
        
        Δv = v_e * ln(m_0/m_f) - g * t_burn
        
        Args:
            params: Rocket parameters
            g: Gravitational acceleration
            
        Returns:
            Effective velocity change
        """
        t_burn = self.burn_time(params)
        ideal_dv = self.ideal_delta_v(params)
        
        return ideal_dv - g * t_burn
    
    def simulate(self, params: RocketParameters,
                 t_span: Tuple[float, float],
                 g: float = 0.0,
                 num_points: int = 100) -> Dict[str, np.ndarray]:
        """
        Simulate rocket motion numerically.
        
        Equations:
        dv/dt = (v_e * ṁ - m*g + F_ext) / m
        dm/dt = -ṁ (while fuel remains)
        dx/dt = v
        
        Args:
            params: Rocket parameters
            t_span: (t_start, t_end)
            g: Gravitational acceleration
            num_points: Number of output points
            
        Returns:
            Dictionary with time, velocity, mass, altitude arrays
        """
        v_e = params.exhaust_velocity
        m_dot = params.mass_flow_rate
        m_dry = params.dry_mass
        F_ext = params.external_force
        
        def derivatives(t, state):
            v, x, m = state
            
            # Check if fuel exhausted
            if m <= m_dry:
                # No thrust, only gravity
                dv_dt = -g
                dm_dt = 0
            else:
                # Thrust from mass ejection
                thrust = v_e * m_dot
                if F_ext is not None:
                    rocket_state = RocketState(t, v, m, x, m - m_dry)
                    thrust += F_ext(t, rocket_state)
                
                dv_dt = (thrust - m * g) / m
                dm_dt = -m_dot
            
            dx_dt = v
            
            return [dv_dt, dx_dt, dm_dt]
        
        # Initial conditions
        y0 = [0.0, 0.0, params.initial_mass]  # v=0, x=0, m=m_0
        
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        
        solution = solve_ivp(derivatives, t_span, y0, t_eval=t_eval, 
                             method='RK45')
        
        return {
            'time': solution.t,
            'velocity': solution.y[0],
            'altitude': solution.y[1],
            'mass': solution.y[2]
        }
    
    def specific_impulse(self, params: RocketParameters, 
                         g0: float = 9.81) -> float:
        """
        Compute specific impulse.
        
        I_sp = v_e / g_0
        
        Args:
            params: Rocket parameters
            g0: Standard gravity (for unit conversion)
            
        Returns:
            Specific impulse in seconds
        """
        return params.exhaust_velocity / g0


class VariableMassSystem:
    """
    General variable mass system solver.
    
    For a body with mass m(t) and external forces:
    F_ext = m * dv/dt + v_rel * dm/dt
    
    where v_rel is the velocity of mass entering/leaving.
    
    Supports:
    - Mass ejection (rockets)
    - Mass accretion (collecting rain)
    - Conveyor belt problems
    """
    
    def __init__(self):
        self._symbol_cache: Dict[str, sp.Symbol] = {}
    
    def get_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """Get or create a symbol."""
        if name not in self._symbol_cache:
            default_assumptions = {'real': True}
            default_assumptions.update(assumptions)
            self._symbol_cache[name] = sp.Symbol(name, **default_assumptions)
        return self._symbol_cache[name]
    
    def derive_eom(self, mass_rate: sp.Expr,
                   relative_velocity: sp.Expr,
                   external_force: sp.Expr) -> sp.Expr:
        """
        Derive equation of motion for variable mass system.
        
        F_ext = m * a + v_rel * dm/dt
        => a = (F_ext - v_rel * ṁ) / m
        
        Args:
            mass_rate: dm/dt expression
            relative_velocity: Velocity of added/removed mass
            external_force: External force expression
            
        Returns:
            Acceleration expression
        """
        m = self.get_symbol('m', positive=True)
        
        acceleration = (external_force - relative_velocity * mass_rate) / m
        
        return sp.simplify(acceleration)
    
    def conveyor_belt_force(self, belt_velocity: float,
                            mass_rate: float) -> float:
        """
        Compute force needed to maintain conveyor belt velocity.
        
        Material lands on belt at rate dm/dt with zero horizontal velocity.
        Force needed: F = v_belt * dm/dt
        
        Args:
            belt_velocity: Velocity of belt
            mass_rate: Rate of mass addition
            
        Returns:
            Required force
        """
        return belt_velocity * mass_rate
    
    def falling_chain(self, chain_length: float,
                      linear_density: float,
                      height: float,
                      g: float = 9.81) -> float:
        """
        Compute force on table from falling chain.
        
        A chain of length L falls from height h. The force on the
        table has two components:
        1. Weight of chain already on table
        2. Impact force from falling links
        
        Args:
            chain_length: Total chain length
            linear_density: Mass per unit length λ
            height: Current falling height
            g: Gravitational acceleration
            
        Returns:
            Total force on table
        """
        # Velocity of falling chain at current height
        v = np.sqrt(2 * g * height)
        
        # Chain on table
        on_table = max(0, chain_length - height)
        weight_on_table = linear_density * on_table * g
        
        # Impact force: F_impact = λv² (from momentum flux)
        impact_force = linear_density * v**2
        
        return weight_on_table + impact_force


class SymbolicVariableMass:
    """
    Symbolic analysis of variable mass systems.
    """
    
    def __init__(self):
        self._symbol_cache: Dict[str, sp.Symbol] = {}
    
    def get_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """Get or create a symbol."""
        if name not in self._symbol_cache:
            default_assumptions = {'real': True}
            default_assumptions.update(assumptions)
            self._symbol_cache[name] = sp.Symbol(name, **default_assumptions)
        return self._symbol_cache[name]
    
    def rocket_equation_symbolic(self) -> Dict[str, sp.Expr]:
        """
        Derive rocket equation symbolically.
        
        Returns:
            Dictionary of symbolic expressions
        """
        m0 = self.get_symbol('m_0', positive=True)
        mf = self.get_symbol('m_f', positive=True)
        ve = self.get_symbol('v_e', positive=True)
        g = self.get_symbol('g', positive=True)
        t = self.get_symbol('t', positive=True)
        m_dot = self.get_symbol('m_dot', negative=True)
        
        # Ideal delta-v
        delta_v_ideal = ve * sp.log(m0 / mf)
        
        # Burn time
        m_fuel = m0 - mf
        t_burn = -m_fuel / m_dot  # m_dot is negative
        
        # Delta-v with gravity loss
        delta_v_gravity = delta_v_ideal - g * t_burn
        
        # Specific impulse
        g0 = self.get_symbol('g_0', positive=True)
        I_sp = ve / g0
        
        return {
            'delta_v_ideal': delta_v_ideal,
            'delta_v_with_gravity': delta_v_gravity,
            'burn_time': t_burn,
            'specific_impulse': I_sp,
            'mass_ratio': m0 / mf
        }
    
    def multistage_rocket(self, stages: int = 2) -> sp.Expr:
        """
        Derive delta-v for multistage rocket.
        
        Δv_total = Σᵢ v_eᵢ * ln(m_0ᵢ / m_fᵢ)
        
        Args:
            stages: Number of stages
            
        Returns:
            Total delta-v expression
        """
        total_dv = sp.S.Zero
        
        for i in range(1, stages + 1):
            ve = self.get_symbol(f'v_e{i}', positive=True)
            m0 = self.get_symbol(f'm_{i}0', positive=True)
            mf = self.get_symbol(f'm_{i}f', positive=True)
            
            total_dv += ve * sp.log(m0 / mf)
        
        return total_dv


# Convenience functions

def tsiolkovsky_delta_v(m_initial: float, m_final: float,
                        exhaust_velocity: float) -> float:
    """
    Compute ideal delta-v from Tsiolkovsky equation.
    
    Δv = v_e * ln(m_0 / m_f)
    
    Args:
        m_initial: Initial total mass
        m_final: Final mass (after fuel burned)
        exhaust_velocity: Exhaust velocity
        
    Returns:
        Delta-v
    """
    return exhaust_velocity * np.log(m_initial / m_final)


def required_mass_ratio(delta_v: float, exhaust_velocity: float) -> float:
    """
    Compute required mass ratio for desired delta-v.
    
    m_0/m_f = exp(Δv / v_e)
    
    Args:
        delta_v: Desired velocity change
        exhaust_velocity: Exhaust velocity
        
    Returns:
        Required mass ratio
    """
    return np.exp(delta_v / exhaust_velocity)


def specific_impulse_to_exhaust_velocity(isp: float, 
                                          g0: float = 9.81) -> float:
    """
    Convert specific impulse to exhaust velocity.
    
    v_e = I_sp * g_0
    
    Args:
        isp: Specific impulse in seconds
        g0: Standard gravity
        
    Returns:
        Exhaust velocity
    """
    return isp * g0
