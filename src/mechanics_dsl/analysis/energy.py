"""
Energy Analysis Tools

Provides energy calculation and conservation analysis for mechanical systems.
"""
from typing import Dict, List, Optional, Callable
import numpy as np
import sympy as sp

from ..utils import logger


class EnergyAnalyzer:
    """
    Energy analysis for mechanical systems.
    
    Computes kinetic energy, potential energy, and validates conservation.
    """
    
    def __init__(self):
        self._kinetic_func: Optional[Callable] = None
        self._potential_func: Optional[Callable] = None
    
    def compute_kinetic_energy(self, solution: dict, 
                              masses: Dict[str, float],
                              velocities: Optional[List[str]] = None) -> np.ndarray:
        """
        Compute kinetic energy T = 0.5 * m * v^2 for each timestep.
        
        Args:
            solution: Simulation result
            masses: Dict mapping coordinate names to masses
            velocities: List of velocity variable names
            
        Returns:
            Kinetic energy array
        """
        y = solution['y']
        coords = solution.get('coordinates', [])
        n_points = len(solution['t'])
        
        T = np.zeros(n_points)
        
        for i, q in enumerate(coords):
            m = masses.get(q, masses.get('m', 1.0))
            v = y[2*i + 1]  # velocity is odd indices
            T += 0.5 * m * v**2
        
        return T
    
    def compute_potential_energy(self, solution: dict,
                                potential_func: Callable) -> np.ndarray:
        """
        Compute potential energy at each timestep.
        
        Args:
            solution: Simulation result
            potential_func: Function V(state) -> float
            
        Returns:
            Potential energy array
        """
        y = solution['y']
        n_points = y.shape[1]
        
        V = np.array([potential_func(y[:, i]) for i in range(n_points)])
        return V
    
    def check_conservation(self, solution: dict,
                          kinetic: np.ndarray,
                          potential: np.ndarray,
                          tolerance: float = 1e-3) -> Dict:
        """
        Check energy conservation.
        
        Args:
            solution: Simulation result
            kinetic: Kinetic energy array
            potential: Potential energy array
            tolerance: Relative tolerance for conservation check
            
        Returns:
            Dict with conservation analysis results
        """
        total = kinetic + potential
        E0 = total[0]
        
        if abs(E0) < 1e-10:
            relative_error = np.abs(total - E0)
        else:
            relative_error = np.abs((total - E0) / E0)
        
        max_error = np.max(relative_error)
        mean_error = np.mean(relative_error)
        
        conserved = max_error < tolerance
        
        result = {
            'conserved': conserved,
            'initial_energy': E0,
            'max_relative_error': max_error,
            'mean_relative_error': mean_error,
            'total_energy': total,
            'kinetic_energy': kinetic,
            'potential_energy': potential,
        }
        
        if conserved:
            logger.info(f"Energy conserved: max error = {max_error:.2e}")
        else:
            logger.warning(f"Energy NOT conserved: max error = {max_error:.2e}")
        
        return result
    
    def compute_pendulum_energy(self, solution: dict,
                               m: float, l: float, g: float) -> Dict:
        """
        Compute energy for simple pendulum.
        
        Args:
            solution: Simulation result
            m: Mass
            l: Length
            g: Gravitational acceleration
            
        Returns:
            Dict with kinetic, potential, and total energy
        """
        y = solution['y']
        theta = y[0]
        theta_dot = y[1]
        
        T = 0.5 * m * (l * theta_dot)**2
        V = m * g * l * (1 - np.cos(theta))
        
        return {
            'kinetic': T,
            'potential': V,
            'total': T + V
        }
