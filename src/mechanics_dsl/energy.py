import numpy as np
from typing import Dict
from .utils import logger, validate_solution_dict

class PotentialEnergyCalculator:
    """Compute potential energy with proper offset for different systems"""
    
    @staticmethod
    def compute_pe_offset(system_type: str, parameters: Dict[str, float]) -> float:
        """Compute PE offset to set minimum PE = 0"""
        system = system_type.lower()
        if 'pendulum' in system:
            m = parameters.get('m', 1.0)
            l = parameters.get('l', 1.0)
            g = parameters.get('g', 9.81)
            # Minimum when hanging down
            if 'double' in system:
                m1, m2 = parameters.get('m1', 1.0), parameters.get('m2', 1.0)
                l1, l2 = parameters.get('l1', 1.0), parameters.get('l2', 1.0)
                return -m1*g*l1 - m2*g*(l1+l2)
            return -m * g * l
        return 0.0
    
    @staticmethod
    def compute_kinetic_energy(solution: dict, parameters: Dict[str, float]) -> np.ndarray:
        t = solution['t']
        y = solution['y']
        coords = solution['coordinates']
        KE = np.zeros_like(t)
        
        if not coords: return KE
        
        # Heuristic Implementation (Full version parses AST)
        if 'theta' in coords[0]:  # Rotational systems
            if len(coords) == 1:  # Simple
                v = y[1]
                m = parameters.get('m', 1.0)
                l = parameters.get('l', 1.0)
                KE = 0.5 * m * (l*v)**2
            elif len(coords) >= 2:  # Double
                m1, m2 = parameters.get('m1', 1.0), parameters.get('m2', 1.0)
                l1, l2 = parameters.get('l1', 1.0), parameters.get('l2', 1.0)
                t1, t2 = y[0], y[2]
                v1, v2 = y[1], y[3]
                KE = 0.5*m1*(l1*v1)**2 + 0.5*m2*((l1*v1)**2 + (l2*v2)**2 + 2*l1*l2*v1*v2*np.cos(t1-t2))
        else:  # Cartesian systems
            m = parameters.get('m', 1.0)
            # Sum 1/2 m v^2 for all coords
            for i in range(len(coords)):
                v = y[2*i+1]
                KE += 0.5 * m * v**2
        return KE
    
    @staticmethod
    def compute_potential_energy(solution: dict, parameters: Dict[str, float], 
                                system_type: str = "") -> np.ndarray:
        t = solution['t']
        y = solution['y']
        coords = solution['coordinates']
        PE = np.zeros_like(t)
        
        if not coords: return PE
        
        if 'theta' in coords[0]:  # Pendulum
            g = parameters.get('g', 9.81)
            if len(coords) == 1:
                m, l = parameters.get('m', 1.0), parameters.get('l', 1.0)
                PE = -m * g * l * np.cos(y[0])
            elif len(coords) >= 2:
                m1, m2 = parameters.get('m1', 1.0), parameters.get('m2', 1.0)
                l1, l2 = parameters.get('l1', 1.0), parameters.get('l2', 1.0)
                PE = -m1*g*l1*np.cos(y[0]) - m2*g*(l1*np.cos(y[0]) + l2*np.cos(y[2]))
            
            # Apply offset
            offset = PotentialEnergyCalculator.compute_pe_offset(system_type, parameters)
            PE -= offset
        else:  # Spring/Oscillator
            k = parameters.get('k', 1.0)
            x = y[0]
            PE = 0.5 * k * x**2
            
        return PE
