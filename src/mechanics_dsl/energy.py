"""
Energy calculation utilities for MechanicsDSL
"""
import numpy as np
from typing import Dict

from .utils import logger, validate_solution_dict

class PotentialEnergyCalculator:
    """Compute potential energy with proper offset for different systems"""
    
    @staticmethod
    def compute_pe_offset(system_type: str, parameters: Dict[str, float]) -> float:
        """
        Compute PE offset to set minimum PE = 0
        
        Args:
            system_type: Type of mechanical system
            parameters: System parameters
            
        Returns:
            PE offset value
        """
        system = system_type.lower()
        
        if 'double' in system and 'pendulum' in system:
            m1 = parameters.get('m1', 1.0)
            m2 = parameters.get('m2', 1.0)
            l1 = parameters.get('l1', 1.0)
            l2 = parameters.get('l2', 1.0)
            g = parameters.get('g', 9.81)
            # Minimum PE when both pendulums hang straight down
            return -m1 * g * l1 - m2 * g * (l1 + l2)
            
        elif 'pendulum' in system:
            m = parameters.get('m', 1.0)
            l = parameters.get('l', 1.0)
            g = parameters.get('g', 9.81)
            # Minimum PE when pendulum hangs straight down
            return -m * g * l
            
        elif 'oscillator' in system or 'spring' in system:
            # Harmonic oscillator: PE minimum is already at x=0
            return 0.0
            
        else:
            # Default: no offset
            return 0.0
    
    @staticmethod
    def compute_kinetic_energy(solution: dict, parameters: Dict[str, float]) -> np.ndarray:
        """
        Compute kinetic energy from solution with validation.
        
        Args:
            solution: Solution dictionary (validated)
            parameters: System parameters dictionary
            
        Returns:
            Array of kinetic energy values
            
        Raises:
            TypeError: If inputs have wrong types
            ValueError: If solution is invalid
        """
        if not isinstance(parameters, dict):
            raise TypeError(f"parameters must be dict, got {type(parameters).__name__}")
        
        validate_solution_dict(solution)
        
        t = solution['t']
        y = solution['y']
        coords = solution['coordinates']
        
        KE = np.zeros_like(t)
        
        if len(coords) == 0:
            logger.warning("No coordinates found for kinetic energy calculation")
            return KE
        
        if 'theta' in coords[0]:  # Pendulum systems
            if len(coords) == 1:  # Simple pendulum
                if y.shape[0] < 2:
                    logger.warning("Insufficient state vector for simple pendulum KE")
                    return KE
                theta_dot = y[1]
                m = parameters.get('m', 1.0)
                l = parameters.get('l', 1.0)
                KE = 0.5 * m * l**2 * theta_dot**2
                
            elif len(coords) >= 2:  # Double pendulum
                if y.shape[0] < 4:
                    logger.warning("Insufficient state vector for double pendulum KE")
                    return KE
                theta1_dot, theta2_dot = y[1], y[3]
                theta1, theta2 = y[0], y[2]
                m1 = parameters.get('m1', 1.0)
                m2 = parameters.get('m2', 1.0)
                l1 = parameters.get('l1', 1.0)
                l2 = parameters.get('l2', 1.0)
                
                KE1 = 0.5 * m1 * l1**2 * theta1_dot**2
                KE2 = 0.5 * m2 * (l1**2 * theta1_dot**2 + l2**2 * theta2_dot**2 + 
                                  2 * l1 * l2 * theta1_dot * theta2_dot * np.cos(theta1 - theta2))
                KE = KE1 + KE2
                
        else:  # Cartesian systems
            v = y[1] if y.shape[0] > 1 else np.zeros_like(t)
            m = parameters.get('m', 1.0)
            KE = 0.5 * m * v**2
            
        return KE
    
    @staticmethod
    def compute_potential_energy(solution: dict, parameters: Dict[str, float], 
                                system_type: str = "") -> np.ndarray:
        """Compute potential energy from solution with proper offset"""
        t = solution['t']
        y = solution['y']
        coords = solution['coordinates']
        
        PE = np.zeros_like(t)
        
        if len(coords) == 0:
            logger.warning("No coordinates found for potential energy calculation")
            return PE
        
        if 'theta' in coords[0]:  # Pendulum systems
            if len(coords) == 1:  # Simple pendulum
                if y.shape[0] < 1:
                    logger.warning("Insufficient state vector for simple pendulum PE")
                    return PE
                theta = y[0]
                m = parameters.get('m', 1.0)
                l = parameters.get('l', 1.0)
                g = parameters.get('g', 9.81)
                
                PE = -m * g * l * np.cos(theta)
                offset = PotentialEnergyCalculator.compute_pe_offset('simple_pendulum', parameters)
                PE = PE - offset
                
            elif len(coords) >= 2:  # Double pendulum
                if y.shape[0] < 3:
                    logger.warning("Insufficient state vector for double pendulum PE")
                    return PE
                theta1, theta2 = y[0], y[2]
                m1 = parameters.get('m1', 1.0)
                m2 = parameters.get('m2', 1.0)
                l1 = parameters.get('l1', 1.0)
                l2 = parameters.get('l2', 1.0)
                g = parameters.get('g', 9.81)
                
                PE1 = -m1 * g * l1 * np.cos(theta1)
                PE2 = -m2 * g * (l1 * np.cos(theta1) + l2 * np.cos(theta2))
                PE = PE1 + PE2
                
                offset = PotentialEnergyCalculator.compute_pe_offset('double_pendulum', parameters)
                PE = PE - offset
                
        else:  # Cartesian systems
            if y.shape[0] < 1:
                logger.warning("Insufficient state vector for Cartesian PE")
                return PE
            x = y[0]
            k = parameters.get('k', 1.0)
            PE = 0.5 * k * x**2
            
        return PE
