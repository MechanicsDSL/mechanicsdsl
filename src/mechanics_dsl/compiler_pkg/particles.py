"""
Particle generation for SPH fluid simulation.

This module provides utilities for generating particle positions
within geometric regions for Smoothed Particle Hydrodynamics (SPH)
simulations.

Classes:
    ParticleGenerator: Generates discrete particle positions from regions.

Example:
    >>> from mechanics_dsl.parser import RegionDef
    >>> from mechanics_dsl.compiler import ParticleGenerator
    >>> region = RegionDef('rectangle', {'x': (0, 1), 'y': (0, 1)})
    >>> particles = ParticleGenerator.generate(region, spacing=0.1)
    >>> print(len(particles))  # 100 particles in a 10x10 grid
"""
import numpy as np
from typing import List, Tuple

from ..parser import RegionDef


class ParticleGenerator:
    """
    Generates discrete particle positions from geometric regions.
    
    This class provides static methods for generating evenly-spaced
    particle positions within defined regions, useful for SPH fluid
    simulation setup.
    
    Supported shapes:
        - rectangle: 2D grid of particles
        - line: 1D line of particles (for boundaries)
    
    Example:
        >>> region = RegionDef('rectangle', {'x': (0, 2), 'y': (0, 1)})
        >>> particles = ParticleGenerator.generate(region, spacing=0.5)
        >>> print(particles)
        [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.5, 0.0),
         (0.0, 0.5), (0.5, 0.5), (1.0, 0.5), (1.5, 0.5)]
    """
    
    @staticmethod
    def generate(region: RegionDef, spacing: float) -> List[Tuple[float, float]]:
        """
        Generate grid of particles within a region.
        
        Args:
            region: RegionDef specifying shape and constraints.
            spacing: Distance between adjacent particles.
            
        Returns:
            List of (x, y) tuples representing particle positions.
            
        Example:
            >>> region = RegionDef('rectangle', {'x': (0, 1), 'y': (0, 1)})
            >>> particles = ParticleGenerator.generate(region, 0.5)
            >>> len(particles)
            4
        """
        if region.shape == "rectangle":
            x_range = region.constraints.get('x', (0, 0))
            y_range = region.constraints.get('y', (0, 0))
            
            # Create grid
            x_points = np.arange(x_range[0], x_range[1], spacing)
            y_points = np.arange(y_range[0], y_range[1], spacing)
            
            # Meshgrid
            xx, yy = np.meshgrid(x_points, y_points)
            
            # Flatten to list of (x, y) tuples
            return list(zip(xx.flatten(), yy.flatten()))
            
        elif region.shape == "line":
            # Useful for boundaries
            x_range = region.constraints.get('x', (0, 0))
            y_range = region.constraints.get('y', (0, 0))
            
            if x_range[0] == x_range[1]:  # Vertical line
                y_points = np.arange(y_range[0], y_range[1], spacing/2.0)  # Denser walls
                return [(x_range[0], y) for y in y_points]
            else:  # Horizontal line
                x_points = np.arange(x_range[0], x_range[1], spacing/2.0)
                return [(x, y_range[0]) for x in x_points]
        
        return []


__all__ = ['ParticleGenerator']
