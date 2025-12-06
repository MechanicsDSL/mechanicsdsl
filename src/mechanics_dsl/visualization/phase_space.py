"""
Phase space visualization for MechanicsDSL

Specialized tools for phase space and Poincaré section analysis.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from ..utils import logger


class PhaseSpaceVisualizer:
    """
    Phase space and Poincaré section visualization.
    
    Provides tools for analyzing dynamical systems through
    phase portraits and stroboscopic maps.
    """
    
    def __init__(self):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for visualization")
    
    def plot_phase_portrait(self, solution: dict, 
                           coordinate_index: int = 0,
                           title: str = "Phase Portrait") -> plt.Figure:
        """
        Plot phase space trajectory (q vs q_dot).
        
        Args:
            solution: Simulation result
            coordinate_index: Which coordinate to plot
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        y = solution['y']
        coords = solution.get('coordinates', [])
        
        if coordinate_index >= len(coords):
            raise ValueError(f"coordinate_index {coordinate_index} out of range")
        
        q = y[2 * coordinate_index]
        q_dot = y[2 * coordinate_index + 1]
        coord_name = coords[coordinate_index]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(q, q_dot, lw=0.5)
        ax.scatter(q[0], q_dot[0], c='green', s=50, zorder=5, label='Start')
        ax.scatter(q[-1], q_dot[-1], c='red', s=50, zorder=5, label='End')
        
        ax.set_xlabel(f'{coord_name}')
        ax.set_ylabel(f'{coord_name}_dot')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_phase_portrait_3d(self, solution: dict,
                              coords: Tuple[int, int, int] = (0, 0, 1),
                              title: str = "3D Phase Space") -> plt.Figure:
        """
        Plot 3D phase space trajectory.
        
        Args:
            solution: Simulation result
            coords: Tuple of (coord1_idx, coord1_type, coord2_idx)
                   where type 0=position, 1=velocity
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        y = solution['y']
        t = solution['t']
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        x = y[2 * coords[0] + coords[1]]
        y_data = y[2 * coords[2]]
        z = t
        
        ax.plot(x, y_data, z, lw=0.5)
        ax.set_xlabel('Coordinate 1')
        ax.set_ylabel('Coordinate 2')
        ax.set_zlabel('Time')
        ax.set_title(title)
        
        return fig
    
    def plot_poincare_section(self, solution: dict,
                             section_var: int = 0,
                             section_value: float = 0.0,
                             plot_vars: Tuple[int, int] = (1, 2),
                             title: str = "Poincaré Section") -> plt.Figure:
        """
        Plot Poincaré section (stroboscopic map).
        
        Args:
            solution: Simulation result
            section_var: State variable index for section condition
            section_value: Value where section is taken
            plot_vars: Which variables to plot (indices)
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        y = solution['y']
        
        # Find crossings
        section_data = y[section_var]
        crossings_idx = []
        
        for i in range(len(section_data) - 1):
            if (section_data[i] - section_value) * (section_data[i+1] - section_value) < 0:
                if section_data[i+1] > section_data[i]:  # Upward crossing
                    crossings_idx.append(i)
        
        if len(crossings_idx) < 2:
            logger.warning("Not enough crossings for Poincaré section")
            return plt.figure()
        
        # Extract values at crossings
        x_points = [y[plot_vars[0], i] for i in crossings_idx]
        y_points = [y[plot_vars[1], i] for i in crossings_idx]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x_points, y_points, s=5, alpha=0.5)
        ax.set_xlabel(f'State variable {plot_vars[0]}')
        ax.set_ylabel(f'State variable {plot_vars[1]}')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return fig
