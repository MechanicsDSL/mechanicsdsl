"""
Plotting tools for MechanicsDSL

Provides plotting utilities for simulation results, trajectories, and energy.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from ..utils import logger


class Plotter:
    """
    Plotting utilities for simulation analysis.
    
    Provides methods for:
    - Time series plots
    - Trajectory plots
    - Energy plots
    - Multi-panel figures
    """
    
    def __init__(self):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")
    
    def plot_time_series(self, solution: dict, 
                        variables: Optional[List[str]] = None,
                        title: str = "Time Series") -> plt.Figure:
        """
        Plot state variables vs time.
        
        Args:
            solution: Simulation result
            variables: List of variables to plot (default: all coordinates)
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        t = solution['t']
        y = solution['y']
        coords = solution.get('coordinates', [])
        
        if variables is None:
            variables = coords
        
        n_vars = len(variables)
        fig, axes = plt.subplots(n_vars, 1, figsize=(10, 3*n_vars), sharex=True)
        if n_vars == 1:
            axes = [axes]
        
        for i, var in enumerate(variables):
            if var in coords:
                idx = coords.index(var)
                axes[i].plot(t, y[2*idx], label=var)
                axes[i].plot(t, y[2*idx + 1], '--', label=f'{var}_dot')
            axes[i].set_ylabel(var)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(title)
        plt.tight_layout()
        
        return fig
    
    def plot_trajectory_2d(self, solution: dict,
                          x_var: str = 'x', y_var: str = 'y',
                          title: str = "Trajectory") -> plt.Figure:
        """
        Plot 2D trajectory from solution.
        
        Args:
            solution: Simulation result
            x_var, y_var: Variable names for x and y axes
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        y = solution['y']
        coords = solution.get('coordinates', [])
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        if x_var in coords and y_var in coords:
            x_idx = coords.index(x_var)
            y_idx = coords.index(y_var)
            ax.plot(y[2*x_idx], y[2*y_idx])
            ax.scatter(y[2*x_idx, 0], y[2*y_idx, 0], c='green', s=100, 
                      zorder=5, label='Start')
            ax.scatter(y[2*x_idx, -1], y[2*y_idx, -1], c='red', s=100,
                      zorder=5, label='End')
        
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_energy(self, solution: dict, 
                   kinetic: np.ndarray, potential: np.ndarray,
                   title: str = "Energy Conservation") -> plt.Figure:
        """
        Plot energy components over time.
        
        Args:
            solution: Simulation result
            kinetic: Kinetic energy array
            potential: Potential energy array
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        t = solution['t']
        total = kinetic + potential
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        ax1.plot(t, kinetic, label='Kinetic')
        ax1.plot(t, potential, label='Potential')
        ax1.plot(t, total, label='Total', lw=2, color='black')
        ax1.set_ylabel('Energy (J)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(title)
        
        # Energy error
        E0 = total[0]
        error = (total - E0) / abs(E0) * 100 if abs(E0) > 1e-10 else total - E0
        ax2.plot(t, error, color='red')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Energy Error (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def show():
        """Display all figures."""
        plt.show()
    
    @staticmethod
    def save_figure(fig: plt.Figure, filename: str, dpi: int = 150) -> None:
        """Save figure to file."""
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        logger.info(f"Figure saved to {filename}")
