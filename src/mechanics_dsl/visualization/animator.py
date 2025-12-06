"""
Animation tools for MechanicsDSL

Provides animation capabilities for mechanical systems and fluid simulations.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Circle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from ..utils import logger, config


class Animator:
    """
    Animation handler for mechanical system simulations.
    
    Supports:
    - Pendulum animations (single, double, multi-body)
    - Particle trajectory animations
    - Fluid particle visualizations
    """
    
    def __init__(self, trail_length: int = None, fps: int = None):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for animations")
        
        self.trail_length = trail_length or config.trail_length
        self.fps = fps or config.animation_fps
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.animation: Optional[animation.FuncAnimation] = None
        
    def setup_figure(self, xlim: Tuple[float, float] = (-2, 2),
                    ylim: Tuple[float, float] = (-2, 2),
                    title: str = "Animation") -> Tuple[plt.Figure, plt.Axes]:
        """Create and configure figure for animation."""
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_aspect('equal')
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        return self.fig, self.ax
    
    def animate_pendulum(self, solution: dict, length: float = 1.0,
                        title: str = "Pendulum") -> animation.FuncAnimation:
        """
        Create pendulum animation from simulation solution.
        
        Args:
            solution: Simulation result with 't' and 'y' arrays
            length: Pendulum length for visualization
            title: Animation title
            
        Returns:
            matplotlib FuncAnimation object
        """
        t = solution['t']
        y = solution['y']
        
        # Assume first coordinate is angle
        theta = y[0]
        
        self.setup_figure(xlim=(-1.5*length, 1.5*length),
                         ylim=(-1.5*length, 0.5*length),
                         title=title)
        
        # Create elements
        line, = self.ax.plot([], [], 'o-', lw=2, color='blue')
        trail, = self.ax.plot([], [], '-', lw=1, alpha=0.5, color='red')
        time_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)
        
        # Trail buffer
        trail_x = []
        trail_y = []
        
        def init():
            line.set_data([], [])
            trail.set_data([], [])
            time_text.set_text('')
            return line, trail, time_text
        
        def update(frame):
            x = length * np.sin(theta[frame])
            y_pos = -length * np.cos(theta[frame])
            
            line.set_data([0, x], [0, y_pos])
            
            # Update trail
            trail_x.append(x)
            trail_y.append(y_pos)
            if len(trail_x) > self.trail_length:
                trail_x.pop(0)
                trail_y.pop(0)
            trail.set_data(trail_x, trail_y)
            
            time_text.set_text(f't = {t[frame]:.2f}s')
            return line, trail, time_text
        
        interval = 1000 / self.fps
        self.animation = animation.FuncAnimation(
            self.fig, update, frames=len(t),
            init_func=init, blit=True, interval=interval
        )
        
        return self.animation
    
    def animate(self, solution: dict, parameters: dict = None, system_name: str = "system"):
        """
        Generic animation dispatcher (backward compatibility with MechanicsVisualizer).
        
        Args:
            solution: Simulation result dictionary
            parameters: Physical parameters (optional)
            system_name: Name of the system
            
        Returns:
            matplotlib FuncAnimation object
        """
        if not solution or not solution.get('success'):
            return None
        
        coords = solution.get('coordinates', [])
        name = (system_name or '').lower()
        length = parameters.get('l', 1.0) if parameters else 1.0
        
        # Dispatch to appropriate animation
        if 'pendulum' in name or any('theta' in str(c) for c in coords):
            return self.animate_pendulum(solution, length=length, title=system_name)
        else:
            return self.animate_pendulum(solution, length=length, title=system_name)
    
    def animate_particles(self, positions: List[Tuple[np.ndarray, np.ndarray]],
                         title: str = "Particles") -> animation.FuncAnimation:
        """
        Animate particle positions over time.
        
        Args:
            positions: List of (x_array, y_array) for each frame
            title: Animation title
            
        Returns:
            matplotlib FuncAnimation object
        """
        # Determine bounds
        all_x = np.concatenate([p[0] for p in positions])
        all_y = np.concatenate([p[1] for p in positions])
        margin = 0.1 * max(all_x.max() - all_x.min(), all_y.max() - all_y.min())
        
        self.setup_figure(
            xlim=(all_x.min() - margin, all_x.max() + margin),
            ylim=(all_y.min() - margin, all_y.max() + margin),
            title=title
        )
        
        scatter = self.ax.scatter([], [], c='blue', s=10)
        
        def update(frame):
            x, y = positions[frame]
            scatter.set_offsets(np.c_[x, y])
            return scatter,
        
        interval = 1000 / self.fps
        self.animation = animation.FuncAnimation(
            self.fig, update, frames=len(positions),
            blit=True, interval=interval
        )
        
        return self.animation
    
    def save(self, filename: str, dpi: int = 100) -> bool:
        """
        Save animation to file.
        
        Args:
            filename: Output filename (mp4, gif, etc.)
            dpi: Resolution
            
        Returns:
            True if successful
        """
        if self.animation is None:
            logger.error("No animation to save")
            return False
        
        try:
            writer = None
            if filename.endswith('.gif'):
                writer = 'pillow'
            elif filename.endswith('.mp4'):
                writer = 'ffmpeg'
            
            self.animation.save(filename, writer=writer, dpi=dpi)
            logger.info(f"Animation saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save animation: {e}")
            return False
