"""
Visualization engine for MechanicsDSL
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
from typing import Dict, Optional
import sympy as sp

from .utils import (
    logger, config, validate_solution_dict, validate_file_path,
    validate_array_safe, ANIMATION_INTERVAL_MS, TRAIL_ALPHA,
    PRIMARY_COLOR, SECONDARY_COLOR, TERTIARY_COLOR
)
from .energy import PotentialEnergyCalculator

class MechanicsVisualizer:
    """Enhanced visualization with circular buffers and configurable options"""
    
    def __init__(self, trail_length: int = None, fps: int = None):
        self.fig = None
        self.ax = None
        self.animation = None
        self.trail_length = trail_length or config.trail_length
        self.fps = fps or config.animation_fps
        logger.debug(f"Visualizer initialized: trail_length={self.trail_length}, fps={self.fps}")

    def has_ffmpeg(self) -> bool:
        """Check if ffmpeg is available"""
        import shutil
        return shutil.which('ffmpeg') is not None

    def save_animation_to_file(self, anim: animation.FuncAnimation, 
                               filename: str, fps: int = None, dpi: int = 100) -> bool:
        """
        Save animation to file with validation.
        
        Args:
            anim: Animation object to save
            filename: Output filename (validated)
            fps: Frames per second (optional)
            dpi: Dots per inch (default: 100)
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            TypeError: If inputs have wrong types
            ValueError: If filename is invalid or parameters out of range
        """
        if anim is None:
            raise ValueError("anim cannot be None")
        
        validate_file_path(filename, must_exist=False)
        
        if fps is not None:
            if not isinstance(fps, int):
                raise TypeError(f"fps must be int, got {type(fps).__name__}")
            if fps < 1 or fps > 120:
                raise ValueError(f"fps must be in [1, 120], got {fps}")
        
        if not isinstance(dpi, int):
            raise TypeError(f"dpi must be int, got {type(dpi).__name__}")
        if dpi < 10 or dpi > 1000:
            raise ValueError(f"dpi must be in [10, 1000], got {dpi}")

        fps = fps or self.fps

        try:
            if filename.lower().endswith('.mp4') and self.has_ffmpeg():
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=fps, metadata=dict(artist='MechanicsDSL'), bitrate=1800)
                anim.save(filename, writer=writer, dpi=dpi)
                logger.info(f"Animation saved to {filename}")
                return True
            elif filename.lower().endswith('.gif'):
                anim.save(filename, writer='pillow', fps=fps)
                logger.info(f"Animation saved to {filename}")
                return True
            else:
                if self.has_ffmpeg():
                    Writer = animation.writers['ffmpeg']
                    writer = Writer(fps=fps, metadata=dict(artist='MechanicsDSL'), bitrate=1800)
                    anim.save(filename, writer=writer, dpi=dpi)
                    logger.info(f"Animation saved to {filename}")
                    return True
        except (IOError, OSError, PermissionError, ValueError, AttributeError) as e:
            logger.error(f"Failed to save animation: {e}")
        except Exception as e:
            logger.error(f"Unexpected error saving animation: {type(e).__name__}: {e}")
        
        return False

    def setup_3d_plot(self, title: str = "Classical Mechanics Simulation"):
        """Setup 3D plotting environment"""
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title(title, fontsize=16, fontweight='bold')
        self.ax.set_xlabel('X (m)', fontsize=12)
        self.ax.set_ylabel('Y (m)', fontsize=12)
        self.ax.set_zlabel('Z (m)', fontsize=12)
        self.ax.grid(True, alpha=0.3)

    def animate_pendulum(self, solution: dict, parameters: dict, system_name: str = "pendulum"):
        """
        Create animated pendulum visualization with validation.
        
        Args:
            solution: Solution dictionary (validated)
            parameters: System parameters dictionary
            system_name: Name of the system
            
        Returns:
            Animation object or None if failed
            
        Raises:
            TypeError: If inputs have wrong types
            ValueError: If solution is invalid
        """
        if not isinstance(parameters, dict):
            raise TypeError(f"parameters must be dict, got {type(parameters).__name__}")
        if not isinstance(system_name, str):
            raise TypeError(f"system_name must be str, got {type(system_name).__name__}")
        
        if not isinstance(solution, dict) or not solution.get('success', False):
            logger.warning("Cannot animate failed simulation")
            return None
        
        validate_solution_dict(solution)
            
        self.setup_3d_plot(f"{system_name.title()} Animation")
        
        t = solution['t']
        y = solution['y']
        coordinates = solution['coordinates']
        
        name = (system_name or '').lower()
        
        if len(coordinates) >= 2 or 'double' in name:
            return self._animate_double_pendulum(t, y, parameters)
        else:
            return self._animate_single_pendulum(t, y, parameters)
    
    def _animate_single_pendulum(self, t: np.ndarray, y: np.ndarray, parameters: dict):
        """Animate single pendulum with circular buffer"""
        if y.shape[0] < 1:
            logger.error("Insufficient state vector for single pendulum animation")
            return None
        theta = y[0]
        l = parameters.get('l', 1.0)
        
        x = l * np.sin(theta)
        y_pos = -l * np.cos(theta)
        z = np.zeros_like(x)
        
        self.ax.set_xlim(-l*1.2, l*1.2)
        self.ax.set_ylim(-l*1.2, l*0.2)
        self.ax.set_zlim(-0.1, 0.1)
        
        line, = self.ax.plot([], [], [], 'o-', linewidth=3, markersize=10, 
                            color=PRIMARY_COLOR, label='Pendulum')
        trail, = self.ax.plot([], [], [], '-', alpha=TRAIL_ALPHA, linewidth=1.5, 
                             color=SECONDARY_COLOR, label='Trail')
        time_text = self.ax.text2D(0.05, 0.95, '', transform=self.ax.transAxes, fontsize=12)
        
        self.ax.legend(loc='upper right')
        
        # Circular buffer for trail
        trail_buffer = deque(maxlen=self.trail_length)
        
        def animate_frame(frame):
            if frame < len(t):
                # Update pendulum position
                line.set_data([0, x[frame]], [0, y_pos[frame]])
                line.set_3d_properties([0, z[frame]])
                
                # Update trail using circular buffer
                trail_buffer.append((x[frame], y_pos[frame], z[frame]))
                if len(trail_buffer) > 1:
                    trail_x, trail_y, trail_z = zip(*trail_buffer)
                    trail.set_data(trail_x, trail_y)
                    trail.set_3d_properties(trail_z)
                
                time_text.set_text(f'Time: {t[frame]:.2f} s')
                
            return line, trail, time_text
        
        interval = ANIMATION_INTERVAL_MS
        self.animation = animation.FuncAnimation(
            self.fig, animate_frame, frames=len(t),
            interval=interval, blit=False, repeat=True
        )
        
        logger.info("Single pendulum animation created")
        return self.animation
    
    def _animate_double_pendulum(self, t: np.ndarray, y: np.ndarray, parameters: dict):
        """Animate double pendulum with circular buffers"""
        if y.shape[0] < 1:
            logger.error("Insufficient state vector for double pendulum animation")
            return None
        theta1 = y[0]
        theta2 = y[2] if y.shape[0] >= 4 else y[0]
        
        l1 = parameters.get('l1', parameters.get('l', 1.0))
        l2 = parameters.get('l2', 1.0)
        
        x1 = l1 * np.sin(theta1)
        y1 = -l1 * np.cos(theta1)
        x2 = x1 + l2 * np.sin(theta2)
        y2 = y1 - l2 * np.cos(theta2)
        
        max_reach = l1 + l2
        self.ax.set_xlim(-max_reach*1.1, max_reach*1.1)
        self.ax.set_ylim(-max_reach*1.1, max_reach*0.2)
        self.ax.set_zlim(-0.1, 0.1)
        
        line1, = self.ax.plot([], [], [], 'o-', linewidth=3, markersize=10, 
                             color=PRIMARY_COLOR, label='Pendulum 1')
        line2, = self.ax.plot([], [], [], 'o-', linewidth=3, markersize=10, 
                             color=TERTIARY_COLOR, label='Pendulum 2')
        trail1, = self.ax.plot([], [], [], '-', alpha=0.3, linewidth=1, color=PRIMARY_COLOR)
        trail2, = self.ax.plot([], [], [], '-', alpha=TRAIL_ALPHA, linewidth=1.5, color=SECONDARY_COLOR)
        time_text = self.ax.text2D(0.05, 0.95, '', transform=self.ax.transAxes, fontsize=12)
        
        self.ax.legend(loc='upper right')
        
        # Circular buffers for trails
        trail_buffer1 = deque(maxlen=self.trail_length)
        trail_buffer2 = deque(maxlen=self.trail_length)
        
        def animate_frame(frame):
            if frame < len(t):
                line1.set_data([0, x1[frame]], [0, y1[frame]])
                line1.set_3d_properties([0, 0])
                
                line2.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
                line2.set_3d_properties([0, 0])
                
                # Update trails using circular buffers
                trail_buffer1.append((x1[frame], y1[frame], 0))
                trail_buffer2.append((x2[frame], y2[frame], 0))
                
                if len(trail_buffer1) > 1:
                    t1_x, t1_y, t1_z = zip(*trail_buffer1)
                    trail1.set_data(t1_x, t1_y)
                    trail1.set_3d_properties(t1_z)
                
                if len(trail_buffer2) > 1:
                    t2_x, t2_y, t2_z = zip(*trail_buffer2)
                    trail2.set_data(t2_x, t2_y)
                    trail2.set_3d_properties(t2_z)
                
                time_text.set_text(f'Time: {t[frame]:.2f} s')
                
            return line1, line2, trail1, trail2, time_text
        
        interval = ANIMATION_INTERVAL_MS
        self.animation = animation.FuncAnimation(
            self.fig, animate_frame, frames=len(t),
            interval=interval, blit=False, repeat=True
        )
        
        logger.info("Double pendulum animation created")
        return self.animation

    def animate_fluid_from_csv(self, csv_filename: str, title: str = "Fluid Simulation"):
        """
        Animate SPH particle data from CSV.
        Expected CSV Format: t, id, x, y, rho
        """
        import pandas as pd
        
        if not validate_file_path(csv_filename, must_exist=True):
            return None
            
        logger.info(f"Loading fluid data from {csv_filename}...")
        try:
            df = pd.read_csv(csv_filename)
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            return None

        # Get unique time steps
        times = df['t'].unique()
        logger.info(f"Found {len(times)} frames for {len(df[df['t']==times[0]])} particles")
        
        # Setup Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(title, fontsize=16)
        ax.set_xlim(df['x'].min()-0.1, df['x'].max()+0.1)
        ax.set_ylim(df['y'].min()-0.1, df['y'].max()+0.1)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.3)
        
        # Scatter plot for particles
        # We use a colormap 'coolwarm' mapped to density (rho)
        scatter = ax.scatter([], [], c=[], cmap='coolwarm', s=10, vmin=900, vmax=1100)
        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label('Density (kg/m^3)')
        
        time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12)
        
        def animate(frame_idx):
            t_val = times[frame_idx]
            frame_data = df[df['t'] == t_val]
            
            # Update positions and colors
            scatter.set_offsets(frame_data[['x', 'y']].values)
            scatter.set_array(frame_data['rho'].values)
            
            time_text.set_text(f"Time: {t_val:.3f}s")
            return scatter, time_text

        self.animation = animation.FuncAnimation(
            fig, animate, frames=len(times), interval=30, blit=False
        )
        
        return self.animation

    def animate_oscillator(self, solution: dict, parameters: dict, system_name: str = "oscillator"):
        """Animate harmonic oscillator"""
        
        if not solution['success']:
            logger.warning("Cannot animate failed simulation")
            return None
        
        t = solution['t']
        y = solution['y']
        
        if y.shape[0] < 1:
            logger.error("Insufficient state vector for oscillator animation")
            return None
        x = y[0]
        v = y[1] if y.shape[0] > 1 else np.zeros_like(x)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.set_xlim(t[0], t[-1])
        ax1.set_ylim(np.min(x)*1.2, np.max(x)*1.2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position (m)')
        ax1.set_title(f'{system_name.title()} - Position vs Time')
        ax1.grid(True, alpha=0.3)
        
        line1, = ax1.plot([], [], 'b-', linewidth=2, label='Position')
        point1, = ax1.plot([], [], 'ro', markersize=8)
        ax1.legend()
        
        ax2.set_xlim(np.min(x)*1.2, np.max(x)*1.2)
        ax2.set_ylim(np.min(v)*1.2, np.max(v)*1.2)
        ax2.set_xlabel('Position (m)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Phase Space')
        ax2.grid(True, alpha=0.3)
        
        line2, = ax2.plot([], [], 'g-', linewidth=1.5, alpha=0.6, label='Trajectory')
        point2, = ax2.plot([], [], 'ro', markersize=8)
        ax2.legend()
        
        def init():
            line1.set_data([], [])
            point1.set_data([], [])
            line2.set_data([], [])
            point2.set_data([], [])
            return line1, point1, line2, point2
        
        def animate_frame(frame):
            if frame < len(t):
                line1.set_data(t[:frame], x[:frame])
                point1.set_data([t[frame]], [x[frame]])
                
                line2.set_data(x[:frame], v[:frame])
                point2.set_data([x[frame]], [v[frame]])
            
            return line1, point1, line2, point2
        
        interval = ANIMATION_INTERVAL_MS
        self.animation = animation.FuncAnimation(
            fig, animate_frame, frames=len(t), init_func=init,
            interval=interval, blit=True, repeat=True
        )
        
        self.fig = fig
        self.ax = ax1
        
        logger.info("Oscillator animation created")
        return self.animation

    def animate(self, solution: dict, parameters: dict, system_name: str = "system"):
        """Generic animation dispatcher"""
        if not solution or not solution.get('success'):
            logger.warning("Cannot animate: invalid solution")
            return None

        coords = solution.get('coordinates', [])
        name = (system_name or '').lower()

        try:
            if 'pendulum' in name or any('theta' in c for c in coords):
                return self.animate_pendulum(solution, parameters, system_name)
            elif 'oscillator' in name or 'spring' in name or (len(coords) == 1 and 'x' in coords):
                return self.animate_oscillator(solution, parameters, system_name)
            else:
                return self._animate_phase_space(solution, system_name)
                
        except Exception as e:
            logger.error(f"Animation failed: {e}")
            return None
    
    def _animate_phase_space(self, solution: dict, system_name: str):
        """Generic phase space animation"""
        t = solution['t']
        y = solution['y']
        coords = solution['coordinates']
        
        if len(coords) == 0:
            logger.warning("No coordinates to animate")
            return None
        
        if y.shape[0] < 1:
            logger.error("Insufficient state vector for phase space animation")
            return None
        q = y[0]
        q_dot = y[1] if y.shape[0] > 1 else np.zeros_like(q)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f'{system_name} - Phase Space')
        ax.set_xlabel(f'{coords[0]}')
        ax.set_ylabel(f'{coords[0]}_dot')
        ax.grid(True, alpha=0.3)
        
        line, = ax.plot([], [], 'b-', linewidth=1.5, alpha=0.6)
        point, = ax.plot([], [], 'ro', markersize=8)
        time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12)
        
        def init():
            ax.set_xlim(np.min(q)*1.1, np.max(q)*1.1)
            ax.set_ylim(np.min(q_dot)*1.1, np.max(q_dot)*1.1)
            line.set_data([], [])
            point.set_data([], [])
            return line, point, time_text
        
        def animate_frame(frame):
            if frame < len(t):
                line.set_data(q[:frame], q_dot[:frame])
                point.set_data([q[frame]], [q_dot[frame]])
                time_text.set_text(f'Time: {t[frame]:.2f} s')
            return line, point, time_text
        
        interval = ANIMATION_INTERVAL_MS
        self.animation = animation.FuncAnimation(
            fig, animate_frame, frames=len(t), init_func=init,
            interval=interval, blit=True, repeat=True
        )
        
        self.fig = fig
        self.ax = ax
        
        logger.info("Phase space animation created")
        return self.animation

    def plot_energy(self, solution: dict, parameters: dict, system_name: str = "",
                   lagrangian: sp.Expr = None):
        """Plot energy conservation analysis with proper offset correction"""
        
        if not solution['success']:
            logger.warning("Cannot plot energy for failed simulation")
            return
        
        t = solution['t']
        
        # Use the new energy calculator
        KE = PotentialEnergyCalculator.compute_kinetic_energy(solution, parameters)
        PE = PotentialEnergyCalculator.compute_potential_energy(solution, parameters, system_name)
        E_total = KE + PE
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Energy Analysis', fontsize=16, fontweight='bold')
        
        axes[0, 0].plot(t, KE, 'r-', linewidth=2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Energy (J)')
        axes[0, 0].set_title('Kinetic Energy')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(t, PE, 'b-', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Energy (J)')
        axes[0, 1].set_title('Potential Energy')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(t, E_total, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Energy (J)')
        axes[1, 0].set_title('Total Energy')
        axes[1, 0].grid(True, alpha=0.3)
        
        E_error = (E_total - E_total[0]) / np.abs(E_total[0]) * 100 if E_total[0] != 0 else (E_total - E_total[0])
        axes[1, 1].plot(t, E_error, 'purple', linewidth=2)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Relative Error (%)')
        axes[1, 1].set_title('Energy Conservation Error')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        logger.info(f"\n{'='*50}")
        logger.info("Energy Conservation Analysis")
        logger.info(f"{'='*50}")
        logger.info(f"Initial Total Energy: {E_total[0]:.6f} J")
        logger.info(f"Final Total Energy:   {E_total[-1]:.6f} J")
        logger.info(f"Energy Change:        {E_total[-1] - E_total[0]:.6e} J")
        if E_total[0] != 0:
            logger.info(f"Relative Error:       {E_error[-1]:.6f}%")
            logger.info(f"Max Relative Error:   {np.max(np.abs(E_error)):.6f}%")
        logger.info(f"{'='*50}\n")

    def plot_phase_space(self, solution: dict, coordinate_index: int = 0):
        """
        Plot phase space trajectory with validation.
        
        Args:
            solution: Solution dictionary (validated)
            coordinate_index: Index of coordinate to plot (default: 0)
            
        Raises:
            TypeError: If inputs have wrong types
            ValueError: If solution is invalid or coordinate_index out of range
        """
        if not isinstance(coordinate_index, int):
            raise TypeError(f"coordinate_index must be int, got {type(coordinate_index).__name__}")
        if coordinate_index < 0:
            raise ValueError(f"coordinate_index must be non-negative, got {coordinate_index}")
        
        if not isinstance(solution, dict) or not solution.get('success', False):
            logger.warning("Cannot plot phase space for failed simulation")
            return
        
        validate_solution_dict(solution)
        
        y = solution['y']
        coords = solution['coordinates']
        
        if coordinate_index >= len(coords):
            raise ValueError(f"coordinate_index {coordinate_index} out of range [0, {len(coords)})")
        
        # Safe array access with validation
        pos_idx = 2 * coordinate_index
        vel_idx = 2 * coordinate_index + 1
        
        if pos_idx >= y.shape[0] or vel_idx >= y.shape[0]:
            raise ValueError(f"State vector too small: need indices {pos_idx} and {vel_idx}, got size {y.shape[0]}")
        
        position = y[pos_idx]
        velocity = y[vel_idx]
        
        # Validate arrays
        if not validate_array_safe(position, "position", check_finite=True):
            logger.warning("plot_phase_space: position array has issues, attempting to fix")
            position = np.nan_to_num(position, nan=0.0, posinf=1e10, neginf=-1e10)
        
        if not validate_array_safe(velocity, "velocity", check_finite=True):
            logger.warning("plot_phase_space: velocity array has issues, attempting to fix")
            velocity = np.nan_to_num(velocity, nan=0.0, posinf=1e10, neginf=-1e10)
        
        if len(position) == 0 or len(velocity) == 0:
            logger.error("plot_phase_space: empty position or velocity arrays")
            return
        
        if len(position) != len(velocity):
            logger.warning(f"plot_phase_space: position and velocity length mismatch ({len(position)} vs {len(velocity)})")
            min_len = min(len(position), len(velocity))
            position = position[:min_len]
            velocity = velocity[:min_len]
        
        plt.figure(figsize=(10, 10))
        try:
            plt.plot(position, velocity, 'b-', alpha=0.7, linewidth=1.5, label='Trajectory')
            if len(position) > 0:
                plt.plot(position[0], velocity[0], 'go', markersize=10, label='Start', zorder=5)
                plt.plot(position[-1], velocity[-1], 'ro', markersize=10, label='End', zorder=5)
        except Exception as e:
            logger.error(f"plot_phase_space: error plotting: {e}", exc_info=True)
            return
        
        plt.xlabel(f'{coords[coordinate_index]} (position)', fontsize=12)
        plt.ylabel(f'd{coords[coordinate_index]}/dt (velocity)', fontsize=12)
        plt.title(f'Phase Space: {coords[coordinate_index]}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()
        
        logger.info(f"Phase space plot created for {coords[coordinate_index]}")
