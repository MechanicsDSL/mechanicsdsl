import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from collections import deque
from typing import Optional
from .utils import logger, config, validate_file_path, validate_solution_dict, ANIMATION_INTERVAL_MS, TRAIL_ALPHA, PRIMARY_COLOR, SECONDARY_COLOR, TERTIARY_COLOR
from .energy import PotentialEnergyCalculator

class MechanicsVisualizer:
    def __init__(self, trail_length: int = None, fps: int = None):
        self.fig = None
        self.ax = None
        self.trail_length = trail_length or config.trail_length
        self.fps = fps or config.animation_fps

    def save_animation_to_file(self, anim, filename, fps=None, dpi=100):
        import shutil
        if not shutil.which('ffmpeg'):
            logger.warning("FFmpeg not found, cannot save animation.")
            return False
        try:
            fps = fps or self.fps
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(filename, writer=writer, dpi=dpi)
            return True
        except Exception as e:
            logger.error(f"Failed to save animation: {e}")
            return False

    def setup_3d_plot(self, title="Simulation"):
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)

    def animate(self, solution, parameters, system_name="system"):
        if not solution or not solution.get('success'): return None
        
        coords = solution['coordinates']
        t = solution['t']
        y = solution['y']
        name = system_name.lower()
        
        if 'pendulum' in name:
            if len(coords) >= 2 or 'double' in name:
                return self._animate_double_pendulum(t, y, parameters)
            return self._animate_single_pendulum(t, y, parameters)
        elif 'oscillator' in name:
            return self.animate_oscillator(t, y, system_name)
            
        return self._animate_phase_space(t, y, coords, system_name)

    def _animate_single_pendulum(self, t, y, params):
        self.setup_3d_plot("Single Pendulum")
        l = params.get('l', 1.0)
        theta = y[0]
        
        x = l * np.sin(theta)
        z_pos = -l * np.cos(theta)
        
        self.ax.set_xlim(-l*1.2, l*1.2)
        self.ax.set_zlim(-l*1.2, l*0.2)
        self.ax.set_ylim(-0.5, 0.5) # Flat 2D in 3D space
        
        line, = self.ax.plot([], [], [], 'o-', lw=3, color=PRIMARY_COLOR)
        trail, = self.ax.plot([], [], [], '-', alpha=TRAIL_ALPHA, color=SECONDARY_COLOR)
        
        history = deque(maxlen=self.trail_length)
        
        def update(i):
            if i >= len(t): return line, trail
            history.append((x[i], 0, z_pos[i]))
            
            line.set_data([0, x[i]], [0, 0])
            line.set_3d_properties([0, z_pos[i]])
            
            if len(history) > 1:
                hx, hy, hz = zip(*history)
                trail.set_data(hx, hy)
                trail.set_3d_properties(hz)
            return line, trail

        return animation.FuncAnimation(self.fig, update, frames=len(t), interval=ANIMATION_INTERVAL_MS, blit=False)

    def _animate_double_pendulum(self, t, y, params):
        self.setup_3d_plot("Double Pendulum")
        l1 = params.get('l1', 1.0)
        l2 = params.get('l2', 1.0)
        t1, t2 = y[0], y[2]
        
        x1 = l1 * np.sin(t1)
        z1 = -l1 * np.cos(t1)
        x2 = x1 + l2 * np.sin(t2)
        z2 = z1 - l2 * np.cos(t2)
        
        bound = l1 + l2 + 0.5
        self.ax.set_xlim(-bound, bound)
        self.ax.set_zlim(-bound, bound)
        self.ax.set_ylim(-1, 1)

        line, = self.ax.plot([], [], [], 'o-', lw=3, color=PRIMARY_COLOR)
        trail, = self.ax.plot([], [], [], '-', alpha=TRAIL_ALPHA, color=SECONDARY_COLOR)
        history = deque(maxlen=self.trail_length)
        
        def update(i):
            if i >= len(t): return line, trail
            history.append((x2[i], 0, z2[i]))
            
            line.set_data([0, x1[i], x2[i]], [0, 0, 0])
            line.set_3d_properties([0, z1[i], z2[i]])
            
            if len(history) > 1:
                hx, hy, hz = zip(*history)
                trail.set_data(hx, hy)
                trail.set_3d_properties(hz)
            return line, trail

        return animation.FuncAnimation(self.fig, update, frames=len(t), interval=ANIMATION_INTERVAL_MS, blit=False)

    def animate_oscillator(self, t, y, name):
        fig, ax = plt.subplots()
        ax.set_title(f"{name} - Position")
        x = y[0]
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(min(x)*1.1, max(x)*1.1)
        
        line, = ax.plot([], [], 'b-')
        dot, = ax.plot([], [], 'ro')
        
        def update(i):
            line.set_data(t[:i], x[:i])
            dot.set_data([t[i]], [x[i]])
            return line, dot
            
        return animation.FuncAnimation(fig, update, frames=len(t), interval=ANIMATION_INTERVAL_MS, blit=True)

    def _animate_phase_space(self, t, y, coords, name):
        fig, ax = plt.subplots()
        ax.set_title(f"{name} - Phase Space")
        q, v = y[0], y[1]
        
        ax.set_xlabel(coords[0])
        ax.set_ylabel(f"d{coords[0]}/dt")
        ax.set_xlim(min(q)*1.1, max(q)*1.1)
        ax.set_ylim(min(v)*1.1, max(v)*1.1)
        
        line, = ax.plot([], [], 'k-', alpha=0.7)
        head, = ax.plot([], [], 'ro')
        
        def update(i):
            line.set_data(q[:i], v[:i])
            head.set_data([q[i]], [v[i]])
            return line, head
            
        return animation.FuncAnimation(fig, update, frames=len(t), interval=ANIMATION_INTERVAL_MS, blit=True)
    
    def plot_energy(self, solution, params, system_name, lagrangian=None):
        ke = PotentialEnergyCalculator.compute_kinetic_energy(solution, params)
        pe = PotentialEnergyCalculator.compute_potential_energy(solution, params, system_name)
        total = ke + pe
        
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        t = solution['t']
        
        ax1.plot(t, ke, label='Kinetic')
        ax1.plot(t, pe, label='Potential')
        ax1.plot(t, total, 'k--', label='Total')
        ax1.set_ylabel('Energy')
        ax1.legend()
        ax1.set_title('Energy Conservation')
        
        # Relative Error
        err = np.abs((total - total[0]) / total[0])
        ax2.plot(t, err, 'r-')
        ax2.set_ylabel('Rel. Error')
        ax2.set_xlabel('Time')
        
        plt.tight_layout()
        plt.show()
    
    def plot_phase_space(self, solution, idx=0):
        y = solution['y']
        q = y[2*idx]
        v = y[2*idx+1]
        coord = solution['coordinates'][idx]
        
        plt.figure()
        plt.plot(q, v, 'k-', lw=1)
        plt.xlabel(coord)
        plt.ylabel(f"{coord}_dot")
        plt.title(f"Phase Space: {coord}")
        plt.grid(True, alpha=0.3)
        plt.show()
