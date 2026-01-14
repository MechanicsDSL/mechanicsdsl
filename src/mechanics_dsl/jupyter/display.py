"""
Display utilities for Jupyter notebooks.

Provides interactive visualizations of simulation results.
"""
import numpy as np
from typing import Dict, List, Any, Optional

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML, display as ipython_display
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import ipywidgets as widgets
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False


def display_simulation(
    solution: Dict[str, Any],
    coordinates: Optional[List[str]] = None,
    fps: int = 30,
    trail_length: int = 50,
    figsize: tuple = (8, 6),
) -> Optional[Any]:
    """
    Display animated simulation in Jupyter notebook.
    
    Args:
        solution: Simulation result dictionary
        coordinates: Coordinate names
        fps: Frames per second
        trail_length: Number of trailing points
        figsize: Figure size
        
    Returns:
        HTML animation or None
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available for animation")
        return None
    
    if not solution.get('success'):
        print("Simulation was not successful")
        return None
    
    t = solution['t']
    y = solution['y']
    
    if coordinates is None:
        coordinates = solution.get('coordinates', [f'q{i}' for i in range(y.shape[0]//2)])
    
    n_coords = len(coordinates)
    
    # Create figure
    fig, axes = plt.subplots(1, n_coords, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    lines = []
    trails = []
    
    for i, (ax, coord) in enumerate(zip(axes, coordinates)):
        ax.set_xlim(t[0], t[-1])
        y_data = y[2*i]
        ax.set_ylim(np.min(y_data)*1.2, np.max(y_data)*1.2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(coord)
        ax.set_title(f'{coord}(t)')
        ax.grid(True, alpha=0.3)
        
        line, = ax.plot([], [], 'b-', lw=2)
        trail, = ax.plot([], [], 'b.', alpha=0.3, markersize=3)
        lines.append(line)
        trails.append(trail)
    
    plt.tight_layout()
    
    def init():
        for line, trail in zip(lines, trails):
            line.set_data([], [])
            trail.set_data([], [])
        return lines + trails
    
    def animate(frame):
        idx = min(frame, len(t) - 1)
        
        for i, (line, trail) in enumerate(zip(lines, trails)):
            line.set_data(t[:idx+1], y[2*i, :idx+1])
            
            # Trail
            start = max(0, idx - trail_length)
            trail.set_data(t[start:idx+1], y[2*i, start:idx+1])
        
        return lines + trails
    
    interval = 1000 // fps
    n_frames = len(t)
    
    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=n_frames, interval=interval, blit=True
    )
    
    plt.close(fig)  # Prevent static display
    
    html = HTML(anim.to_jshtml())
    ipython_display(html)
    return html


def display_phase_portrait(
    solution: Dict[str, Any],
    coord_idx: int = 0,
    figsize: tuple = (6, 6),
) -> Optional[Any]:
    """
    Display phase portrait (q vs q_dot).
    
    Args:
        solution: Simulation result
        coord_idx: Which coordinate to plot
        figsize: Figure size
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return None
    
    if not solution.get('success'):
        print("Simulation was not successful")
        return None
    
    y = solution['y']
    coordinates = solution.get('coordinates', ['q'])
    
    q = y[2*coord_idx]
    q_dot = y[2*coord_idx + 1]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color by time
    t = solution['t']
    colors = plt.cm.viridis(np.linspace(0, 1, len(t)))
    
    for i in range(len(t) - 1):
        ax.plot(q[i:i+2], q_dot[i:i+2], color=colors[i], lw=1)
    
    ax.scatter([q[0]], [q_dot[0]], c='green', s=100, zorder=5, label='Start')
    ax.scatter([q[-1]], [q_dot[-1]], c='red', s=100, zorder=5, label='End')
    
    coord_name = coordinates[coord_idx] if coord_idx < len(coordinates) else 'q'
    ax.set_xlabel(coord_name)
    ax.set_ylabel(f'd{coord_name}/dt')
    ax.set_title('Phase Portrait')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def display_energy_plot(
    solution: Dict[str, Any],
    figsize: tuple = (10, 4),
) -> Optional[Any]:
    """
    Display energy over time.
    
    Args:
        solution: Simulation result
        figsize: Figure size
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return None
    
    if not solution.get('success'):
        print("Simulation was not successful")
        return None
    
    t = solution['t']
    y = solution['y']
    
    # Compute kinetic energy (1/2 * sum of velocities^2)
    n_coords = y.shape[0] // 2
    KE = np.zeros(len(t))
    for i in range(n_coords):
        KE += 0.5 * y[2*i + 1]**2
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(t, KE, 'b-', lw=2, label='Kinetic Energy')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy')
    ax.set_title('Energy vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def create_parameter_sliders(
    compiler,
    param_names: Optional[List[str]] = None,
    on_change: Optional[callable] = None
):
    """
    Create interactive parameter sliders.
    
    Requires ipywidgets.
    
    Args:
        compiler: PhysicsCompiler instance
        param_names: Parameters to create sliders for
        on_change: Callback when slider changes
        
    Returns:
        Widget container
    """
    if not WIDGETS_AVAILABLE:
        print("ipywidgets not available")
        return None
    
    params = compiler.simulator.parameters
    if param_names is None:
        param_names = list(params.keys())
    
    sliders = {}
    
    for name in param_names:
        if name not in params:
            continue
        
        val = params[name]
        slider = widgets.FloatSlider(
            value=val,
            min=val * 0.1,
            max=val * 10,
            step=val * 0.01,
            description=name,
            continuous_update=False
        )
        sliders[name] = slider
        
        def make_callback(param_name):
            def callback(change):
                compiler.simulator.set_parameters({param_name: change['new']})
                if on_change:
                    on_change()
            return callback
        
        slider.observe(make_callback(name), names='value')
    
    container = widgets.VBox(list(sliders.values()))
    return container


__all__ = [
    'display_simulation',
    'display_phase_portrait',
    'display_energy_plot',
    'create_parameter_sliders',
]
