import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class MechanicsVisualizer:
    def animate(self, solution, params, name):
        # Basic Phase Space Plotter for generic systems
        t = solution['t']
        y = solution['y']
        
        fig, ax = plt.subplots()
        ax.set_title(f"{name} - Phase Space")
        line, = ax.plot([], [], 'b-')
        
        q = y[0]
        q_dot = y[1]
        
        ax.set_xlim(np.min(q), np.max(q))
        ax.set_ylim(np.min(q_dot), np.max(q_dot))
        
        def update(frame):
            line.set_data(q[:frame], q_dot[:frame])
            return line,
            
        return animation.FuncAnimation(fig, update, frames=len(t), blit=True, interval=30)
