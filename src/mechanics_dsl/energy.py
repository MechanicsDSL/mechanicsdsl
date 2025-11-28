import numpy as np

class PotentialEnergyCalculator:
    @staticmethod
    def compute_kinetic_energy(solution, params):
        # Placeholder for simple kinetic energy T = 0.5 * m * v^2
        # In full version, this parses the T term from Lagrangian
        y = solution['y']
        # Assume generalized coordinates q, q_dot
        # KE = sum(0.5 * m * q_dot^2)
        m = params.get('m', 1.0)
        ke = np.zeros_like(solution['t'])
        
        # Heuristic for simple systems (oscillator/pendulum)
        # Real implementation would evaluate the symbolic T expression numerically
        num_coords = len(solution['coordinates'])
        for i in range(num_coords):
            v = y[2*i+1]
            ke += 0.5 * m * v**2
        return ke

    @staticmethod
    def compute_potential_energy(solution, params, system_type=""):
        # Placeholder for PE = 0.5 * k * x^2 (Spring) or m*g*l*(1-cos theta)
        # Real implementation evaluates V expression
        y = solution['y']
        t = solution['t']
        pe = np.zeros_like(t)
        
        if "oscillator" in system_type:
            k = params.get('k', 1.0)
            x = y[0]
            pe = 0.5 * k * x**2
        elif "pendulum" in system_type:
            m = params.get('m', 1.0)
            g = params.get('g', 9.81)
            l = params.get('l', 1.0)
            theta = y[0]
            pe = m * g * l * (1 - np.cos(theta))
            
        return pe
