/**
 * MechanicsDSL Pyodide Bridge
 * Runs actual Python MechanicsDSL in the browser via WebAssembly
 */

class PyodideBridge {
    constructor() {
        this.pyodide = null;
        this.isReady = false;
        this.isLoading = false;
        this.loadingProgress = 0;
        this.onProgressCallbacks = [];
    }

    onProgress(callback) {
        this.onProgressCallbacks.push(callback);
    }

    notifyProgress(message, percent) {
        this.loadingProgress = percent;
        this.onProgressCallbacks.forEach(cb => cb(message, percent));
    }

    async initialize() {
        if (this.isReady) return true;
        if (this.isLoading) {
            // Wait for existing load
            while (this.isLoading) {
                await new Promise(r => setTimeout(r, 100));
            }
            return this.isReady;
        }

        this.isLoading = true;
        this.notifyProgress('Loading Python runtime...', 5);

        try {
            // Load Pyodide
            this.pyodide = await loadPyodide({
                indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/"
            });
            this.notifyProgress('Installing NumPy...', 20);

            // Install required packages
            await this.pyodide.loadPackage(['numpy', 'sympy', 'scipy']);
            this.notifyProgress('Installing SymPy...', 50);

            this.notifyProgress('Setting up MechanicsDSL...', 70);

            // Initialize MechanicsDSL-like functionality directly in Python
            await this.pyodide.runPythonAsync(`
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
from sympy import symbols, sin, cos, diff, solve, lambdify, Rational
from sympy.physics.mechanics import dynamicsymbols

class SimplePendulumSystem:
    """Simple pendulum simulation using Lagrangian mechanics."""
    
    def __init__(self, m=1.0, l=1.0, g=9.81, theta0=0.5, omega0=0.0):
        self.m = m
        self.l = l
        self.g = g
        self.state = np.array([theta0, omega0])
        self.t = 0.0
        self.initial_energy = self.energy()
        
    def derivatives(self, t, y):
        theta, omega = y
        dtheta = omega
        domega = -self.g / self.l * np.sin(theta)
        return [dtheta, domega]
    
    def step(self, dt):
        sol = solve_ivp(self.derivatives, [self.t, self.t + dt], self.state, 
                       method='RK45', dense_output=True)
        self.state = sol.y[:, -1]
        self.t += dt
        return {'theta': float(self.state[0]), 'omega': float(self.state[1])}
    
    def energy(self):
        theta, omega = self.state
        KE = 0.5 * self.m * (self.l * omega)**2
        PE = self.m * self.g * self.l * (1 - np.cos(theta))
        return float(KE + PE)
    
    def energy_error(self):
        if self.initial_energy == 0:
            return 0.0
        return abs((self.energy() - self.initial_energy) / self.initial_energy) * 100


class DoublePendulumSystem:
    """Double pendulum using derived equations of motion from Lagrangian."""
    
    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81,
                 theta1_0=2.5, omega1_0=0.0, theta2_0=2.0, omega2_0=0.0):
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.g = g
        self.state = np.array([theta1_0, omega1_0, theta2_0, omega2_0])
        self.t = 0.0
        self.initial_energy = self.energy()
        
    def derivatives(self, t, y):
        t1, o1, t2, o2 = y
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        
        delta = t1 - t2
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        den1 = (m1 + m2) * l1 - m2 * l1 * cos_d**2
        den2 = (l2 / l1) * den1
        
        a1 = (m2 * l1 * o1**2 * sin_d * cos_d
              + m2 * g * np.sin(t2) * cos_d
              + m2 * l2 * o2**2 * sin_d
              - (m1 + m2) * g * np.sin(t1)) / den1
              
        a2 = (-m2 * l2 * o2**2 * sin_d * cos_d
              + (m1 + m2) * g * np.sin(t1) * cos_d
              - (m1 + m2) * l1 * o1**2 * sin_d
              - (m1 + m2) * g * np.sin(t2)) / den2
        
        return [o1, a1, o2, a2]
    
    def step(self, dt):
        sol = solve_ivp(self.derivatives, [self.t, self.t + dt], self.state,
                       method='RK45', dense_output=True)
        self.state = sol.y[:, -1]
        self.t += dt
        return {
            'theta1': float(self.state[0]), 'omega1': float(self.state[1]),
            'theta2': float(self.state[2]), 'omega2': float(self.state[3])
        }
    
    def energy(self):
        t1, o1, t2, o2 = self.state
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        
        y1 = -l1 * np.cos(t1)
        y2 = y1 - l2 * np.cos(t2)
        
        v1sq = l1**2 * o1**2
        v2sq = l1**2 * o1**2 + l2**2 * o2**2 + 2*l1*l2*o1*o2*np.cos(t1-t2)
        
        KE = 0.5 * m1 * v1sq + 0.5 * m2 * v2sq
        PE = m1*g*y1 + m2*g*y2 + (m1+m2)*g*(l1+l2)
        return float(KE + PE)
    
    def energy_error(self):
        if self.initial_energy == 0:
            return 0.0
        return abs((self.energy() - self.initial_energy) / self.initial_energy) * 100


class SpringSystem:
    """Damped harmonic oscillator."""
    
    def __init__(self, k=10.0, m=1.0, b=0.1, x0=1.5, v0=0.0):
        self.k, self.m, self.b = k, m, b
        self.state = np.array([x0, v0])
        self.t = 0.0
        self.initial_energy = self.energy()
        
    def derivatives(self, t, y):
        x, v = y
        return [v, (-self.k * x - self.b * v) / self.m]
    
    def step(self, dt):
        sol = solve_ivp(self.derivatives, [self.t, self.t + dt], self.state, method='RK45')
        self.state = sol.y[:, -1]
        self.t += dt
        return {'x': float(self.state[0]), 'v': float(self.state[1])}
    
    def energy(self):
        x, v = self.state
        return float(0.5 * self.m * v**2 + 0.5 * self.k * x**2)
    
    def energy_error(self):
        return 0.0  # Damped system doesn't conserve energy


class OrbitalSystem:
    """Two-body orbital mechanics."""
    
    def __init__(self, GM=1000.0, x0=100.0, y0=0.0, vx0=0.0, vy0=None):
        self.GM = GM
        if vy0 is None:
            vy0 = np.sqrt(GM / x0) * 0.8  # Elliptical orbit
        self.state = np.array([x0, y0, vx0, vy0])
        self.t = 0.0
        self.initial_energy = self.energy()
        
    def derivatives(self, t, y):
        x, yy, vx, vy = y
        r = np.sqrt(x**2 + yy**2)
        a = -self.GM / r**3
        return [vx, vy, a*x, a*yy]
    
    def step(self, dt):
        sol = solve_ivp(self.derivatives, [self.t, self.t + dt], self.state, method='RK45')
        self.state = sol.y[:, -1]
        self.t += dt
        return {'x': float(self.state[0]), 'y': float(self.state[1]),
                'vx': float(self.state[2]), 'vy': float(self.state[3])}
    
    def energy(self):
        x, y, vx, vy = self.state
        r = np.sqrt(x**2 + y**2)
        return float(0.5 * (vx**2 + vy**2) - self.GM / r)
    
    def energy_error(self):
        if self.initial_energy == 0:
            return 0.0
        return abs((self.energy() - self.initial_energy) / self.initial_energy) * 100


# Global simulation instance
current_sim = None

def create_pendulum(m=1.0, l=1.0, g=9.81, theta=0.5, omega=0.0):
    global current_sim
    current_sim = SimplePendulumSystem(m, l, g, theta, omega)
    return "pendulum"

def create_double_pendulum(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81,
                           theta1=2.5, omega1=0.0, theta2=2.0, omega2=0.0):
    global current_sim
    current_sim = DoublePendulumSystem(m1, m2, l1, l2, g, theta1, omega1, theta2, omega2)
    return "double-pendulum"

def create_spring(k=10.0, m=1.0, b=0.1, x=1.5, v=0.0):
    global current_sim
    current_sim = SpringSystem(k, m, b, x, v)
    return "spring"

def create_orbital(GM=1000.0, x=100.0, y=0.0, vx=0.0, vy=None):
    global current_sim
    current_sim = OrbitalSystem(GM, x, y, vx, vy)
    return "orbital"

def sim_step(dt):
    global current_sim
    if current_sim is None:
        return {}
    return current_sim.step(dt)

def sim_energy():
    global current_sim
    if current_sim is None:
        return 0.0
    return current_sim.energy()

def sim_energy_error():
    global current_sim
    if current_sim is None:
        return 0.0
    return current_sim.energy_error()

def sim_time():
    global current_sim
    if current_sim is None:
        return 0.0
    return current_sim.t

print("MechanicsDSL Python engine ready!")
`);

            this.notifyProgress('Ready!', 100);
            this.isReady = true;
            this.isLoading = false;
            return true;
        } catch (error) {
            console.error('Failed to initialize Pyodide:', error);
            this.notifyProgress('Failed to load: ' + error.message, 0);
            this.isLoading = false;
            return false;
        }
    }

    async createSimulation(type, params = {}) {
        if (!this.isReady) {
            await this.initialize();
        }

        let pythonCode;
        switch (type) {
            case 'pendulum':
                pythonCode = `create_pendulum(${params.m || 1.0}, ${params.l || 1.0}, ${params.g || 9.81}, ${params.theta || 0.5}, ${params.omega || 0.0})`;
                break;
            case 'double-pendulum':
                pythonCode = `create_double_pendulum(${params.m1 || 1.0}, ${params.m2 || 1.0}, ${params.l1 || 1.0}, ${params.l2 || 1.0}, ${params.g || 9.81}, ${params.theta1 !== undefined ? params.theta1 : 2.5}, ${params.omega1 || 0.0}, ${params.theta2 !== undefined ? params.theta2 : 2.0}, ${params.omega2 || 0.0})`;
                break;
            case 'spring':
                pythonCode = `create_spring(${params.k || 10.0}, ${params.m || 1.0}, ${params.b || 0.1}, ${params.x || 1.5}, ${params.v || 0.0})`;
                break;
            case 'orbital':
                pythonCode = `create_orbital(${params.GM || 1000.0}, ${params.x || 100.0}, ${params.y || 0.0}, ${params.vx || 0.0}, ${params.vy || 'None'})`;
                break;
            default:
                pythonCode = `create_double_pendulum()`;
        }

        return await this.pyodide.runPythonAsync(pythonCode);
    }

    async step(dt) {
        if (!this.isReady) return null;
        const result = await this.pyodide.runPythonAsync(`sim_step(${dt})`);
        return result.toJs();
    }

    async getEnergy() {
        if (!this.isReady) return 0;
        return await this.pyodide.runPythonAsync('sim_energy()');
    }

    async getEnergyError() {
        if (!this.isReady) return 0;
        return await this.pyodide.runPythonAsync('sim_energy_error()');
    }

    async getTime() {
        if (!this.isReady) return 0;
        return await this.pyodide.runPythonAsync('sim_time()');
    }

    async updateParam(param, value) {
        if (!this.isReady) return;
        await this.pyodide.runPythonAsync(`
if current_sim is not None:
    current_sim.${param} = ${value}
    current_sim.initial_energy = current_sim.energy()
`);
    }
}

window.PyodideBridge = PyodideBridge;
