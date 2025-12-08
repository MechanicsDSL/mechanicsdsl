/**
 * MechanicsDSL Pyodide Bridge
 * Runs actual MechanicsDSL Python package in the browser via WebAssembly
 */

class PyodideBridge {
    constructor() {
        this.pyodide = null;
        this.isReady = false;
        this.isLoading = false;
        this.loadingProgress = 0;
        this.onProgressCallbacks = [];
        this.compiledSystem = null;
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
            this.notifyProgress('Installing NumPy & SciPy...', 15);

            // Install required packages
            await this.pyodide.loadPackage(['numpy', 'scipy', 'sympy', 'micropip']);
            this.notifyProgress('Installing MechanicsDSL...', 50);

            // Install mechanicsdsl-core from PyPI
            await this.pyodide.runPythonAsync(`
import micropip
await micropip.install('mechanicsdsl-core')
`);
            this.notifyProgress('Setting up simulation engine...', 80);

            // Initialize the bridge code
            await this.pyodide.runPythonAsync(`
import numpy as np
from scipy.integrate import solve_ivp

# Try to import MechanicsDSL
try:
    from mechanics_dsl import PhysicsCompiler, NumericalSimulator, SymbolicEngine
    HAVE_DSL = True
    print("MechanicsDSL loaded successfully!")
except ImportError as e:
    print(f"MechanicsDSL import failed: {e}")
    HAVE_DSL = False

# Simulation state
current_sim = None
initial_energy = None
sim_time = 0.0

def parse_dsl_code(code):
    """Parse DSL code and return compiled system."""
    global HAVE_DSL
    if not HAVE_DSL:
        return None, "MechanicsDSL not available"
    
    try:
        compiler = PhysicsCompiler()
        system = compiler.compile(code)
        return system, None
    except Exception as e:
        return None, str(e)

def create_simulation_from_dsl(code, params=None):
    """Create simulation from DSL code."""
    global current_sim, initial_energy, sim_time, HAVE_DSL
    
    system, error = parse_dsl_code(code)
    if error:
        return {"error": error}
    
    try:
        # Create numerical simulator
        simulator = NumericalSimulator(system)
        
        # Apply custom parameters if provided
        if params:
            for key, value in params.items():
                if hasattr(simulator, key):
                    setattr(simulator, key, value)
        
        current_sim = simulator
        initial_energy = simulator.compute_energy() if hasattr(simulator, 'compute_energy') else None
        sim_time = 0.0
        
        return {"success": True, "system_name": system.name if hasattr(system, 'name') else "system"}
    except Exception as e:
        return {"error": str(e)}

# Fallback simulation classes for when DSL compilation isn't ready
class FallbackDoublePendulum:
    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81,
                 theta1=2.5, theta2=2.0, omega1=0.0, omega2=0.0):
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.g = g
        self.state = np.array([theta1, omega1, theta2, omega2])
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


class FallbackPendulum:
    def __init__(self, m=1.0, l=1.0, g=9.81, theta=0.5, omega=0.0):
        self.m, self.l, self.g = m, l, g
        self.state = np.array([theta, omega])
        self.t = 0.0
        self.initial_energy = self.energy()
        
    def derivatives(self, t, y):
        theta, omega = y
        return [omega, -self.g / self.l * np.sin(theta)]
    
    def step(self, dt):
        sol = solve_ivp(self.derivatives, [self.t, self.t + dt], self.state, method='RK45')
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


class FallbackSpring:
    def __init__(self, k=10.0, m=1.0, b=0.1, x=1.5, v=0.0):
        self.k, self.m, self.b = k, m, b
        self.state = np.array([x, v])
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
        return 0.0  # Damped system


class FallbackOrbital:
    def __init__(self, GM=1000.0, x=100.0, y=0.0, vx=0.0, vy=None):
        self.GM = GM
        if vy is None:
            vy = np.sqrt(GM / x) * 0.8
        self.state = np.array([x, y, vx, vy])
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


def create_sim(sim_type, **kwargs):
    global current_sim
    if sim_type == 'pendulum':
        current_sim = FallbackPendulum(**kwargs)
    elif sim_type == 'double-pendulum':
        current_sim = FallbackDoublePendulum(**kwargs)
    elif sim_type == 'spring':
        current_sim = FallbackSpring(**kwargs)
    elif sim_type == 'orbital':
        current_sim = FallbackOrbital(**kwargs)
    else:
        current_sim = FallbackDoublePendulum(**kwargs)
    return sim_type

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

def has_dsl():
    return HAVE_DSL

print("Python simulation engine ready!")
print(f"MechanicsDSL available: {HAVE_DSL}")
`);

            this.notifyProgress('Ready!', 100);
            this.isReady = true;
            this.isLoading = false;

            // Check if DSL is available
            this.hasDSL = await this.pyodide.runPythonAsync('has_dsl()');
            console.log('MechanicsDSL available:', this.hasDSL);

            return true;
        } catch (error) {
            console.error('Failed to initialize Pyodide:', error);
            this.notifyProgress('Failed to load: ' + error.message, 0);
            this.isLoading = false;
            return false;
        }
    }

    async compileDSL(code) {
        if (!this.isReady || !this.hasDSL) {
            return { error: 'MechanicsDSL not available' };
        }

        try {
            const result = await this.pyodide.runPythonAsync(`
import json
result = create_simulation_from_dsl('''${code.replace(/'/g, "\\'")}''')
json.dumps(result)
`);
            return JSON.parse(result);
        } catch (error) {
            return { error: error.message };
        }
    }

    async createSimulation(type, params = {}) {
        if (!this.isReady) {
            await this.initialize();
        }

        // Build kwargs string
        const kwargs = Object.entries(params)
            .filter(([k, v]) => k !== 'type' && v !== undefined && v !== null)
            .map(([k, v]) => `${k}=${typeof v === 'string' ? `"${v}"` : v}`)
            .join(', ');

        const pythonCode = `create_sim("${type}", ${kwargs})`;

        try {
            return await this.pyodide.runPythonAsync(pythonCode);
        } catch (error) {
            console.error('Create simulation error:', error);
            // Fallback to default
            return await this.pyodide.runPythonAsync(`create_sim("double-pendulum")`);
        }
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
