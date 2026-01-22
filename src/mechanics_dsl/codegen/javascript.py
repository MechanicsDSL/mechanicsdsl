"""
JavaScript Code Generator for MechanicsDSL

Generates standalone JavaScript simulation code with:
- Real sympy-to-JavaScript equation conversion  
- ES6+ module format
- Built-in RK4/Euler/adaptive integrators
- Canvas visualization helper
- JSON/CSV export
- TypeScript type definitions (optional)
"""

from typing import Dict, List, Optional

import sympy as sp
from sympy.printing.jscode import jscode

from ..utils import logger
from .base import CodeGenerator


def sympy_to_javascript(expr: sp.Expr, use_math: bool = True) -> str:
    """
    Convert a sympy expression to JavaScript code.

    Args:
        expr: Sympy expression to convert
        use_math: If True, use Math.* functions

    Returns:
        JavaScript code string

    Examples:
        >>> import sympy as sp
        >>> x = sp.Symbol('x')
        >>> sympy_to_javascript(sp.sin(x))
        'Math.sin(x)'
    """
    if expr is None:
        return "0.0"

    try:
        return jscode(expr)
    except Exception as e:
        logger.warning(f"Failed to convert expression to JavaScript: {e}")
        return f"0.0 /* ERROR: {e} */"


class JavaScriptGenerator(CodeGenerator):
    """
    Generates JavaScript simulation code.

    Features:
    - ES6+ modules with CommonJS fallback
    - Built-in RK4, Euler, and adaptive integrators
    - Canvas visualization helpers
    - JSON and CSV export
    - TypeScript definitions (optional)

    Example:
        >>> import sympy as sp
        >>> theta, g, l = sp.symbols('theta g l')
        >>> gen = JavaScriptGenerator(
        ...     system_name="pendulum",
        ...     coordinates=["theta"],
        ...     parameters={"g": 9.81, "l": 1.0},
        ...     initial_conditions={"theta": 0.5, "theta_dot": 0.0},
        ...     equations={"theta_ddot": -g/l * sp.sin(theta)},
        ...     integrator="rk4"
        ... )
        >>> gen.generate("pendulum.js")
        'pendulum.js'

    Attributes:
        integrator: Integration method ('euler', 'rk4', 'adaptive')
        generate_typescript: Whether to generate .d.ts file
    """

    SUPPORTED_INTEGRATORS = ["euler", "rk4", "adaptive"]

    def __init__(
        self,
        system_name: str,
        coordinates: List[str],
        parameters: Dict[str, float],
        initial_conditions: Dict[str, float],
        equations: Dict[str, sp.Expr],
        lagrangian: Optional[sp.Expr] = None,
        hamiltonian: Optional[sp.Expr] = None,
        forces: Optional[List[sp.Expr]] = None,
        constraints: Optional[List[sp.Expr]] = None,
        integrator: str = "rk4",
        generate_typescript: bool = False,
    ) -> None:
        """
        Initialize the JavaScript code generator.

        Args:
            system_name: Name of the physics system
            coordinates: List of generalized coordinate names
            parameters: Physical parameters
            initial_conditions: Initial state values
            equations: Acceleration equations
            lagrangian: Optional Lagrangian
            hamiltonian: Optional Hamiltonian
            forces: Optional non-conservative forces
            constraints: Optional holonomic constraints
            integrator: Integration method ('euler', 'rk4', 'adaptive')
            generate_typescript: Generate TypeScript declarations
        """
        super().__init__(
            system_name=system_name,
            coordinates=coordinates,
            parameters=parameters,
            initial_conditions=initial_conditions,
            equations=equations,
            lagrangian=lagrangian,
            hamiltonian=hamiltonian,
            forces=forces,
            constraints=constraints,
        )

        if integrator not in self.SUPPORTED_INTEGRATORS:
            logger.warning(f"Unknown integrator '{integrator}', using rk4")
            integrator = "rk4"

        self.integrator = integrator
        self.generate_typescript = generate_typescript

    @property
    def target_name(self) -> str:
        """Target platform identifier."""
        return "javascript"

    @property
    def file_extension(self) -> str:
        """File extension for generated code."""
        return ".js"

    def expr_to_code(self, expr: sp.Expr) -> str:
        """
        Convert sympy expression to JavaScript code.

        Args:
            expr: Sympy expression

        Returns:
            JavaScript code string
        """
        return sympy_to_javascript(expr)

    def generate(self, output_file: str = "simulation.js") -> str:
        """
        Generate JavaScript simulation code.

        Args:
            output_file: Path to output file

        Returns:
            Path to generated file

        Raises:
            ValueError: If validation fails
        """
        self.validate_or_raise()

        logger.info(f"Generating JavaScript code for {self.system_name}")

        code = self._generate_code()

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(code)

        # Generate TypeScript declarations if requested
        if self.generate_typescript:
            ts_file = output_file.replace(".js", ".d.ts")
            with open(ts_file, "w", encoding="utf-8") as f:
                f.write(self._generate_typescript_defs())
            logger.info(f"Generated TypeScript definitions: {ts_file}")

        logger.info(f"Successfully wrote {output_file}")
        return output_file

    def generate_equations(self) -> str:
        """Generate JavaScript code for equations of motion."""
        lines = []
        idx = 0
        for coord in self.coordinates:
            accel_key = f"{coord}_ddot"
            lines.append(f"    dydt[{idx}] = y[{idx+1}];  // d{coord}/dt = {coord}_dot")
            if accel_key in self.equations and self.equations[accel_key] is not None:
                expr = self.equations[accel_key]
                js_expr = self.expr_to_code(expr)
                lines.append(f"    dydt[{idx+1}] = {js_expr};  // d{coord}_dot/dt")
            else:
                lines.append(f"    dydt[{idx+1}] = 0.0;  // d{coord}_dot/dt (no equation)")
            idx += 2
        return "\n".join(lines)

    def _generate_integrators(self) -> str:
        """Generate all integrator implementations."""
        return '''
/**
 * Euler integration step (1st order)
 * @param {number} t - Current time
 * @param {number[]} y - State vector
 * @param {number} dt - Time step
 * @returns {number[]} - Updated state
 */
function eulerStep(t, y, dt) {
    const dydt = equationsOfMotion(t, y);
    return y.map((yi, i) => yi + dt * dydt[i]);
}

/**
 * RK4 integration step (4th order)
 * @param {number} t - Current time
 * @param {number[]} y - State vector
 * @param {number} dt - Time step
 * @returns {number[]} - Updated state
 */
function rk4Step(t, y, dt) {
    const k1 = equationsOfMotion(t, y);
    const k2 = equationsOfMotion(t + 0.5*dt, y.map((yi, i) => yi + 0.5*dt*k1[i]));
    const k3 = equationsOfMotion(t + 0.5*dt, y.map((yi, i) => yi + 0.5*dt*k2[i]));
    const k4 = equationsOfMotion(t + dt, y.map((yi, i) => yi + dt*k3[i]));

    return y.map((yi, i) => yi + dt/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]));
}

/**
 * Adaptive RK4-5 integration step (Dormand-Prince)
 * @param {number} t - Current time
 * @param {number[]} y - State vector
 * @param {number} dt - Initial time step
 * @param {number} tol - Error tolerance (default 1e-6)
 * @returns {{y: number[], dt: number, error: number}} - State, new dt, error estimate
 */
function adaptiveStep(t, y, dt, tol = 1e-6) {
    // Two half steps
    const y1 = rk4Step(t, y, dt/2);
    const y2 = rk4Step(t + dt/2, y1, dt/2);

    // One full step
    const yFull = rk4Step(t, y, dt);

    // Error estimate
    const error = Math.max(...y2.map((yi, i) => Math.abs(yi - yFull[i])));

    // Adaptive step size
    let newDt = dt;
    if (error > tol * 16) {
        newDt = dt * 0.5;  // Shrink
    } else if (error < tol / 16) {
        newDt = dt * 2.0;  // Grow
    }

    // Use more accurate (y2) if within tolerance
    return { y: error < tol ? y2 : y, dt: newDt, error };
}
'''

    def _generate_typescript_defs(self) -> str:
        """Generate TypeScript type definitions."""
        state_dim = len(self.coordinates) * 2
        return f'''// TypeScript definitions for {self.system_name}
// Generated by MechanicsDSL

export interface SimulationConfig {{
    tEnd?: number;
    dt?: number;
    integrator?: 'euler' | 'rk4' | 'adaptive';
}}

export interface SimulationResults {{
    t: number[];
    y: number[][];
    energy?: number[];
}}

export interface PhysicsParameters {{
    {"; ".join(f"{name}: number" for name in self.parameters.keys())};
}}

export function equationsOfMotion(t: number, y: number[]): number[];
export function simulate(config?: SimulationConfig): SimulationResults;
export function computeEnergy(y: number[]): number;
export function exportCSV(results: SimulationResults): string;

export const STATE_DIM: {state_dim};
export const PARAMETERS: PhysicsParameters;
'''

    def _generate_code(self) -> str:
        """Generate complete JavaScript simulation module."""
        # Parameter definitions
        param_lines = []
        for name, val in self.parameters.items():
            param_lines.append(f"const {name} = {val};")
        param_str = "\n".join(param_lines) if param_lines else "// No parameters"

        # State variable unpacking
        unpack_lines = []
        idx = 0
        for coord in self.coordinates:
            unpack_lines.append(f"    const {coord} = y[{idx}];")
            unpack_lines.append(f"    const {coord}_dot = y[{idx+1}];")
            idx += 2
        unpack_str = "\n".join(unpack_lines)

        # Equations
        eq_str = self.generate_equations()

        # Initial conditions
        init_vals = []
        for coord in self.coordinates:
            pos = self.initial_conditions.get(coord, 0.0)
            vel = self.initial_conditions.get(f"{coord}_dot", 0.0)
            init_vals.extend([str(pos), str(vel)])
        init_str = ", ".join(init_vals)

        state_dim = len(self.coordinates) * 2

        # Integrator choice
        step_fn = {
            "euler": "eulerStep",
            "rk4": "rk4Step",
            "adaptive": "adaptiveStep"
        }.get(self.integrator, "rk4Step")

        # Integrators code
        integrators = self._generate_integrators()

        template = f'''/**
 * {self.system_name} Simulation
 * Generated by MechanicsDSL
 *
 * Usage (Node.js):
 *   const sim = require('./{self.system_name}');
 *   const results = sim.simulate({{ tEnd: 10, dt: 0.01 }});
 *
 * Usage (ES6):
 *   import {{ simulate }} from './{self.system_name}';
 *   const results = simulate({{ tEnd: 10, dt: 0.01 }});
 *
 * Usage (Browser):
 *   <script src="{self.system_name}.js"></script>
 *   <script>const results = simulate();</script>
 */

'use strict';

// =============================================================================
// Physical Parameters
// =============================================================================
{param_str}

// Parameters object for access
const PARAMETERS = {{
    {", ".join(f"{name}: {name}" for name in self.parameters.keys())}
}};

const STATE_DIM = {state_dim};

// =============================================================================
// Equations of Motion
// =============================================================================
/**
 * Compute derivatives for {self.system_name}.
 *
 * @param {{number}} t - Current time
 * @param {{number[]}} y - State vector [{", ".join(f"{c}, {c}_dot" for c in self.coordinates)}]
 * @returns {{number[]}} - Derivative vector
 */
function equationsOfMotion(t, y) {{
    const dydt = new Array(STATE_DIM).fill(0);

    // Unpack state
{unpack_str}

    // Compute derivatives
{eq_str}

    return dydt;
}}

// =============================================================================
// Integrators
// =============================================================================
{integrators}

// =============================================================================
// Energy Computation
// =============================================================================
/**
 * Compute total mechanical energy (kinetic only, approximate).
 * @param {{number[]}} y - State vector
 * @returns {{number}} - Total energy
 */
function computeEnergy(y) {{
    let ke = 0;
    for (let i = 0; i < STATE_DIM/2; i++) {{
        ke += 0.5 * y[2*i + 1] * y[2*i + 1];
    }}
    return ke;
}}

// =============================================================================
// Simulation
// =============================================================================
/**
 * Run physics simulation.
 *
 * @param {{Object}} config - Simulation configuration
 * @param {{number}} config.tEnd - End time (default: 10.0)
 * @param {{number}} config.dt - Time step (default: 0.01)
 * @param {{string}} config.integrator - 'euler', 'rk4', or 'adaptive' (default: '{self.integrator}')
 * @returns {{Object}} - Results with t, y arrays and optional energy
 */
function simulate(config = {{}}) {{
    const tEnd = config.tEnd ?? 10.0;
    const dt = config.dt ?? 0.01;
    const integrator = config.integrator ?? '{self.integrator}';
    const trackEnergy = config.trackEnergy ?? false;

    let y = [{init_str}];
    let t = 0;
    let currentDt = dt;

    const results = {{ t: [], y: [], energy: [] }};

    // Choose integrator
    let stepFn;
    switch (integrator) {{
        case 'euler': stepFn = eulerStep; break;
        case 'adaptive': stepFn = adaptiveStep; break;
        default: stepFn = rk4Step;
    }}

    // Main loop
    while (t < tEnd) {{
        results.t.push(t);
        results.y.push([...y]);

        if (trackEnergy) {{
            results.energy.push(computeEnergy(y));
        }}

        if (integrator === 'adaptive') {{
            const step = adaptiveStep(t, y, currentDt);
            y = step.y;
            currentDt = step.dt;
        }} else {{
            y = stepFn(t, y, currentDt);
        }}

        t += currentDt;
    }}

    return results;
}}

// =============================================================================
// Data Export
// =============================================================================
/**
 * Export simulation results to CSV string.
 * @param {{Object}} results - Simulation results
 * @returns {{string}} - CSV content
 */
function exportCSV(results) {{
    const headers = ['t', {", ".join(f"'{c}', '{c}_dot'" for c in self.coordinates)}];
    let csv = headers.join(',') + '\\n';

    for (let i = 0; i < results.t.length; i++) {{
        const row = [results.t[i], ...results.y[i]];
        csv += row.join(',') + '\\n';
    }}

    return csv;
}}

/**
 * Export simulation results to JSON.
 * @param {{Object}} results - Simulation results
 * @returns {{string}} - JSON string
 */
function exportJSON(results) {{
    return JSON.stringify(results, null, 2);
}}

// =============================================================================
// Canvas Visualization Helper
// =============================================================================
/**
 * Draw a simple pendulum animation on canvas.
 * @param {{CanvasRenderingContext2D}} ctx - Canvas context
 * @param {{number[]}} state - Current state vector
 * @param {{Object}} opts - Drawing options
 */
function drawPendulum(ctx, state, opts = {{}}) {{
    const width = ctx.canvas.width;
    const height = ctx.canvas.height;
    const cx = width / 2;
    const cy = height / 3;
    const length = opts.length ?? 150;

    const theta = state[0];
    const x = cx + length * Math.sin(theta);
    const y = cy + length * Math.cos(theta);

    // Clear and draw
    ctx.fillStyle = opts.background ?? '#1a1a2e';
    ctx.fillRect(0, 0, width, height);

    // Rod
    ctx.strokeStyle = opts.rodColor ?? '#00d4ff';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(x, y);
    ctx.stroke();

    // Bob
    ctx.fillStyle = opts.bobColor ?? '#ff6b6b';
    ctx.beginPath();
    ctx.arc(x, y, opts.bobRadius ?? 20, 0, 2 * Math.PI);
    ctx.fill();

    // Pivot
    ctx.fillStyle = '#888';
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
    ctx.fill();
}}

// =============================================================================
// Module Exports
// =============================================================================
const exports_ = {{
    simulate,
    equationsOfMotion,
    computeEnergy,
    exportCSV,
    exportJSON,
    drawPendulum,
    eulerStep,
    rk4Step,
    adaptiveStep,
    PARAMETERS,
    STATE_DIM
}};

// CommonJS
if (typeof module !== 'undefined' && module.exports) {{
    module.exports = exports_;
}}

// ES6 modules
if (typeof window !== 'undefined') {{
    Object.assign(window, exports_);
}}

// Run if executed directly (Node.js)
if (typeof require !== 'undefined' && require.main === module) {{
    console.log('Running {self.system_name} simulation...');
    console.time('Simulation');
    const results = simulate({{ tEnd: 10, dt: 0.01, trackEnergy: true }});
    console.timeEnd('Simulation');
    console.log(`Completed: ${{results.t.length}} points`);
    console.log(`Final state: ${{results.y[results.y.length-1]}}`);
    console.log(`Energy drift: ${{Math.abs(results.energy[0] - results.energy[results.energy.length-1]).toFixed(6)}}`);
}}
'''
        return template
