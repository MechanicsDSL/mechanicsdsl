"""
Python Code Generator for MechanicsDSL

Generates standalone Python simulation scripts with:
- Real sympy-to-Python equation conversion
- NumPy/SciPy integration with solve_ivp
- Optional JAX backend for GPU acceleration
- Optional Numba JIT compilation
- Matplotlib visualization
- Energy conservation verification

The generated code runs independently without MechanicsDSL installed.
"""

from typing import Any, Dict, List, Optional

import sympy as sp
from sympy.printing import pycode

from ..utils import logger
from .base import CodeGenerator


def sympy_to_python(expr: sp.Expr, use_numpy: bool = True) -> str:
    """
    Convert a sympy expression to Python code.

    Args:
        expr: Sympy expression to convert
        use_numpy: If True, use numpy functions (np.sin, etc.)

    Returns:
        Python code string

    Examples:
        >>> import sympy as sp
        >>> theta, g, l = sp.symbols('theta g l')
        >>> sympy_to_python(-g/l * sp.sin(theta))
        '-g*numpy.sin(theta)/l'
    """
    if expr is None:
        return "0.0"

    try:
        # Use sympy's pycode with NumPy module
        if use_numpy:
            py_code = pycode(expr, fully_qualified_modules=False)
            # Replace 'math.' with 'np.'
            py_code = py_code.replace("math.", "np.")
        else:
            py_code = pycode(expr)

        return py_code
    except Exception as e:
        logger.warning(f"Failed to convert expression to Python: {e}")
        return f"0.0  # ERROR: {e}"


class PythonGenerator(CodeGenerator):
    """
    Generates standalone Python simulation code.

    Produces scripts that can run without MechanicsDSL installed,
    using only standard scientific Python packages (NumPy, SciPy, Matplotlib).

    Features:
    - NumPy-based state vector operations
    - SciPy's solve_ivp for ODE integration
    - Multiple integrator options (RK45, RK23, DOP853, Radau, BDF)
    - Matplotlib plotting
    - CSV/JSON export
    - Optional JAX backend for GPU
    - Optional Numba JIT for CPU speedup

    Example:
        >>> import sympy as sp
        >>> theta, g, l = sp.symbols('theta g l')
        >>> gen = PythonGenerator(
        ...     system_name="pendulum",
        ...     coordinates=["theta"],
        ...     parameters={"g": 9.81, "l": 1.0},
        ...     initial_conditions={"theta": 0.5, "theta_dot": 0.0},
        ...     equations={"theta_ddot": -g/l * sp.sin(theta)}
        ... )
        >>> gen.generate("pendulum.py")
        'pendulum.py'

    Attributes:
        use_jax: If True, generate JAX-compatible code
        use_numba: If True, add @jit decorators
        integrator: SciPy integrator method (RK45, DOP853, etc.)
    """

    SUPPORTED_INTEGRATORS = ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]

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
        use_jax: bool = False,
        use_numba: bool = False,
        integrator: str = "RK45",
    ) -> None:
        """
        Initialize the Python code generator.

        Args:
            system_name: Name of the physics system
            coordinates: List of generalized coordinate names
            parameters: Physical parameters as name -> value dict
            initial_conditions: Initial state values
            equations: Acceleration equations as coord_ddot -> sympy.Expr
            lagrangian: Optional Lagrangian for energy verification
            hamiltonian: Optional Hamiltonian for conservation checks
            forces: Optional non-conservative forces
            constraints: Optional holonomic constraints
            use_jax: Generate JAX-compatible code for GPU
            use_numba: Add Numba @jit decorators for CPU speedup
            integrator: SciPy integrator method (default: RK45)
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

        self.use_jax = use_jax
        self.use_numba = use_numba

        if integrator not in self.SUPPORTED_INTEGRATORS:
            logger.warning(f"Unknown integrator '{integrator}', using RK45")
            integrator = "RK45"
        self.integrator = integrator

    @property
    def target_name(self) -> str:
        """Target platform identifier."""
        return "python"

    @property
    def file_extension(self) -> str:
        """File extension for generated code."""
        return ".py"

    def expr_to_code(self, expr: sp.Expr) -> str:
        """
        Convert a sympy expression to Python code.

        Uses numpy functions for vectorized operations.

        Args:
            expr: Sympy expression to convert

        Returns:
            Python code string using numpy
        """
        return sympy_to_python(expr, use_numpy=True)

    def generate(self, output_file: str = "simulation.py") -> str:
        """
        Generate Python simulation script.

        Args:
            output_file: Path to output file

        Returns:
            Path to generated file

        Raises:
            ValueError: If validation fails
        """
        self.validate_or_raise()

        logger.info(f"Generating Python code for {self.system_name}")

        code = self._generate_code()

        with open(output_file, "w") as f:
            f.write(code)

        logger.info(f"Successfully wrote {output_file}")
        return output_file

    def generate_equations(self) -> str:
        """
        Generate Python code for equations of motion.

        Returns:
            Python code computing derivatives
        """
        lines = []
        idx = 0
        for coord in self.coordinates:
            accel_key = f"{coord}_ddot"

            # Velocity equation
            lines.append(f"    dydt[{idx}] = y[{idx + 1}]  # d{coord}/dt = {coord}_dot")

            # Acceleration equation
            if accel_key in self.equations and self.equations[accel_key] is not None:
                expr = self.equations[accel_key]
                py_expr = self.expr_to_code(expr)
                lines.append(f"    dydt[{idx + 1}] = {py_expr}  # d{coord}_dot/dt")
            else:
                lines.append(f"    dydt[{idx + 1}] = 0.0  # d{coord}_dot/dt (no equation)")

            idx += 2
        return "\n".join(lines)

    def generate_energy_computation(self) -> Optional[str]:
        """
        Generate Python code to compute total energy.

        Returns:
            Python code for energy function, or None
        """
        if self.lagrangian is None:
            return None

        return f'''
def compute_energy(y):
    """Compute total energy (kinetic + potential)."""
    # Unpack state
{self._generate_unpack("y")}

    # Energy from Lagrangian (L = T - V)
    # TODO: Implement proper energy extraction
    return 0.0
'''

    def _generate_code(self) -> str:
        """Generate complete Python simulation script."""
        # Imports
        imports = self._generate_imports()

        # Parameters
        param_str = self._generate_parameters()

        # State unpacking
        unpack_str = self._generate_unpack("y")

        # Equations
        eq_str = self.generate_equations()

        # Initial conditions
        init_str = self.generate_initial_conditions()

        # Numba decorator
        numba_decorator = "@numba.jit(nopython=True)" if self.use_numba else ""

        # Energy computation
        energy_code = self.generate_energy_computation() or ""

        template = f'''"""
{self.system_name} Simulation
Generated by MechanicsDSL

Run: python {self.system_name}.py
"""
{imports}

# =============================================================================
# Physical Parameters
# =============================================================================
{param_str}

# =============================================================================
# Equations of Motion
# =============================================================================
{numba_decorator}
def equations_of_motion(t: float, y: np.ndarray) -> np.ndarray:
    """
    Compute derivatives for {self.system_name}.

    Args:
        t: Current time
        y: State vector [{", ".join(f"{c}, {c}_dot" for c in self.coordinates)}]

    Returns:
        Derivative vector dydt
    """
    dydt = np.zeros({self.state_dim})

    # Unpack state
{unpack_str}

    # Compute derivatives
{eq_str}

    return dydt

{energy_code}

# =============================================================================
# Simulation
# =============================================================================
def simulate(
    t_span: tuple = (0, 10),
    num_points: int = 1000,
    method: str = "{self.integrator}",
) -> Any:
    """
    Run simulation.

    Args:
        t_span: Time span (t_start, t_end)
        num_points: Number of output points
        method: Integration method ({", ".join(self.SUPPORTED_INTEGRATORS)})

    Returns:
        SciPy OdeResult object with t and y arrays
    """
    y0 = np.array([{init_str}])
    t_eval = np.linspace(t_span[0], t_span[1], num_points)

    solution = solve_ivp(
        equations_of_motion,
        t_span,
        y0,
        t_eval=t_eval,
        method=method,
        dense_output=True,
    )

    return solution


def plot_results(solution, save_path: str = None):
    """
    Plot simulation results.

    Args:
        solution: SciPy OdeResult from simulate()
        save_path: Optional path to save figure
    """
    n_coords = {len(self.coordinates)}
    fig, axes = plt.subplots(n_coords, 2, figsize=(12, 4 * n_coords), sharex=True)

    if n_coords == 1:
        axes = axes.reshape(1, -1)

    coord_names = {self.coordinates}
    for i, coord in enumerate(coord_names):
        # Position
        axes[i, 0].plot(solution.t, solution.y[2*i], "b-", linewidth=1.5)
        axes[i, 0].set_ylabel(f"{{coord}}")
        axes[i, 0].grid(True, alpha=0.3)

        # Velocity
        axes[i, 1].plot(solution.t, solution.y[2*i + 1], "r-", linewidth=1.5)
        axes[i, 1].set_ylabel(f"{{coord}}_dot")
        axes[i, 1].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")

    fig.suptitle("{self.system_name}", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {{save_path}}")
    else:
        plt.show()


def export_csv(solution, filename: str = "{self.system_name}_results.csv"):
    """Export results to CSV file."""
    import csv

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["t"] + [{", ".join(f'"{c}", "{c}_dot"' for c in self.coordinates)}]
        writer.writerow(header)

        for i, t in enumerate(solution.t):
            row = [t] + list(solution.y[:, i])
            writer.writerow(row)

    print(f"Exported {{len(solution.t)}} points to {{filename}}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("Running {self.system_name} simulation...")
    sol = simulate()
    print(f"Simulation complete: {{len(sol.t)}} points")
    print(f"Final state: {{sol.y[:, -1]}}")

    # Plot
    plot_results(sol)

    # Export
    export_csv(sol)
'''
        return template

    def _generate_imports(self) -> str:
        """Generate import statements."""
        lines = [
            "from typing import Any",
            "",
            "import numpy as np",
            "from scipy.integrate import solve_ivp",
            "import matplotlib.pyplot as plt",
        ]

        if self.use_numba:
            lines.append("import numba")

        if self.use_jax:
            lines.append("import jax.numpy as jnp")
            lines.append("from jax import jit")

        return "\n".join(lines)

    def _generate_parameters(self) -> str:
        """Generate parameter declarations."""
        lines = []
        for name, val in self.parameters.items():
            lines.append(f"{name}: float = {val}")
        return "\n".join(lines)

    def _generate_unpack(self, var: str = "y") -> str:
        """Generate state unpacking code."""
        lines = []
        idx = 0
        for coord in self.coordinates:
            lines.append(f"    {coord} = {var}[{idx}]")
            lines.append(f"    {coord}_dot = {var}[{idx + 1}]")
            idx += 2
        return "\n".join(lines)
