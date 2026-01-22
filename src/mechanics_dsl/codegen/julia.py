"""
Julia Code Generator for MechanicsDSL

Generates standalone Julia simulation scripts with:
- Real sympy-to-Julia equation conversion
- DifferentialEquations.jl integration
- Multiple solver options (Tsit5, Vern9, Rodas5)
- Plots.jl visualization
- CSV export
"""

from typing import Dict, List, Optional

import sympy as sp
from sympy.printing.julia import julia_code

from ..utils import logger
from .base import CodeGenerator


def sympy_to_julia(expr: sp.Expr) -> str:
    """
    Convert a sympy expression to Julia code.

    Args:
        expr: Sympy expression to convert

    Returns:
        Julia code string

    Examples:
        >>> import sympy as sp
        >>> theta, g, l = sp.symbols('theta g l')
        >>> sympy_to_julia(-g/l * sp.sin(theta))
        '-g*sin(theta)/l'
    """
    if expr is None:
        return "0.0"

    try:
        return julia_code(expr)
    except Exception as e:
        logger.warning(f"Failed to convert expression to Julia: {e}")
        return f"0.0  # ERROR: {e}"


class JuliaGenerator(CodeGenerator):
    """
    Generates Julia simulation code with DifferentialEquations.jl.

    Features:
    - DifferentialEquations.jl ODE solvers (Tsit5, Vern9, Rodas5)
    - Plots.jl visualization
    - CSV export
    - Energy conservation tracking

    Example:
        >>> import sympy as sp
        >>> theta, g, l = sp.symbols('theta g l')
        >>> gen = JuliaGenerator(
        ...     system_name="pendulum",
        ...     coordinates=["theta"],
        ...     parameters={"g": 9.81, "l": 1.0},
        ...     initial_conditions={"theta": 0.5, "theta_dot": 0.0},
        ...     equations={"theta_ddot": -g/l * sp.sin(theta)}
        ... )
        >>> gen.generate("pendulum.jl")
        'pendulum.jl'

    Attributes:
        solver: Julia ODE solver to use (default: Tsit5)
        abstol: Absolute tolerance (default: 1e-8)
        reltol: Relative tolerance (default: 1e-8)
    """

    SUPPORTED_SOLVERS = ["Tsit5", "Vern9", "Rodas5", "CVODE_BDF", "RK4"]

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
        solver: str = "Tsit5",
        abstol: float = 1e-8,
        reltol: float = 1e-8,
    ) -> None:
        """
        Initialize the Julia code generator.

        Args:
            system_name: Name of the physics system
            coordinates: List of generalized coordinate names
            parameters: Physical parameters
            initial_conditions: Initial state values
            equations: Acceleration equations
            lagrangian: Optional Lagrangian for energy
            hamiltonian: Optional Hamiltonian
            forces: Optional non-conservative forces
            constraints: Optional holonomic constraints
            solver: Julia ODE solver (Tsit5, Vern9, Rodas5, etc.)
            abstol: Absolute tolerance
            reltol: Relative tolerance
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

        if solver not in self.SUPPORTED_SOLVERS:
            logger.warning(f"Unknown solver '{solver}', using Tsit5")
            solver = "Tsit5"

        self.solver = solver
        self.abstol = abstol
        self.reltol = reltol

    @property
    def target_name(self) -> str:
        """Target platform identifier."""
        return "julia"

    @property
    def file_extension(self) -> str:
        """File extension for generated code."""
        return ".jl"

    def expr_to_code(self, expr: sp.Expr) -> str:
        """
        Convert sympy expression to Julia code.

        Args:
            expr: Sympy expression

        Returns:
            Julia code string
        """
        return sympy_to_julia(expr)

    def generate(self, output_file: str = "simulation.jl") -> str:
        """
        Generate Julia simulation code.

        Args:
            output_file: Path to output file

        Returns:
            Path to generated file

        Raises:
            ValueError: If validation fails
        """
        self.validate_or_raise()

        logger.info(f"Generating Julia code for {self.system_name}")

        code = self._generate_code()

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(code)

        logger.info(f"Successfully wrote {output_file}")
        return output_file

    def generate_equations(self) -> str:
        """Generate Julia code for equations of motion."""
        lines = []
        idx = 1  # Julia is 1-indexed
        for coord in self.coordinates:
            accel_key = f"{coord}_ddot"
            lines.append(f"    du[{idx}] = u[{idx+1}]  # d{coord}/dt = {coord}_dot")
            if accel_key in self.equations and self.equations[accel_key] is not None:
                expr = self.equations[accel_key]
                jl_expr = self.expr_to_code(expr)
                lines.append(f"    du[{idx+1}] = {jl_expr}  # d{coord}_dot/dt")
            else:
                lines.append(f"    du[{idx+1}] = 0.0  # d{coord}_dot/dt (no equation)")
            idx += 2
        return "\n".join(lines)

    def generate_energy_function(self) -> str:
        """Generate Julia energy computation function."""
        if self.hamiltonian is None and self.lagrangian is None:
            return ""

        return """
# Compute total energy (for conservation checking)
function compute_energy(u)
    # Kinetic energy (approximation: 0.5 * v^2 for unit mass)
    n_coords = div(length(u), 2)
    KE = sum(0.5 * u[2*i]^2 for i in 1:n_coords)
    # Potential energy would need to be added based on Lagrangian
    return KE
end
"""

    def _generate_code(self) -> str:
        """Generate complete Julia simulation script."""
        # Parameter definitions
        param_lines = []
        for name, val in self.parameters.items():
            param_lines.append(f"const {name} = {val}")
        param_str = "\n".join(param_lines) if param_lines else "# No parameters"

        # State variable unpacking (Julia is 1-indexed)
        unpack_lines = []
        idx = 1
        for coord in self.coordinates:
            unpack_lines.append(f"    {coord} = u[{idx}]")
            unpack_lines.append(f"    {coord}_dot = u[{idx+1}]")
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

        # Energy function
        energy_fn = self.generate_energy_function()

        # CSV header
        csv_coords = ", ".join(f'"{c}", "{c}_dot"' for c in self.coordinates)

        template = f'''#=
    {self.system_name} Simulation
    Generated by MechanicsDSL

    Requirements:
        using Pkg
        Pkg.add(["DifferentialEquations", "Plots", "CSV", "DataFrames"])

    Run:
        julia {self.system_name}.jl
=#

using DifferentialEquations
using Plots
using CSV
using DataFrames

# =============================================================================
# Physical Parameters
# =============================================================================
{param_str}

# =============================================================================
# Equations of Motion
# =============================================================================
"""
Compute derivatives for {self.system_name}.

Arguments:
- du: Derivative vector (output)
- u: State vector [{", ".join(f"{c}, {c}_dot" for c in self.coordinates)}]
- p: Parameters (unused, already const)
- t: Current time
"""
function equations_of_motion!(du, u, p, t)
    # Unpack state
{unpack_str}

    # Compute derivatives
{eq_str}

    return nothing
end
{energy_fn}
# =============================================================================
# Simulation
# =============================================================================
"""
Run simulation with configurable options.

Arguments:
- tspan: Time span tuple (default: (0.0, 10.0))
- saveat: Save interval (default: 0.01)

Returns:
    ODE solution object
"""
function simulate(;
    tspan::Tuple{{Float64, Float64}} = (0.0, 10.0),
    saveat::Float64 = 0.01
)
    u0 = [{init_str}]

    prob = ODEProblem(equations_of_motion!, u0, tspan)
    sol = solve(prob, {self.solver}();
        abstol = {self.abstol},
        reltol = {self.reltol},
        saveat = saveat
    )

    return sol
end

# =============================================================================
# Visualization
# =============================================================================
function plot_results(sol; save_path::String = "{self.system_name}.png")
    n_coords = {len(self.coordinates)}

    # Create subplots for each coordinate
    plots = []
    coord_names = [{", ".join(f'"{c}"' for c in self.coordinates)}]

    for i in 1:n_coords
        p1 = plot(sol.t, sol[2*i - 1, :],
            label = coord_names[i],
            xlabel = "Time (s)",
            ylabel = "Position",
            linewidth = 2
        )
        p2 = plot(sol.t, sol[2*i, :],
            label = "$(coord_names[i])_dot",
            xlabel = "Time (s)",
            ylabel = "Velocity",
            linewidth = 2,
            color = :red
        )
        push!(plots, p1)
        push!(plots, p2)
    end

    p = plot(plots..., layout = (n_coords, 2), title = "{self.system_name}")
    savefig(p, save_path)
    println("Saved plot to $save_path")
    return p
end

function plot_phase_space(sol; save_path::String = "{self.system_name}_phase.png")
    plot(sol[1, :], sol[2, :],
        xlabel = "{self.coordinates[0] if self.coordinates else 'q'}",
        ylabel = "{self.coordinates[0] + '_dot' if self.coordinates else 'q_dot'}",
        title = "Phase Space: {self.system_name}",
        linewidth = 1.5,
        legend = false
    )
    savefig(save_path)
    println("Saved phase plot to $save_path")
end

# =============================================================================
# Data Export
# =============================================================================
function export_csv(sol; filename::String = "{self.system_name}_results.csv")
    df = DataFrame(
        t = sol.t,
        {", ".join(f'{c} = sol[{2*i+1}, :], {c}_dot = sol[{2*i+2}, :]' for i, c in enumerate(self.coordinates))}
    )
    CSV.write(filename, df)
    println("Exported $(length(sol.t)) points to $filename")
end

# =============================================================================
# Main
# =============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    println("Running {self.system_name} simulation...")
    println("  Solver: {self.solver}")
    println("  Tolerances: abstol={self.abstol}, reltol={self.reltol}")

    @time sol = simulate()

    println("Simulation complete: $(length(sol.t)) points")
    println("Final state: $(sol.u[end])")

    plot_results(sol)
    plot_phase_space(sol)
    export_csv(sol)
end
'''
        return template
