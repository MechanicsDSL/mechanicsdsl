"""
OpenMP Code Generator for MechanicsDSL

Generates OpenMP-parallel C++ code for multi-core CPU simulations with:
- Thread-parallel integration for parameter sweeps
- OpenMP reduction for energy calculations
- SIMD-friendly loop structures
- Scalability benchmarking support
"""

from typing import Dict, List, Optional

import sympy as sp
from sympy.printing.cxx import cxxcode

from ..utils import logger
from .base import CodeGenerator


def sympy_to_cpp_openmp(expr: sp.Expr) -> str:
    """
    Convert a sympy expression to C++ code for OpenMP.

    Args:
        expr: Sympy expression to convert

    Returns:
        C++ code string
    """
    if expr is None:
        return "0.0"
    try:
        return cxxcode(expr, standard="c++17")
    except Exception as e:
        logger.warning(f"Failed to convert expression to C++: {e}")
        return f"0.0 /* ERROR: {e} */"


class OpenMPGenerator(CodeGenerator):
    """
    Generates OpenMP-parallel C++ simulation code.

    Features:
    - Thread-parallel integration for multi-body systems
    - OpenMP reduction for energy calculations
    - SIMD-friendly loop structures
    - Automatic thread count detection
    - Performance timing infrastructure

    Example:
        >>> import sympy as sp
        >>> theta, g, l = sp.symbols('theta g l')
        >>> gen = OpenMPGenerator(
        ...     system_name="pendulum",
        ...     coordinates=['theta'],
        ...     parameters={'g': 9.81, 'l': 1.0},
        ...     initial_conditions={'theta': 0.1, 'theta_dot': 0.0},
        ...     equations={'theta_ddot': -g/l * sp.sin(theta)},
        ...     num_threads=4,
        ...     num_systems=1000
        ... )
        >>> gen.generate("pendulum_openmp.cpp")
        'pendulum_openmp.cpp'

    Attributes:
        num_threads: Number of OpenMP threads (0 = auto-detect)
        num_systems: Number of parallel trajectories to simulate
    """

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
        num_threads: int = 0,
        num_systems: int = 100,
    ) -> None:
        """
        Initialize the OpenMP code generator.

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
            num_threads: Number of threads (0 = auto-detect)
            num_systems: Number of parallel trajectories
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
        self.num_threads = num_threads
        self.num_systems = num_systems

    @property
    def target_name(self) -> str:
        """Target platform identifier."""
        return "openmp"

    @property
    def file_extension(self) -> str:
        """File extension for generated code."""
        return ".cpp"

    def expr_to_code(self, expr: sp.Expr) -> str:
        """
        Convert sympy expression to C++ code.

        Args:
            expr: Sympy expression

        Returns:
            C++ code string
        """
        return sympy_to_cpp_openmp(expr)

    def generate(self, output_file: str) -> str:
        """
        Generate OpenMP C++ code.

        Args:
            output_file: Path to output file

        Returns:
            Path to generated file

        Raises:
            ValueError: If validation fails
        """
        self.validate_or_raise()

        logger.info(f"Generating OpenMP code for {self.system_name}")

        code = self._generate_source()

        with open(output_file, "w") as f:
            f.write(code)

        logger.info(f"Generated {output_file}")
        return output_file

    def generate_equations(self) -> str:
        """Generate equations code."""
        lines = []
        idx = 0
        for coord in self.coordinates:
            accel_key = f"{coord}_ddot"
            lines.append(f"        dydt[{idx}] = y[{idx+1}];")
            if accel_key in self.equations and self.equations[accel_key] is not None:
                expr = self.equations[accel_key]
                cpp_expr = self.expr_to_code(expr)
                lines.append(f"        dydt[{idx+1}] = {cpp_expr};")
            else:
                lines.append(f"        dydt[{idx+1}] = 0.0;")
            idx += 2
        return "\n".join(lines)

    def _generate_source(self) -> str:
        """Generate the complete OpenMP source file."""
        state_dim = len(self.coordinates) * 2

        # Parameters
        params = "\n".join(f"const double {name} = {val};" for name, val in self.parameters.items())

        # State unpacking
        unpack = "\n".join(
            f"        const double {c} = y[{2*i}]; const double {c}_dot = y[{2*i+1}];"
            for i, c in enumerate(self.coordinates)
        )

        # Initial conditions
        init_vals = []
        for coord in self.coordinates:
            init_vals.append(str(self.initial_conditions.get(coord, 0.0)))
            init_vals.append(str(self.initial_conditions.get(f"{coord}_dot", 0.0)))
        init_str = ", ".join(init_vals)

        # CSV header
        header = ",".join(["t"] + [x for c in self.coordinates for x in [c, f"{c}_dot"]])

        # Thread setting
        thread_init = ""
        if self.num_threads > 0:
            thread_init = f"omp_set_num_threads({self.num_threads});"

        return f"""/*
 * OpenMP Parallel Simulation: {self.system_name}
 * Generated by MechanicsDSL
 * 
 * Compile with: g++ -fopenmp -O3 -o {self.system_name} {self.system_name}.cpp
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <omp.h>

using std::sin; using std::cos; using std::tan;
using std::exp; using std::log; using std::sqrt;
using std::pow; using std::abs;

// Physical Parameters
{params}

constexpr int STATE_DIM = {state_dim};
constexpr int NUM_SYSTEMS = 100;  // Number of parallel trajectories

// Compute derivatives for a single system
inline void compute_derivatives(const double* y, double* dydt, double t) {{
{unpack}

{self.generate_equations()}
}}

// RK4 step for a single system
inline void rk4_step(double* y, double t, double dt) {{
    double k1[STATE_DIM], k2[STATE_DIM], k3[STATE_DIM], k4[STATE_DIM];
    double temp[STATE_DIM];
    
    compute_derivatives(y, k1, t);
    
    for (int i = 0; i < STATE_DIM; i++) temp[i] = y[i] + 0.5 * dt * k1[i];
    compute_derivatives(temp, k2, t + 0.5*dt);
    
    for (int i = 0; i < STATE_DIM; i++) temp[i] = y[i] + 0.5 * dt * k2[i];
    compute_derivatives(temp, k3, t + 0.5*dt);
    
    for (int i = 0; i < STATE_DIM; i++) temp[i] = y[i] + dt * k3[i];
    compute_derivatives(temp, k4, t + dt);
    
    for (int i = 0; i < STATE_DIM; i++) {{
        y[i] += dt * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) / 6.0;
    }}
}}

int main() {{
    {thread_init}
    
    std::cout << "OpenMP Simulation: {self.system_name}" << std::endl;
    std::cout << "Using " << omp_get_max_threads() << " threads" << std::endl;
    
    // Initialize multiple systems with slightly different initial conditions
    std::vector<std::vector<double>> systems(NUM_SYSTEMS, std::vector<double>(STATE_DIM));
    
    double base_ic[STATE_DIM] = {{ {init_str} }};
    
    #pragma omp parallel for
    for (int s = 0; s < NUM_SYSTEMS; s++) {{
        for (int i = 0; i < STATE_DIM; i++) {{
            // Add small perturbation to each system
            systems[s][i] = base_ic[i] + 0.01 * (s - NUM_SYSTEMS/2.0) / NUM_SYSTEMS;
        }}
    }}
    
    // Simulation parameters
    double t = 0.0;
    double dt = 0.001;
    double t_end = 10.0;
    int steps = static_cast<int>(t_end / dt);
    int output_interval = 100;
    
    // Output file for system 0
    std::ofstream outfile("{self.system_name}_openmp_results.csv");
    outfile << "{header}" << std::endl;
    outfile << std::fixed << std::setprecision(6);
    
    // Timing
    double start_time = omp_get_wtime();
    
    // Main simulation loop
    for (int step = 0; step <= steps; step++) {{
        // Output first system
        if (step % output_interval == 0) {{
            outfile << t;
            for (int i = 0; i < STATE_DIM; i++) {{
                outfile << "," << systems[0][i];
            }}
            outfile << std::endl;
        }}
        
        // Parallel RK4 integration
        #pragma omp parallel for schedule(dynamic)
        for (int s = 0; s < NUM_SYSTEMS; s++) {{
            rk4_step(systems[s].data(), t, dt);
        }}
        
        t += dt;
    }}
    
    double elapsed = omp_get_wtime() - start_time;
    
    std::cout << "Simulated " << NUM_SYSTEMS << " trajectories in " 
              << elapsed << " seconds" << std::endl;
    std::cout << "Results saved to {self.system_name}_openmp_results.csv" << std::endl;
    
    return 0;
}}
"""
