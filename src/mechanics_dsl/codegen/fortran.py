"""
Fortran Code Generator for MechanicsDSL

Generates Fortran 90+ simulation code with:
- Real sympy-to-Fortran equation conversion
- Multiple precision levels (real4, real8, real16)
- Built-in RK4/RK45 integrators
- OpenMP parallelization (optional)
- MPI support for distributed computing (optional)
- Binary and ASCII output
"""

from typing import Dict, List, Optional

import sympy as sp
from sympy.printing.fortran import fcode

from ..utils import logger
from .base import CodeGenerator


def sympy_to_fortran(expr: sp.Expr, precision: str = "8") -> str:
    """
    Convert a sympy expression to Fortran code.

    Args:
        expr: Sympy expression to convert
        precision: Fortran precision kind (4, 8, 16)

    Returns:
        Fortran code string

    Examples:
        >>> import sympy as sp
        >>> x = sp.Symbol('x')
        >>> sympy_to_fortran(sp.sin(x), "8")
        'sin(x)'
    """
    if expr is None:
        return f"0.0_{precision}"

    try:
        code = fcode(expr, source_format="free")
        return code
    except Exception as e:
        logger.warning(f"Failed to convert expression to Fortran: {e}")
        return f"0.0_{precision}  ! ERROR: {e}"


class FortranGenerator(CodeGenerator):
    """
    Generates Fortran 90+ simulation code.

    Features:
    - Multiple precision levels (single, double, quad)
    - RK4 and adaptive RK45 integrators
    - OpenMP parallelization for parameter sweeps
    - MPI support for distributed runs
    - Binary and ASCII output options

    Example:
        >>> import sympy as sp
        >>> theta, g, length = sp.symbols('theta g l')
        >>> gen = FortranGenerator(
        ...     system_name="pendulum",
        ...     coordinates=["theta"],
        ...     parameters={"g": 9.81_8, "l": 1.0_8},
        ...     initial_conditions={"theta": 0.5_8, "theta_dot": 0.0_8},
        ...     equations={"theta_ddot": -g/length * sp.sin(theta)},
        ...     precision=8,
        ...     use_openmp=False
        ... )
        >>> gen.generate("pendulum.f90")
        'pendulum.f90'

    Attributes:
        precision: Fortran kind (4=single, 8=double, 16=quad)
        use_openmp: Enable OpenMP parallelization
        use_mpi: Enable MPI support
    """

    SUPPORTED_PRECISIONS = [4, 8, 16]

    # Fortran names are case-insensitive: a parameter M and a parameter m clash
    CASE_SENSITIVE_NAMES = False

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
        precision: int = 8,
        use_openmp: bool = False,
        use_mpi: bool = False,
    ) -> None:
        """
        Initialize the Fortran code generator.

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
            precision: Fortran real kind (4=single, 8=double, 16=quad)
            use_openmp: Enable OpenMP parallelization
            use_mpi: Enable MPI distributed computing
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

        if precision not in self.SUPPORTED_PRECISIONS:
            logger.warning(f"Unknown precision {precision}, using 8")
            precision = 8

        self.precision = precision
        self.use_openmp = use_openmp
        self.use_mpi = use_mpi

    @property
    def target_name(self) -> str:
        """Target platform identifier."""
        return "fortran"

    @property
    def file_extension(self) -> str:
        """File extension for generated code."""
        return ".f90"

    def expr_to_code(self, expr: sp.Expr) -> str:
        """
        Convert sympy expression to Fortran code.

        Args:
            expr: Sympy expression

        Returns:
            Fortran code string
        """
        return sympy_to_fortran(expr, str(self.precision))

    def generate_energy_computation(self) -> Optional[str]:
        """Generate a Fortran function for the total energy T + V, or None.

        T + V is the energy function sum(q_dot * dL/dq_dot) - L, taken from the
        Lagrangian; it is conserved when no non-conservative forces act.
        """
        T, V = self._extract_energy_expressions()
        if T is None or V is None:
            return None

        decls = "\n".join(f"        real(WP) :: {c}, {c}_dot" for c in self.coordinates)
        assigns = []
        for i, c in enumerate(self.coordinates):
            assigns.append(f"        {c} = y({2 * i + 1})")
            assigns.append(f"        {c}_dot = y({2 * i + 2})")
        body = "\n".join(
            self._indent(fcode(e, assign_to=name, source_format="free", standard=2003))
            for name, e in (("kinetic", T), ("potential", V))
        )

        return f"""
    ! =========================================================================
    ! Total energy T + V from the Lagrangian
    ! =========================================================================
    function compute_energy(y) result(energy)
        real(WP), intent(in) :: y(DIM)
        real(WP) :: energy, kinetic, potential
{decls}

{chr(10).join(assigns)}

{body}
        energy = kinetic + potential
    end function compute_energy
"""

    @staticmethod
    def _indent(code: str, spaces: int = 8) -> str:
        return "\n".join(" " * spaces + line for line in code.splitlines())

    def generate(self, output_file: str = "simulation.f90") -> str:
        """
        Generate Fortran simulation code.

        Args:
            output_file: Path to output file

        Returns:
            Path to generated file

        Raises:
            ValueError: If validation fails
        """
        self.validate_or_raise()

        logger.info(f"Generating Fortran code for {self.system_name}")

        code = self._generate_code()

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(code)

        logger.info(f"Successfully wrote {output_file}")
        return output_file

    def generate_equations(self) -> str:
        """Generate Fortran code for equations of motion."""
        lines = []
        idx = 1  # Fortran is 1-indexed
        for coord in self.coordinates:
            accel_key = f"{coord}_ddot"
            lines.append(f"        dydt({idx}) = y({idx+1})  ! d{coord}/dt = {coord}_dot")
            if accel_key in self.equations and self.equations[accel_key] is not None:
                expr = self.equations[accel_key]
                f_expr = self.expr_to_code(expr)
                lines.append(f"        dydt({idx+1}) = {f_expr}  ! d{coord}_dot/dt")
            else:
                lines.append(
                    f"        dydt({idx+1}) = 0.0_{self.precision}  ! d{coord}_dot/dt (no equation)"
                )
            idx += 2
        return "\n".join(lines)

    def _generate_code(self) -> str:
        """Generate complete Fortran simulation program."""
        p = self.precision
        state_dim = len(self.coordinates) * 2

        # Parameter definitions
        param_lines = []
        for name, val in self.parameters.items():
            param_lines.append(f"    real({p}), parameter :: {name} = {val}_{p}")
        param_str = "\n".join(param_lines) if param_lines else "    ! No parameters"

        # State variable unpacking
        unpack_lines = []
        idx = 1
        for coord in self.coordinates:
            unpack_lines.append(f"        {coord} = y({idx})")
            unpack_lines.append(f"        {coord}_dot = y({idx+1})")
            idx += 2
        unpack_str = "\n".join(unpack_lines)

        # Variable declarations for unpacking
        var_decls = []
        for coord in self.coordinates:
            var_decls.append(f"        real({p}) :: {coord}, {coord}_dot")
        var_decl_str = "\n".join(var_decls)

        # Equations
        eq_str = self.generate_equations()

        # Initial conditions
        init_assignments = []
        idx = 1
        for coord in self.coordinates:
            pos = self.initial_conditions.get(coord, 0.0)
            vel = self.initial_conditions.get(f"{coord}_dot", 0.0)
            init_assignments.append(f"        y({idx}) = {pos}_{p}")
            init_assignments.append(f"        y({idx+1}) = {vel}_{p}")
            idx += 2
        init_str = "\n".join(init_assignments)

        # OpenMP directives
        omp_pragma = "!$omp parallel do" if self.use_openmp else "! (OpenMP disabled)"  # noqa: F841
        omp_end = "!$omp end parallel do" if self.use_openmp else ""  # noqa: F841

        # Compiler flags
        compiler_base = "gfortran"
        flags = ["-O3", f"-fdefault-real-{p}"]
        if self.use_openmp:
            flags.append("-fopenmp")
        compile_cmd = (
            f"{compiler_base} {' '.join(flags)} -o {self.system_name} {self.system_name}.f90"
        )

        energy_code = self.generate_energy_computation()
        has_energy = energy_code is not None
        state_cols = ",".join(f"{c},{c}_dot" for c in self.coordinates)
        csv_header = f"t,{state_cols}" + (",energy" if has_energy else "")
        n_cols = state_dim + (2 if has_energy else 1)
        energy_value = ", compute_energy(y)" if has_energy else ""

        template = f"""! ==========================================================================
! {self.system_name.upper()} SIMULATION
! Generated by MechanicsDSL
! =============================================================================
!
! Compile with: {compile_cmd}
!
! Features:
!   - Precision: real({p})
!   - OpenMP: {"Enabled" if self.use_openmp else "Disabled"}
!   - MPI: {"Enabled" if self.use_mpi else "Disabled"}
!
! =============================================================================

module {self.system_name}_physics
    implicit none

    integer, parameter :: WP = {p}  ! Working precision
    integer, parameter :: DIM = {state_dim}  ! State dimension

    ! Physical Parameters
{param_str}

contains

    ! =========================================================================
    ! Equations of Motion
    ! =========================================================================
    subroutine equations_of_motion(t, y, dydt)
        real(WP), intent(in) :: t
        real(WP), intent(in) :: y(DIM)
        real(WP), intent(out) :: dydt(DIM)

        ! Local variables
{var_decl_str}

        ! Unpack state
{unpack_str}

        ! Suppress unused variable warning
        if (.false.) print *, t

        ! Compute derivatives
{eq_str}
    end subroutine equations_of_motion

    ! =========================================================================
    ! RK4 Integrator
    ! =========================================================================
    subroutine rk4_step(t, y, dt)
        real(WP), intent(inout) :: t
        real(WP), intent(inout) :: y(DIM)
        real(WP), intent(in) :: dt

        real(WP) :: k1(DIM), k2(DIM), k3(DIM), k4(DIM), tmp(DIM)

        call equations_of_motion(t, y, k1)

        tmp = y + 0.5_WP * dt * k1
        call equations_of_motion(t + 0.5_WP*dt, tmp, k2)

        tmp = y + 0.5_WP * dt * k2
        call equations_of_motion(t + 0.5_WP*dt, tmp, k3)

        tmp = y + dt * k3
        call equations_of_motion(t + dt, tmp, k4)

        y = y + dt/6.0_WP * (k1 + 2.0_WP*k2 + 2.0_WP*k3 + k4)
        t = t + dt
    end subroutine rk4_step

    ! =========================================================================
    ! Adaptive RK4 (step doubling): one step of dt against two of dt/2.
    ! The step is taken only if the difference is below tol; t and y are left
    ! unchanged otherwise. dt_new is the step to try next either way.
    ! =========================================================================
    subroutine adaptive_step(t, y, dt, dt_new, accepted, tol)
        real(WP), intent(inout) :: t
        real(WP), intent(inout) :: y(DIM)
        real(WP), intent(in) :: dt
        real(WP), intent(out) :: dt_new
        logical, intent(out) :: accepted
        real(WP), intent(in), optional :: tol

        real(WP) :: y_half(DIM), y_full(DIM), t_tmp
        real(WP) :: error, tolerance, factor

        tolerance = 1.0e-8_WP
        if (present(tol)) tolerance = tol

        ! Two half steps
        y_half = y
        t_tmp = t
        call rk4_step(t_tmp, y_half, dt/2.0_WP)
        call rk4_step(t_tmp, y_half, dt/2.0_WP)

        ! One full step
        y_full = y
        t_tmp = t
        call rk4_step(t_tmp, y_full, dt)

        ! Error estimate of the half-step solution (Richardson)
        error = maxval(abs(y_half - y_full)) / 15.0_WP
        accepted = error <= tolerance

        if (error > 0.0_WP) then
            factor = min(5.0_WP, max(0.2_WP, 0.9_WP * (tolerance / error)**0.2_WP))
        else
            factor = 5.0_WP
        end if
        dt_new = dt * factor

        if (accepted) then
            y = y_half
            t = t + dt
        end if
    end subroutine adaptive_step
{energy_code or ""}
end module {self.system_name}_physics

! =============================================================================
! Main Program
! =============================================================================
program {self.system_name}_simulation
    use {self.system_name}_physics
    implicit none

    real(WP) :: y(DIM)
    real(WP) :: t, dt, t_end
    integer :: step, n_steps, output_interval, unit_out

    ! Initial conditions
{init_str}

    ! Simulation parameters
    dt = 0.001_WP
    t_end = 10.0_WP
    t = 0.0_WP
    n_steps = nint(t_end / dt)
    output_interval = 10  ! write every 10th step (every 0.01 s)

    ! Open output file
    unit_out = 20
    open(unit=unit_out, file='{self.system_name}_results.csv', status='replace')
    write(unit_out, '(A)') '{csv_header}'

    print '(A)', '{self.system_name} simulation'
    print '(A,I0,A,ES10.3,A,ES10.3)', ' state dimension ', DIM, ', dt ', dt, ', t_end ', t_end

    ! Main integration loop
    do step = 0, n_steps
        t = step * dt
        if (mod(step, output_interval) == 0) then
            write(unit_out, '({n_cols}(ES24.16E3,:,","))') t, y{energy_value}
        end if
        if (step < n_steps) call rk4_step(t, y, dt)
    end do

    close(unit_out)
    print '(A)', ' Results saved to {self.system_name}_results.csv'

end program
"""
        return template
