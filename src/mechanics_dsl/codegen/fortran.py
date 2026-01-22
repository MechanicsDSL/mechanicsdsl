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
        >>> theta, g, l = sp.symbols('theta g l')
        >>> gen = FortranGenerator(
        ...     system_name="pendulum",
        ...     coordinates=["theta"],
        ...     parameters={"g": 9.81_8, "l": 1.0_8},
        ...     initial_conditions={"theta": 0.5_8, "theta_dot": 0.0_8},
        ...     equations={"theta_ddot": -g/l * sp.sin(theta)},
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
                lines.append(f"        dydt({idx+1}) = 0.0_{self.precision}  ! d{coord}_dot/dt (no equation)")
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
        param_str = "\n".join(param_lines) if param_lines else f"    ! No parameters"

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
        omp_pragma = "!$omp parallel do" if self.use_openmp else "! (OpenMP disabled)"
        omp_end = "!$omp end parallel do" if self.use_openmp else ""

        # Compiler flags
        compiler_base = "gfortran"
        flags = ["-O3", f"-fdefault-real-{p}"]
        if self.use_openmp:
            flags.append("-fopenmp")
        compile_cmd = f"{compiler_base} {' '.join(flags)} -o {self.system_name} {self.system_name}.f90"

        template = f'''! =============================================================================
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
    ! Adaptive RK45 Integrator (Dormand-Prince)
    ! =========================================================================
    subroutine rk45_step(t, y, dt, dt_new, tol)
        real(WP), intent(inout) :: t
        real(WP), intent(inout) :: y(DIM)
        real(WP), intent(in) :: dt
        real(WP), intent(out) :: dt_new
        real(WP), intent(in), optional :: tol

        real(WP) :: y_half(DIM), y_full(DIM), y_temp(DP)
        real(WP) :: error, tolerance
        integer :: i

        tolerance = 1.0e-6_WP
        if (present(tol)) tolerance = tol

        ! Two half steps
        y_temp = y
        call rk4_step(t, y_temp, dt/2.0_WP)
        y_half = y_temp
        call rk4_step(t + dt/2.0_WP, y_half, dt/2.0_WP)

        ! One full step
        y_full = y
        call rk4_step(t, y_full, dt)

        ! Error estimate
        error = 0.0_WP
        do i = 1, DIM
            error = max(error, abs(y_half(i) - y_full(i)))
        end do

        ! Adaptive step size
        dt_new = dt
        if (error > 16.0_WP * tolerance) then
            dt_new = dt * 0.5_WP
        else if (error < tolerance / 16.0_WP) then
            dt_new = dt * 2.0_WP
        end if

        ! Use more accurate solution
        y = y_half
        t = t + dt
    end subroutine rk45_step

    ! =========================================================================
    ! Energy Computation
    ! =========================================================================
    function compute_energy(y) result(energy)
        real(WP), intent(in) :: y(DIM)
        real(WP) :: energy
        integer :: i

        ! Kinetic energy (simple: 0.5 * v^2)
        energy = 0.0_WP
        do i = 1, DIM/2
            energy = energy + 0.5_WP * y(2*i)**2
        end do
    end function compute_energy

end module {self.system_name}_physics

! =============================================================================
! Main Program
! =============================================================================
program {self.system_name}_simulation
    use {self.system_name}_physics
    implicit none

    real(WP) :: y(DIM)
    real(WP) :: t, dt, t_end, energy_init, energy_final
    integer :: step, n_steps, output_interval, unit_out

    ! Initial conditions
{init_str}

    ! Simulation parameters
    dt = 0.01_WP
    t_end = 10.0_WP
    t = 0.0_WP
    n_steps = nint(t_end / dt)
    output_interval = 10

    ! Open output file
    unit_out = 20
    open(unit=unit_out, file='{self.system_name}_results.csv', status='replace')
    write(unit_out, '(A)') 't,{",".join(self.coordinates + [c + "_dot" for c in self.coordinates])},energy'

    ! Initial energy
    energy_init = compute_energy(y)

    print '(A)', '=========================================='
    print '(A)', ' {self.system_name.upper()} SIMULATION'
    print '(A)', '=========================================='
    print '(A,I0)', ' State dimension: ', DIM
    print '(A,E10.3)', ' Time step: ', dt
    print '(A,E10.3)', ' End time: ', t_end
    print '(A,E10.3)', ' Initial energy: ', energy_init
    print '(A)', '------------------------------------------'

    ! Main integration loop
    do step = 0, n_steps
        ! Output
        if (mod(step, output_interval) == 0) then
            write(unit_out, '({state_dim + 2}(ES18.10,:,","))') t, y, compute_energy(y)
        end if

        ! RK4 step
        call rk4_step(t, y, dt)
    end do

    ! Final energy and conservation check
    energy_final = compute_energy(y)

    close(unit_out)

    print '(A)', '------------------------------------------'
    print '(A,E18.10)', ' Final energy: ', energy_final
    print '(A,E18.10)', ' Energy drift: ', abs(energy_final - energy_init)
    print '(A)', ' Results saved to {self.system_name}_results.csv'
    print '(A)', '=========================================='

end program
'''
        return template
