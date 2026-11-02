"""
Base Code Generator for MechanicsDSL

Provides abstract interface for all code generation backends with:
- Abstract sympy-to-target expression conversion
- Equation validation
- Energy conservation verification code generation
- Numerical stability checks
- Comprehensive type hints

All code generators (C++, Python, WASM, CUDA, etc.) should inherit
from this class and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import sympy as sp
from sympy import Symbol, simplify, symbols

# Names a coordinate or parameter must not take in generated code (see
# CodeGenerator._normalise_names). Three groups: identifiers the generated programs
# declare where user names are visible; keywords of the target languages; and C/C++
# library globals that a global ``const double`` of the same name would redeclare.
_PROGRAM_NAMES = """
    y t dt dydt du u p state states tmp temp temp_y k1 k2 k3 k4 i j n s idx step steps
    n_steps t_end tEnd t_span tspan t_eval dim DIM STATE_DIM NUM_SYSTEMS num_systems
    BATCH_SIZE WP DP PI params PARAMETERS result results sol solution y0 u0 prob error tol
    tolerance dt_new newDt currentDt y_half y_full y_temp yFull energy kinetic potential
    file outfile output_interval log_interval unit_out now lastMicros initial initial_state
    base_ic systems data config options opts method integrator trackEnergy d_state d_states
    h_state h_states h_params my_state elapsed start_time main setup loop Serial
"""
_KEYWORDS = """
    auto break case char const continue default do double else enum extern float for goto
    if inline int long register restrict return short signed sizeof static struct switch
    typedef union unsigned void volatile while bool true false class delete new this
    template typename namespace operator private public protected friend virtual try catch
    throw using explicit mutable export asm and or not xor
    as async await crate dyn fn impl in let match mod move mut pub ref self Self super trait
    type unsafe use where abstract become box final macro override priv typeof unsized yield
    False None True assert def del elif except finally from global import is lambda
    nonlocal pass raise with
    arguments debugger eval extends function implements instanceof interface null package
    var undefined NaN Infinity Math
    baremodule begin elseif end local module quote primitive
    classdef otherwise parfor persistent spmd pi eps inf Inf nan ans
"""
_C_LIBRARY = """
    gamma y1 yn j0 j1 jn time rand abs exp log div round floor remainder drem significand
    lgamma tgamma erf hypot trunc sin cos tan asin acos atan atan2 sinh cosh tanh sqrt pow
"""
RESERVED_NAMES = frozenset((_PROGRAM_NAMES + _KEYWORDS + _C_LIBRARY).split())


def accelerations_from_mass_matrix(
    M: sp.Matrix, F: sp.Matrix, coordinates: List[str], max_block: int = 3
) -> Dict[str, sp.Expr]:
    """
    Closed-form accelerations from M(q, q_dot) * q_ddot = F(q, q_dot), for export.

    The compiler keeps M and F unsolved for systems with many coordinates and
    solves the linear system numerically at each step. Generated programs need
    explicit accelerations, so M is split into independent blocks (coordinates
    that share no off-diagonal mass terms): 1x1 blocks are a division, and blocks
    of up to ``max_block`` coordinates are solved symbolically, as the compiler
    itself does for small systems. Particles, N-body problems and chains of
    masses have diagonal mass matrices and always export.

    Raises:
        ValueError: if a block of coupled coordinates is larger than max_block.
    """
    n = len(coordinates)
    # Connected components of the nonzero pattern of M
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if M[i, j] != 0 or M[j, i] != 0:
                parent[find(i)] = find(j)
    blocks: Dict[int, List[int]] = {}
    for i in range(n):
        blocks.setdefault(find(i), []).append(i)

    accelerations: Dict[str, sp.Expr] = {}
    for idx in blocks.values():
        if len(idx) == 1:
            i = idx[0]
            accelerations[f"{coordinates[i]}_ddot"] = sp.cancel(F[i, 0] / M[i, i])
            continue
        if len(idx) > max_block:
            names = ", ".join(coordinates[i] for i in idx)
            raise ValueError(
                f"Coordinates {names} are coupled through the mass matrix, and "
                f"solving {len(idx)} coupled accelerations in closed form is too "
                "large to generate code for. Code export supports systems whose "
                f"coupled groups have at most {max_block} coordinates."
            )
        Mb = sp.Matrix([[M[i, j] for j in idx] for i in idx])
        Fb = sp.Matrix([F[i, 0] for i in idx])
        sol = Mb.LUsolve(Fb)
        for k, i in enumerate(idx):
            accelerations[f"{coordinates[i]}_ddot"] = sp.simplify(sol[k])
    return accelerations


class CodeGenerator(ABC):
    """
    Abstract base class for code generation backends.

    All code generators (C++, Python, WASM, CUDA, etc.) should inherit
    from this class and implement the required abstract methods.

    Attributes:
        system_name: Name of the physics system
        coordinates: List of generalized coordinate names
        parameters: Physical parameters as name -> value dict
        initial_conditions: Initial state as name -> value dict
        equations: Acceleration equations as coord_ddot -> sympy.Expr
        lagrangian: Optional Lagrangian expression for energy checks
        hamiltonian: Optional Hamiltonian for energy conservation

    Example:
        >>> from mechanics_dsl.codegen import PythonGenerator
        >>> import sympy as sp
        >>> theta, g, length = sp.symbols('theta g l')
        >>> gen = PythonGenerator(
        ...     system_name="pendulum",
        ...     coordinates=["theta"],
        ...     parameters={"g": 9.81, "l": 1.0},
        ...     initial_conditions={"theta": 0.5, "theta_dot": 0.0},
        ...     equations={"theta_ddot": -g/length * sp.sin(theta)}
        ... )
        >>> gen.validate()
        (True, [])
    """

    # Class-level configuration
    #: False for targets such as Fortran where ``M`` and ``m`` are the same name
    CASE_SENSITIVE_NAMES: bool = True
    SUPPORTED_INTEGRATORS: List[str] = ["rk4", "euler", "verlet", "rk45"]
    DEFAULT_TIMESTEP: float = 0.01
    DEFAULT_DURATION: float = 10.0

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
    ) -> None:
        """
        Initialize the code generator.

        Args:
            system_name: Name of the physics system (used for class/function names)
            coordinates: List of generalized coordinate variable names
            parameters: Physical parameters as name -> numerical value
            initial_conditions: Initial state values (coord and coord_dot)
            equations: Acceleration equations as "{coord}_ddot" -> sympy expression
            lagrangian: Optional Lagrangian L(q, q_dot) for energy verification
            hamiltonian: Optional Hamiltonian H(q, p) for conservation checks
            forces: Optional list of non-conservative force expressions
            constraints: Optional list of holonomic constraint expressions
        """
        self.system_name = system_name
        self.coordinates = list(coordinates or [])
        self.parameters = dict(parameters or {})
        self.initial_conditions = dict(initial_conditions or {})
        self.equations = dict(equations or {})
        self.lagrangian = lagrangian
        self.hamiltonian = hamiltonian
        self.forces = list(forces or [])
        self.constraints = list(constraints or [])

        # Generated programs use names of their own (y, dt, k1..k4, ...), keywords of
        # the target language and library functions such as C's gamma(). Rename any
        # coordinate or parameter that would collide, and give every symbol plain
        # assumptions so that it matches by name (see _normalise_names).
        self.renamed: Dict[str, str] = {}
        self._normalise_names()

        # Compute state dimension
        self.state_dim = len(self.coordinates) * 2

        # Create sympy symbols for all coordinates and their derivatives
        self._symbols: Dict[str, Symbol] = {}
        for coord in self.coordinates:
            self._symbols[coord] = symbols(coord)
            self._symbols[f"{coord}_dot"] = symbols(f"{coord}_dot")
            self._symbols[f"{coord}_ddot"] = symbols(f"{coord}_ddot")

        # Add parameter symbols
        for param in self.parameters:
            self._symbols[param] = symbols(param)

    def _normalise_names(self) -> None:
        """
        Make user names safe to emit and make every symbol match by name.

        1. Symbols from the DSL parser carry assumptions (``real=True``). A plain
           ``Symbol('x')`` is a different object, so lookups such as ``diff(L, x_dot)``
           silently returned zero, and SymPy's Rust printer drops the brackets
           round a sum multiplied by ``.recip()`` for real symbols, which changed
           the equations. All expressions are rebuilt with plain symbols.
        2. A coordinate or parameter whose name the generated program already uses
           (the state array ``y``, the stages ``k1``..``k4``), that is a keyword in
           one of the target languages (``lambda``, ``end``, ``fn``) or that clashes
           with a C library function (``gamma``, ``y0``, ``j1``) gets a trailing
           underscore. The same names are used for every target, and
           ``self.renamed`` maps each new name back to the original.
        """
        key = (lambda n: n) if self.CASE_SENSITIVE_NAMES else (lambda n: n.lower())
        reserved = {key(n) for n in RESERVED_NAMES}
        used: set = set()
        rename: Dict[str, str] = {}

        def pick(name: str, extra_suffixes: Tuple[str, ...] = ()) -> str:
            new = name
            while any(
                key(new + sfx) in reserved or key(new + sfx) in used
                for sfx in ("",) + extra_suffixes
            ):
                new += "_"
            for sfx in ("",) + extra_suffixes:
                used.add(key(new + sfx))
            return new

        for coord in self.coordinates:
            new = pick(coord, ("_dot", "_ddot"))
            if new != coord:
                rename[coord] = new
                rename[f"{coord}_dot"] = f"{new}_dot"
                rename[f"{coord}_ddot"] = f"{new}_ddot"
        for param in self.parameters:
            if param == "t":
                continue
            new = pick(param)
            if new != param:
                rename[param] = new

        def plain(expr):
            if expr is None or not isinstance(expr, sp.Basic):
                return expr
            subs = {
                s: Symbol(rename.get(s.name, s.name))
                for s in expr.free_symbols
                if isinstance(s, Symbol)
            }
            return expr.xreplace(subs) if subs else expr

        self.coordinates = [rename.get(c, c) for c in self.coordinates]
        self.parameters = {rename.get(k, k): v for k, v in self.parameters.items()}
        self.initial_conditions = {rename.get(k, k): v for k, v in self.initial_conditions.items()}
        if isinstance(self.equations, dict):
            self.equations = {rename.get(k, k): plain(v) for k, v in self.equations.items()}
        self.lagrangian = plain(self.lagrangian)
        self.hamiltonian = plain(self.hamiltonian)
        self.forces = [plain(f) for f in self.forces]
        self.constraints = [plain(c) for c in self.constraints]
        self.renamed = {new: old for old, new in rename.items()}

    @property
    @abstractmethod
    def target_name(self) -> str:
        """
        Name of the target platform.

        Returns:
            Target platform identifier (e.g., 'cpp', 'python', 'cuda', 'wasm')
        """

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """
        File extension for generated code.

        Returns:
            File extension including dot (e.g., '.cpp', '.py', '.cu')
        """

    @abstractmethod
    def expr_to_code(self, expr: sp.Expr) -> str:
        """
        Convert a sympy expression to target language code.

        This is the core method that each generator must implement to convert
        symbolic math expressions to executable code in the target language.

        Args:
            expr: Sympy expression to convert

        Returns:
            String containing the expression in target language syntax

        Example:
            >>> expr = sp.sin(theta) * g / l
            >>> cpp_gen.expr_to_code(expr)
            'std::sin(theta) * g / l'
        """

    @abstractmethod
    def generate(self, output_file: str) -> str:
        """
        Generate code and write to file.

        Args:
            output_file: Output file path

        Returns:
            Path to generated file

        Raises:
            ValueError: If validation fails
            IOError: If file cannot be written
        """

    @abstractmethod
    def generate_equations(self) -> str:
        """
        Generate the equations of motion code.

        Returns:
            String containing equation code in target language

        Example output (C++):
            dydt[0] = y[1];  // dtheta/dt = theta_dot
            dydt[1] = -g/l * sin(y[0]);  // dtheta_dot/dt
        """

    def generate_parameters(self) -> str:
        """
        Generate parameter declarations in target language.

        Override in subclasses for target-specific syntax.

        Returns:
            String containing parameter declarations
        """
        lines = []
        for name, val in self.parameters.items():
            lines.append(f"// {name} = {val}")
        return "\n".join(lines)

    def generate_initial_conditions(self) -> str:
        """
        Generate initial condition values as comma-separated list.

        Override in subclasses for target-specific syntax.

        Returns:
            Comma-separated initial values in order: q1, q1_dot, q2, q2_dot, ...
        """
        vals: List[str] = []
        for coord in self.coordinates:
            pos = self.initial_conditions.get(coord, 0.0)
            vel = self.initial_conditions.get(f"{coord}_dot", 0.0)
            vals.extend([str(pos), str(vel)])
        return ", ".join(vals)

    def noncolliding_name(self, preferred: str) -> str:
        """
        Return `preferred`, or an underscore-prefixed variant if a coordinate
        (or its _dot) already uses that name.

        The generated state-vector array is conventionally called 'y', but a
        coordinate can be named 'y' too (2D motion). Unpacking such a coordinate
        would shadow the array with a scalar; naming the array with this helper
        avoids that. Returns `preferred` unchanged when there is no collision, so
        generated output is identical for the common case.
        """
        reserved = set(self.coordinates) | {f"{c}_dot" for c in self.coordinates}
        name = preferred
        while name in reserved:
            name = "_" + name
        return name

    def generate_state_unpacking(self, state_var: str = "y") -> str:
        """
        Generate code to unpack state vector into named variables.

        Args:
            state_var: Name of the state vector variable

        Returns:
            Code that extracts coord and coord_dot from state vector
        """
        lines: List[str] = []
        idx = 0
        for coord in self.coordinates:
            lines.append(f"    {coord} = {state_var}[{idx}];")
            lines.append(f"    {coord}_dot = {state_var}[{idx + 1}];")
            idx += 2
        return "\n".join(lines)

    def _extract_energy_expressions(self) -> Tuple[Optional[sp.Expr], Optional[sp.Expr]]:
        """
        Extract kinetic (T) and potential (V) energy from Lagrangian L = T - V.

        Uses Euler's theorem for homogeneous functions: for T quadratic in
        velocities, T = (1/2) * sum(q_dot_i * dL/dq_dot_i).

        Returns:
            Tuple of (T, V) as sympy expressions with parameters substituted,
            or (None, None) if Lagrangian is not available.
        """
        if self.lagrangian is None:
            return None, None

        L = self.lagrangian
        vel_syms = [self._symbols[f"{c}_dot"] for c in self.coordinates]

        # Euler's theorem: T = sum(q_dot * dL/dq_dot) / 2
        T = sum(v * sp.diff(L, v) for v in vel_syms) / 2
        T = sp.simplify(T)
        V = sp.simplify(T - L)

        # Substitute numerical parameter values
        param_subs = {self._symbols[k]: v for k, v in self.parameters.items() if k in self._symbols}
        T = T.subs(param_subs)
        V = V.subs(param_subs)

        return T, V

    def generate_energy_computation(self) -> Optional[str]:
        """
        Generate code to compute total energy (kinetic + potential).

        Used for conservation verification during simulation.

        Returns:
            Code to compute energy, or None if Lagrangian not available
        """
        if self.lagrangian is None:
            return None

        # Subclasses should override for language-specific output
        return "// Energy computation not implemented for this target"

    def generate_rk4_integrator(self) -> str:
        """
        Generate RK4 integration step code.

        Override in subclasses for optimized implementations.

        Returns:
            RK4 integrator code in target language
        """
        return "// RK4 integrator - override in subclass"

    def generate_euler_integrator(self) -> str:
        """
        Generate Euler integration step code (for debugging/comparison).

        Returns:
            Euler integrator code in target language
        """
        return "// Euler integrator - override in subclass"

    def generate_verlet_integrator(self) -> str:
        """
        Generate Velocity Verlet integration step code.

        Better for Hamiltonian systems (symplectic integrator).

        Returns:
            Verlet integrator code in target language
        """
        return "// Verlet integrator - override in subclass"

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate that the generator has all required data.

        Checks:
        - System name is provided
        - At least one coordinate exists
        - All coordinates have corresponding acceleration equations
        - No NaN or Inf in parameters
        - Initial conditions cover all state variables

        Returns:
            Tuple of (is_valid, list_of_error_messages)

        Example:
            >>> is_valid, errors = generator.validate()
            >>> if not is_valid:
            ...     print("Errors:", errors)
        """
        errors: List[str] = []

        # Required fields
        if not self.system_name:
            errors.append("system_name is required")
        if not self.coordinates:
            errors.append("at least one coordinate is required")

        # Check equations exist for all coordinates
        for coord in self.coordinates:
            accel_key = f"{coord}_ddot"
            if accel_key not in self.equations:
                errors.append(f"missing equation for {accel_key}")
            elif self.equations[accel_key] is None:
                errors.append(f"equation for {accel_key} is None")

        # Every symbol in the equations must be a coordinate, a velocity, a
        # parameter or the time t; anything else would be an undeclared name in
        # the generated program (for example a \defvar the compiler did not
        # recognise as a coordinate).
        known = {"t"} | set(self.parameters)
        for coord in self.coordinates:
            known |= {coord, f"{coord}_dot"}
        stray = set()
        for expr in self.equations.values():
            if isinstance(expr, sp.Basic):
                stray |= {s.name for s in expr.free_symbols if isinstance(s, Symbol)} - known
        if stray:
            names = ", ".join(sorted(self.renamed.get(n, n) for n in stray))
            errors.append(
                f"the equations use {names}, which is neither a coordinate nor a "
                "parameter (declare it with \\parameter{}, or give its \\defvar a "
                "coordinate type such as Position or Angle)"
            )

        # Check parameters for invalid values
        import math

        for name, val in self.parameters.items():
            if val is None:
                errors.append(f"parameter '{name}' is None")
            elif isinstance(val, (int, float)):
                if math.isnan(val):
                    errors.append(f"parameter '{name}' is NaN")
                elif math.isinf(val):
                    errors.append(f"parameter '{name}' is infinite")

        # Check initial conditions
        for coord in self.coordinates:
            if coord not in self.initial_conditions:
                # Warning, not error - defaults to 0
                pass
            if f"{coord}_dot" not in self.initial_conditions:
                # Warning, not error - defaults to 0
                pass

        return len(errors) == 0, errors

    def validate_or_raise(self) -> None:
        """
        Validate and raise ValueError if invalid.

        Raises:
            ValueError: If validation fails, with all error messages
        """
        is_valid, errors = self.validate()
        if not is_valid:
            raise ValueError(f"Code generator validation failed: {'; '.join(errors)}")

    def get_equation_complexity(self) -> Dict[str, int]:
        """
        Analyze complexity of equations (operation count).

        Useful for estimating computational cost and potential optimizations.

        Returns:
            Dict mapping equation names to operation counts
        """
        complexity: Dict[str, int] = {}
        for name, expr in self.equations.items():
            if expr is not None:
                complexity[name] = sp.count_ops(expr)
            else:
                complexity[name] = 0
        return complexity

    def get_free_symbols(self) -> set:
        """
        Get all free symbols used in equations.

        Returns:
            Set of sympy Symbol objects used in equations
        """
        all_symbols: set = set()
        for expr in self.equations.values():
            if expr is not None:
                all_symbols.update(expr.free_symbols)
        return all_symbols

    def substitute_parameters(self, expr: sp.Expr) -> sp.Expr:
        """
        Substitute numerical parameter values into expression.

        Args:
            expr: Sympy expression with parameter symbols

        Returns:
            Expression with parameters replaced by numbers
        """
        subs = {self._symbols.get(k, symbols(k)): v for k, v in self.parameters.items()}
        return expr.subs(subs)

    def simplify_equations(self) -> None:
        """
        Simplify all equations using sympy.simplify().

        Modifies equations in place. May be slow for complex expressions.
        """
        for key, expr in self.equations.items():
            if expr is not None:
                self.equations[key] = simplify(expr)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"system='{self.system_name}', "
            f"target='{self.target_name}', "
            f"coords={self.coordinates})"
        )

    def __str__(self) -> str:
        return (
            f"{self.target_name.upper()} Generator for '{self.system_name}'\n"
            f"  Coordinates: {', '.join(self.coordinates)}\n"
            f"  Parameters: {', '.join(f'{k}={v}' for k, v in self.parameters.items())}\n"
            f"  Equations: {len(self.equations)}"
        )
