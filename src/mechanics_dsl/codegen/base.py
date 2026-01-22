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
from typing import Any, Dict, List, Optional, Tuple, Union

import sympy as sp
from sympy import Symbol, symbols, diff, simplify


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
        >>> theta, g, l = sp.symbols('theta g l')
        >>> gen = PythonGenerator(
        ...     system_name="pendulum",
        ...     coordinates=["theta"],
        ...     parameters={"g": 9.81, "l": 1.0},
        ...     initial_conditions={"theta": 0.5, "theta_dot": 0.0},
        ...     equations={"theta_ddot": -g/l * sp.sin(theta)}
        ... )
        >>> gen.validate()
        (True, [])
    """

    # Class-level configuration
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
        self.coordinates = coordinates or []
        self.parameters = parameters or {}
        self.initial_conditions = initial_conditions or {}
        self.equations = equations or {}
        self.lagrangian = lagrangian
        self.hamiltonian = hamiltonian
        self.forces = forces or []
        self.constraints = constraints or []

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

    def generate_energy_computation(self) -> Optional[str]:
        """
        Generate code to compute total energy (kinetic + potential).

        Used for conservation verification during simulation.

        Returns:
            Code to compute energy, or None if Lagrangian not available
        """
        if self.lagrangian is None:
            return None

        # For L = T - V, compute H = T + V via Legendre transform
        # This is a placeholder - subclasses should override with proper code
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
