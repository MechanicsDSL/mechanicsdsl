Core API Reference
==================

The ``mechanics_dsl.core`` package contains the fundamental components of the 
MechanicsDSL framework: the compiler, parser, symbolic engine, and numerical solver.

.. contents:: Contents
   :local:
   :depth: 2

PhysicsCompiler
---------------

.. py:class:: mechanics_dsl.core.PhysicsCompiler

   The main entry point for MechanicsDSL. Orchestrates the entire pipeline from 
   DSL source code to simulation results and visualization.

   **Example:**

   .. code-block:: python

      from mechanics_dsl import PhysicsCompiler
      
      compiler = PhysicsCompiler()
      result = compiler.compile(source)
      solution = compiler.simulate((0, 10))
      compiler.visualize(solution)

   .. py:method:: compile(source: str) -> dict

      Compile DSL source code into an executable simulation.

      :param source: DSL source code string
      :type source: str
      :returns: Compilation result dictionary
      :rtype: dict

      **Result Dictionary:**

      .. code-block:: python

         {
             'success': bool,           # True if compilation succeeded
             'system_name': str,        # Name from \system{} command
             'coordinates': list,       # List of generalized coordinates
             'parameters': dict,        # Parameter name -> value mapping
             'initial_conditions': dict, # Initial state
             'equations': dict,         # Symbolic equations of motion
             'error': str,              # Error message if failed (optional)
         }

   .. py:method:: simulate(t_span: tuple, num_points: int = 1000, method: str = None, rtol: float = None, atol: float = None, detect_stiff: bool = True) -> dict

      Run numerical simulation of the compiled system.

      :param t_span: Time interval (t_start, t_end)
      :type t_span: tuple
      :param num_points: Number of output time points
      :type num_points: int
      :param method: Integration method (RK45, LSODA, Radau, etc.)
      :type method: str
      :param rtol: Relative tolerance
      :type rtol: float
      :param atol: Absolute tolerance  
      :type atol: float
      :param detect_stiff: Automatically detect stiff systems
      :type detect_stiff: bool
      :returns: Simulation result dictionary
      :rtype: dict

      **Result Dictionary:**

      .. code-block:: python

         {
             'success': bool,      # True if simulation succeeded
             't': np.ndarray,      # Time points
             'y': np.ndarray,      # State vector (2n x m matrix)
             'coordinates': list,  # Coordinate names
             'nfev': int,          # Function evaluations
             'status': int,        # Solver status code
             'message': str,       # Status message
         }

   .. py:method:: visualize(solution: dict = None, animation_type: str = 'auto') -> None

      Create visualization of simulation results.

      :param solution: Simulation result (uses last simulation if None)
      :param animation_type: 'auto', 'pendulum', 'oscillator', 'phase_space'

   .. py:method:: plot_energy(solution: dict = None) -> None

      Plot energy conservation analysis.

   .. py:method:: compile_to_cpp(filename: str = 'simulation.cpp', target: str = 'standard') -> str

      Generate C++ code for the compiled system.

      :param filename: Output file path
      :param target: 'standard', 'openmp', or 'wasm'
      :returns: Path to generated file

   .. py:method:: set_parameter(name: str, value: float) -> None

      Update a parameter value after compilation.

   .. py:method:: set_initial_condition(name: str, value: float) -> None

      Update an initial condition after compilation.


MechanicsParser
---------------

.. py:class:: mechanics_dsl.core.MechanicsParser

   Parser for the MechanicsDSL language. Converts tokenized input into an 
   Abstract Syntax Tree (AST).

   **Example:**

   .. code-block:: python

      from mechanics_dsl.core.parser import tokenize, MechanicsParser
      
      tokens = tokenize(source)
      parser = MechanicsParser(tokens)
      ast = parser.parse()

   .. py:method:: __init__(tokens: List[Token])

      Initialize parser with token list.

      :param tokens: List of Token objects from tokenize()

   .. py:method:: parse() -> List[ASTNode]

      Parse the token stream into an AST.

      :returns: List of AST nodes representing the program
      :raises: ParserError if syntax error encountered


SymbolicEngine
--------------

.. py:class:: mechanics_dsl.core.SymbolicEngine

   Symbolic mathematics engine built on SymPy. Handles:

   - Conversion from AST expressions to SymPy
   - Euler-Lagrange equation derivation
   - Hamilton's equations derivation
   - Symbolic simplification and caching

   **Example:**

   .. code-block:: python

      from mechanics_dsl.core.symbolic import SymbolicEngine
      
      engine = SymbolicEngine()
      engine.set_coordinates(['theta'])
      engine.set_lagrangian(lagrangian_expr)
      equations = engine.derive_equations()

   .. py:method:: set_coordinates(coords: List[str]) -> None

      Set the generalized coordinates.

   .. py:method:: set_lagrangian(expr) -> None

      Set the Lagrangian expression.

   .. py:method:: set_hamiltonian(expr) -> None

      Set the Hamiltonian expression.

   .. py:method:: derive_euler_lagrange() -> Dict[str, sp.Expr]

      Derive Euler-Lagrange equations.

      :returns: Mapping of acceleration symbols to expressions

   .. py:method:: derive_hamiltons_equations() -> Dict[str, sp.Expr]

      Derive Hamilton's equations of motion.

   .. py:method:: solve_for_accelerations() -> Dict[str, sp.Expr]

      Solve the EOM for acceleration terms.


NumericalSimulator
------------------

.. py:class:: mechanics_dsl.core.NumericalSimulator

   Numerical integration engine using SciPy solvers.

   **Example:**

   .. code-block:: python

      from mechanics_dsl.core.solver import NumericalSimulator
      
      sim = NumericalSimulator(symbolic_engine)
      sim.set_parameters(params)
      sim.set_initial_conditions(ic)
      solution = sim.simulate((0, 10))

   .. py:method:: __init__(symbolic_engine: SymbolicEngine)

      Initialize with a compiled symbolic engine.

   .. py:method:: set_parameters(params: Dict[str, float]) -> None

      Set physical parameter values.

   .. py:method:: set_initial_conditions(ic: Dict[str, float]) -> None

      Set initial state values.

   .. py:method:: compile_equations() -> None

      Compile symbolic equations to numerical functions.

   .. py:method:: simulate(t_span: tuple, num_points: int = 1000, **kwargs) -> dict

      Run the numerical simulation.

   .. py:method:: detect_stiffness() -> str

      Automatically detect if system is stiff and suggest solver.


AST Node Types
--------------

The parser produces the following AST node types:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Node Type
     - Purpose
     - Key Attributes
   * - ``SystemDef``
     - System name
     - ``name``
   * - ``VarDef``
     - Variable definition
     - ``name``, ``var_type``, ``unit``
   * - ``ParameterDef``
     - Parameter definition
     - ``name``, ``value``, ``unit``
   * - ``LagrangianDef``
     - Lagrangian expression
     - ``expression``
   * - ``HamiltonianDef``
     - Hamiltonian expression
     - ``expression``
   * - ``ConstraintDef``
     - Holonomic constraint
     - ``expression``
   * - ``ForceDef``
     - External force
     - ``coordinate``, ``expression``
   * - ``DampingDef``
     - Damping term
     - ``coefficient``
   * - ``InitialCondition``
     - Initial state
     - ``assignments``
   * - ``FluidDef``
     - Fluid region
     - ``name``, ``properties``
   * - ``BoundaryDef``
     - Boundary region
     - ``name``, ``regions``

Expression Types
~~~~~~~~~~~~~~~~

Mathematical expressions are represented by:

- ``NumberExpr``: Numeric literals
- ``IdentExpr``: Variable/parameter identifiers
- ``BinaryOpExpr``: Binary operations (+, -, *, /, ^)
- ``UnaryOpExpr``: Unary operations (-)
- ``FractionExpr``: Fractions (\\frac{}{})
- ``DerivativeVarExpr``: Time derivatives (\\dot{}, q_dot)
- ``FunctionCallExpr``: Function calls (\\sin, \\cos, etc.)
- ``GreekLetterExpr``: Greek letters (\\theta, \\phi)


Error Handling
--------------

.. py:exception:: mechanics_dsl.core.parser.ParserError

   Raised when the parser encounters invalid syntax.

   **Attributes:**

   - ``message``: Error description
   - ``line``: Line number (if available)
   - ``column``: Column number (if available)


Token Types
-----------

The tokenizer recognizes these token types:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Token Type
     - Pattern
   * - ``COMMAND``
     - ``\name`` (backslash followed by word)
   * - ``LBRACE``
     - ``{``
   * - ``RBRACE``
     - ``}``
   * - ``NUMBER``
     - Integer or floating point
   * - ``IDENT``
     - Word characters
   * - ``OPERATOR``
     - ``+``, ``-``, ``*``, ``/``, ``^``, ``=``
   * - ``LPAREN``
     - ``(``
   * - ``RPAREN``
     - ``)``
   * - ``COMMA``
     - ``,``
   * - ``UNDERSCORE``
     - ``_``
   * - ``COMMENT``
     - ``%`` to end of line
   * - ``EOF``
     - End of input
