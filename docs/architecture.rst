Compiler Architecture
=====================

Technical overview of the MechanicsDSL compiler pipeline.

Pipeline Overview
-----------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────┐
   │                    DSL Source Code                      │
   └─────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │  TOKENIZER (lexer.py)                                   │
   │  - Break source into tokens                             │
   │  - Handle LaTeX commands, numbers, operators            │
   └─────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │  PARSER (parser.py)                                     │
   │  - Build Abstract Syntax Tree (AST)                     │
   │  - Validate syntax                                      │
   │  - Error recovery                                       │
   └─────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │  SEMANTIC ANALYZER (semantic.py)                        │
   │  - Type checking                                        │
   │  - Unit inference                                       │
   │  - Symbol resolution                                    │
   └─────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │  SYMBOLIC ENGINE (symbolic.py)                          │
   │  - Convert to SymPy expressions                         │
   │  - Derive equations of motion                           │
   │  - Simplify equations                                   │
   └─────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │  CODE GENERATOR                                         │
   │  ├── NumPy (runtime)                                    │
   │  ├── C++ (compile-time)                                 │
   │  ├── WebAssembly                                        │
   │  └── CUDA (planned)                                     │
   └─────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │  SOLVER / EXECUTION                                     │
   │  - Numerical integration                                │
   │  - Result collection                                    │
   └─────────────────────────────────────────────────────────┘

Tokenizer
---------

The tokenizer (``core/lexer.py``) handles LaTeX-style input:

**Token Types**:

- ``COMMAND``: ``\system``, ``\defvar``, ``\lagrangian``
- ``LBRACE``, ``RBRACE``: ``{``, ``}``
- ``NUMBER``: ``1.0``, ``9.81``, ``-3.14``
- ``IDENTIFIER``: ``theta``, ``m``, ``g``
- ``OPERATOR``: ``+``, ``-``, ``*``, ``/``, ``^``
- ``FUNCTION``: ``\sin``, ``\cos``, ``\sqrt``
- ``DOT``: ``\dot{x}`` (time derivative)
- ``COMMENT``: ``% ...``

Example tokenization:

.. code-block:: text

   Input: \lagrangian{\frac{1}{2} m \dot{x}^2}
   
   Tokens:
   COMMAND('lagrangian')
   LBRACE
   FUNCTION('frac')
   LBRACE
   NUMBER(1)
   RBRACE
   LBRACE
   NUMBER(2)
   RBRACE
   IDENTIFIER('m')
   DOT
   LBRACE
   IDENTIFIER('x')
   RBRACE
   OPERATOR('^')
   NUMBER(2)
   RBRACE

Parser
------

The parser (``core/parser.py``) builds an AST using recursive descent:

**AST Node Types**:

.. code-block:: python

   @dataclass
   class SystemNode:
       name: str
   
   @dataclass  
   class DefvarNode:
       name: str
       type: str
       unit: str
   
   @dataclass
   class ParameterNode:
       name: str
       value: float
       unit: str
   
   @dataclass
   class LagrangianNode:
       expression: ExprNode
   
   @dataclass
   class ExprNode:
       op: str  # 'add', 'mul', 'pow', 'func', 'var', 'num'
       args: List[ExprNode]

Semantic Analysis
-----------------

The semantic analyzer (``core/semantic.py``) performs:

1. **Symbol Table Construction**:
   
   - Variables with their types
   - Parameters with values
   - Defined operators

2. **Type Checking**:
   
   - Verify all variables are defined
   - Check unit consistency (warning only)

3. **Transformation**:
   
   - Resolve coordinate transforms
   - Expand custom operators

Symbolic Engine
---------------

The symbolic engine (``core/symbolic.py``) uses SymPy:

**Euler-Lagrange Derivation**:

.. code-block:: python

   def derive_euler_lagrange(L, q, q_dot):
       """
       Compute: d/dt(∂L/∂q̇) - ∂L/∂q = 0
       Solve for q̈
       """
       dL_dqdot = sp.diff(L, q_dot)  # ∂L/∂q̇
       dL_dq = sp.diff(L, q)          # ∂L/∂q
       
       # Time derivative using chain rule
       d_dt_dL_dqdot = sum(
           sp.diff(dL_dqdot, var) * var_dot
           for var, var_dot in zip(coords, velocities)
       ) + sp.diff(dL_dqdot, q_ddot) * ???
       
       # Solve for acceleration
       equation = d_dt_dL_dqdot - dL_dq
       q_ddot_solution = sp.solve(equation, q_ddot)[0]
       
       return q_ddot_solution

**Solving Strategy** ("Search & Destroy"):

For coupled systems, we use iterative substitution:

1. Attempt direct solve for each ``q_ddot``
2. If coupled, build matrix equation
3. Solve linear system for accelerations

Code Generation
---------------

Code generators (``codegen/``) translate SymPy to target languages:

**Common Interface**:

.. code-block:: python

   class CodeGenerator(ABC):
       @abstractmethod
       def generate_derivatives(self, equations) -> str:
           """Generate the ODE right-hand side function."""
           pass
       
       @abstractmethod  
       def generate_integrator(self) -> str:
           """Generate time stepping code."""
           pass

**SymPy Code Printers**:

- ``sympy.printing.ccode`` for C++
- ``sympy.printing.NumPyPrinter`` for Python
- Custom printer for WASM

Solver Integration
------------------

The solver (``core/solver.py``) wraps SciPy:

.. code-block:: python

   class Simulator:
       def simulate(self, t_span, y0, method='RK45', **kwargs):
           def derivatives(t, y):
               return self.compiled_eqns(t, y, self.params)
           
           solution = solve_ivp(
               derivatives, t_span, y0,
               method=method,
               dense_output=True,
               **kwargs
           )
           
           return self.format_solution(solution)

Error Handling
--------------

Errors are categorized:

- **Lexer errors**: Invalid characters, unclosed strings
- **Parser errors**: Syntax errors, unbalanced braces
- **Semantic errors**: Undefined variables, type mismatches
- **Symbolic errors**: Cannot solve for accelerations
- **Runtime errors**: Numerical instability, NaN values

Each error type provides:

- Line/column location
- Context (surrounding code)
- Suggested fix (when possible)

Performance Considerations
--------------------------

**Compilation Caching**:

Parsed AST and derived equations are cached using LRU cache.

**Lazy Evaluation**:

Equations are only derived when simulation starts.

**NumPy Vectorization**:

Generated Python uses NumPy operations for speed.

**SymPy Optimization**:

- Common subexpression elimination
- Constant folding
- Trigonometric simplification
