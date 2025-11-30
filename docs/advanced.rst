Advanced Topics
===============

Compiler Architecture
---------------------

MechanicsDSL uses a multi-stage compilation pipeline:

1.  **Lexing & Parsing**: The source code is tokenized and parsed into a custom Abstract Syntax Tree (AST) using a recursive descent parser.
2.  **Semantic Analysis**: The AST is validated for consistency (e.g., ensuring all variables used are defined).
3.  **Symbolic Derivation**: The ``SymbolicEngine`` converts AST nodes into SymPy expressions and applies the Euler-Lagrange operator.
4.  **Code Generation**: The resulting symbolic equations are compiled into Python functions using ``sympy.lambdify``, optimized for NumPy array operations.

Adaptive Solver Selection
-------------------------

The numerical engine automatically detects system properties to select the best ODE solver:

* **RK45**: Used for non-stiff, smooth systems (e.g., simple harmonic oscillator). It is fast and accurate for standard problems.
* **LSODA**: Used for stiff systems, constrained dynamics, or long-term chaotic integration. It automatically switches between non-stiff (Adams) and stiff (BDF) methods.

Constraint Handling
-------------------

For systems with holonomic constraints :math:`g(q) = 0`, the engine uses the method of **Lagrange Multipliers**. The Lagrangian is augmented:

.. math::

    L' = L + \sum_j \lambda_j g_j(q)

This introduces additional variables (the multipliers :math:`\lambda_j`) and equations, turning the system into a Differential-Algebraic Equation (DAE) system.
