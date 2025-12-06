Analysis API Reference
======================

The ``mechanics_dsl.analysis`` package provides tools for energy analysis,
stability analysis, and validation of simulation results.

.. contents:: Contents
   :local:
   :depth: 2

EnergyAnalyzer
--------------

.. py:class:: mechanics_dsl.analysis.EnergyAnalyzer

   Energy analysis for mechanical systems. Computes kinetic energy, potential
   energy, and validates conservation.

   **Example:**

   .. code-block:: python

      from mechanics_dsl.analysis import EnergyAnalyzer
      import numpy as np
      
      analyzer = EnergyAnalyzer()
      
      # Compute energies for pendulum
      energies = analyzer.compute_pendulum_energy(
          solution, m=1.0, l=1.0, g=9.81
      )
      
      T = energies['kinetic']
      V = energies['potential']
      E = energies['total']
      
      # Check conservation
      result = analyzer.check_conservation(solution, T, V, tolerance=1e-3)
      
      if result['conserved']:
          print(f"Energy conserved! Max error: {result['max_relative_error']:.2e}")
      else:
          print(f"Energy drift detected: {result['max_relative_error']:.2%}")

   **Methods:**

   .. py:method:: compute_kinetic_energy(solution, masses, velocities=None)

      Compute kinetic energy T = ½mv² for each timestep.

      :param solution: Simulation result dictionary
      :param masses: Dict mapping coordinate names to masses
      :returns: Kinetic energy array

   .. py:method:: compute_potential_energy(solution, potential_func)

      Compute potential energy at each timestep.

      :param solution: Simulation result
      :param potential_func: Callable V(state) → float
      :returns: Potential energy array

   .. py:method:: check_conservation(solution, kinetic, potential, tolerance=1e-3)

      Check energy conservation.

      :returns: Dict with conservation analysis results:

      .. code-block:: python

         {
             'conserved': bool,
             'initial_energy': float,
             'max_relative_error': float,
             'mean_relative_error': float,
             'total_energy': np.ndarray,
             'kinetic_energy': np.ndarray,
             'potential_energy': np.ndarray,
         }

   .. py:method:: compute_pendulum_energy(solution, m, l, g)

      Specialized method for simple pendulum energy.

      :param m: Mass
      :param l: Length
      :param g: Gravitational acceleration
      :returns: Dict with kinetic, potential, and total energy arrays


StabilityAnalyzer
-----------------

.. py:class:: mechanics_dsl.analysis.StabilityAnalyzer

   Stability analysis for dynamical systems. Provides linearization,
   eigenvalue analysis, and Lyapunov exponent estimation.

   **Example:**

   .. code-block:: python

      from mechanics_dsl.analysis import StabilityAnalyzer
      import sympy as sp
      
      analyzer = StabilityAnalyzer()
      
      # Define system equations: dx/dt = f(x)
      x, y = sp.symbols('x y')
      equations = {
          'x_dot': y,
          'y_dot': -sp.sin(x) - 0.1*y  # Damped pendulum
      }
      
      # Find fixed points
      fixed_points = analyzer.find_fixed_points(equations, [x, y])
      print(f"Fixed points: {fixed_points}")
      
      # Compute Jacobian
      jacobian = analyzer.compute_jacobian(equations, [x, y])
      
      # Analyze stability at each fixed point
      for fp in fixed_points:
          result = analyzer.analyze_stability(jacobian, fp)
          print(f"At {fp}: {result['stability']}")
          print(f"  Eigenvalues: {result['eigenvalues']}")

   **Methods:**

   .. py:method:: find_fixed_points(equations, variables)

      Find fixed points where all derivatives are zero.

      :param equations: Dict mapping derivative names to expressions
      :param variables: List of state variable symbols
      :returns: List of fixed point dictionaries

   .. py:method:: compute_jacobian(equations, variables)

      Compute the Jacobian matrix of the system.

      :param equations: Dict mapping outputs to expressions
      :param variables: List of input variable symbols
      :returns: SymPy Matrix (Jacobian)

   .. py:method:: analyze_stability(jacobian, fixed_point)

      Analyze stability at a fixed point via eigenvalues.

      :returns: Dict with stability analysis:

      .. code-block:: python

         {
             'eigenvalues': list,    # Complex eigenvalues
             'max_real_part': float, # Largest real part
             'stability': str,       # 'stable', 'unstable', or 'marginally_stable'
             'jacobian': Matrix,     # Evaluated Jacobian
         }

   .. py:method:: estimate_lyapunov_exponent(trajectory, dt, n_renorm=100)

      Estimate largest Lyapunov exponent from trajectory data.

      :param trajectory: State array (n_vars × n_points)
      :param dt: Time step
      :param n_renorm: Number of renormalization steps
      :returns: Estimated Lyapunov exponent (float)

      .. note::

         Positive Lyapunov exponent indicates chaotic behavior.


Stability Types
---------------

The stability analyzer classifies fixed points as:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Type
     - Condition
   * - ``stable``
     - All eigenvalues have negative real parts
   * - ``unstable``
     - At least one eigenvalue has positive real part
   * - ``marginally_stable``
     - All eigenvalues have non-positive real parts, at least one is zero
   * - ``center``
     - All eigenvalues are purely imaginary (oscillatory)

For 2D systems, fixed points can be further classified:

- **Stable node**: Both eigenvalues real and negative
- **Unstable node**: Both eigenvalues real and positive
- **Saddle**: One positive, one negative real eigenvalue
- **Spiral (stable)**: Complex eigenvalues with negative real part
- **Spiral (unstable)**: Complex eigenvalues with positive real part
- **Center**: Pure imaginary eigenvalues
