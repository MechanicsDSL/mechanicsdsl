Stability Analysis
==================

The stability module provides tools for analyzing equilibrium points and system stability in mechanical systems.

Overview
--------

Stability analysis determines whether small perturbations from equilibrium grow or decay over time. The module implements:

- **Equilibrium Finding**: Locate fixed points where :math:`\partial V/\partial q = 0`
- **Linearization**: Taylor expand about equilibrium to get linear dynamics
- **Eigenvalue Analysis**: Determine stability from eigenvalues of linearized system
- **Stability Classification**: Classify as stable, unstable, saddle, or center

Theory
------

Equilibrium Points
~~~~~~~~~~~~~~~~~~

For a potential :math:`V(q)`, equilibrium points satisfy:

.. math::

   \frac{\partial V}{\partial q_i} = 0 \quad \forall i

Linearization
~~~~~~~~~~~~~

Near an equilibrium :math:`q_0`, the equations of motion become:

.. math::

   M \ddot{\delta q} + K \delta q = 0

where :math:`M` is the mass matrix and :math:`K_{ij} = \partial^2 V/\partial q_i \partial q_j` is the stiffness matrix (Hessian).

Stability Criteria
~~~~~~~~~~~~~~~~~~

For the linearized system, stability depends on eigenvalues :math:`\lambda` of :math:`M^{-1}K`:

- **Stable**: All eigenvalues positive (oscillatory motion)
- **Unstable**: Any eigenvalue negative (exponential growth)
- **Saddle**: Mixed positive/negative eigenvalues
- **Center**: All eigenvalues zero (marginal stability)

Usage Examples
--------------

Finding Equilibria
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import StabilityAnalyzer, find_equilibria
   import sympy as sp

   # Define potential
   x = sp.Symbol('x', real=True)
   k = sp.Symbol('k', positive=True)
   
   # Harmonic oscillator potential
   V = sp.Rational(1, 2) * k * x**2
   
   # Find equilibrium
   equilibria = find_equilibria(V, ['x'])
   # Returns: [EquilibriumPoint(coordinates={'x': 0})]

Stability Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import StabilityAnalyzer
   import sympy as sp

   analyzer = StabilityAnalyzer()
   
   # Pendulum potential: V = mgl(1 - cos(θ))
   theta = analyzer.get_symbol('theta')
   m, g, l = sp.symbols('m g l', positive=True)
   
   V = m * g * l * (1 - sp.cos(theta))
   
   # Analyze stability at θ = 0 (hanging down)
   result = analyzer.analyze_equilibrium(V, {'theta': 0}, {'m': 1, 'g': 10, 'l': 1})
   
   print(result.stability_type)  # StabilityType.STABLE
   print(result.eigenvalues)     # [10.0]  (positive = stable)
   
   # Analyze stability at θ = π (inverted)
   result_inv = analyzer.analyze_equilibrium(V, {'theta': sp.pi}, {'m': 1, 'g': 10, 'l': 1})
   
   print(result_inv.stability_type)  # StabilityType.UNSTABLE
   print(result_inv.eigenvalues)     # [-10.0]  (negative = unstable)

Double Well Potential
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import StabilityAnalyzer
   import sympy as sp

   analyzer = StabilityAnalyzer()
   
   x = analyzer.get_symbol('x')
   
   # Double well: V = x^4 - 2*x^2
   V = x**4 - 2*x**2
   
   # Find all equilibria
   equilibria = analyzer.find_equilibria(V, ['x'])
   # Returns: x = -1, 0, +1
   
   for eq in equilibria:
       result = analyzer.analyze_equilibrium(V, eq.coordinates, {})
       print(f"x = {eq.coordinates['x']}: {result.stability_type}")
   
   # Output:
   # x = -1: STABLE (local minimum)
   # x = 0: UNSTABLE (local maximum)
   # x = +1: STABLE (local minimum)

Computing Normal Frequencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import StabilityAnalyzer
   import sympy as sp
   import numpy as np

   analyzer = StabilityAnalyzer()
   
   # For stable equilibrium, eigenvalues give ω²
   # Natural frequency: ω = √(eigenvalue)
   
   x = analyzer.get_symbol('x')
   m, k = sp.symbols('m k', positive=True)
   
   T = sp.Rational(1, 2) * m * analyzer.get_symbol('x_dot')**2
   V = sp.Rational(1, 2) * k * x**2
   
   result = analyzer.analyze_stability(T, V, ['x'], {'m': 1.0, 'k': 4.0})
   
   omega = np.sqrt(result.eigenvalues[0])
   print(f"Natural frequency: ω = {omega}")  # ω = 2.0

API Reference
-------------

Classes
~~~~~~~

.. py:class:: StabilityAnalyzer

   Analyzer for equilibrium stability.
   
   .. py:method:: find_equilibria(potential, coordinates)
   
      Find equilibrium points where ∂V/∂q = 0.
   
   .. py:method:: linearize(lagrangian, equilibrium, coordinates)
   
      Linearize system about equilibrium point.
   
   .. py:method:: analyze_equilibrium(potential, equilibrium, parameters)
   
      Analyze stability at given equilibrium.
   
   .. py:method:: analyze_stability(kinetic, potential, coordinates, parameters)
   
      Full stability analysis with mass and stiffness matrices.

.. py:class:: StabilityResult

   Result of stability analysis.
   
   .. py:attribute:: stability_type
   
      StabilityType enum (STABLE, UNSTABLE, SADDLE, CENTER)
   
   .. py:attribute:: eigenvalues
   
      List of eigenvalues of linearized system.
   
   .. py:attribute:: eigenvectors
   
      Corresponding eigenvectors (mode shapes).

Enums
~~~~~

.. py:class:: StabilityType

   - ``STABLE``: All eigenvalues positive
   - ``UNSTABLE``: At least one negative eigenvalue
   - ``SADDLE``: Mixed positive/negative
   - ``CENTER``: All eigenvalues zero

See Also
--------

- :doc:`oscillations` - Normal mode analysis
- :doc:`dissipation` - Damped stability analysis
