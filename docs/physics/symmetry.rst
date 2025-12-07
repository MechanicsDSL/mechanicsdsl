Noether's Theorem & Symmetries
==============================

The symmetry module implements Noether's theorem for automatic detection of conservation laws from symmetries.

Overview
--------

Noether's theorem establishes a profound connection between symmetries and conservation laws:

   *For every continuous symmetry of a physical system, there exists a corresponding conserved quantity.*

The module provides:

- **Cyclic Coordinate Detection**: Find coordinates absent from the Lagrangian
- **Conjugate Momentum Calculation**: Compute conserved momenta
- **Symmetry Detection**: Identify translation, rotation, and time symmetries
- **Conservation Law Derivation**: Automatically derive conserved quantities

Theory
------

Cyclic Coordinates
~~~~~~~~~~~~~~~~~~

A coordinate :math:`q_i` is **cyclic** (or ignorable) if it does not appear in the Lagrangian:

.. math::

   \frac{\partial L}{\partial q_i} = 0

For a cyclic coordinate, the conjugate momentum is conserved:

.. math::

   p_i = \frac{\partial L}{\partial \dot{q}_i} = \text{constant}

Common Symmetry-Conservation Pairs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Symmetry
     - Conserved Quantity
     - Cyclic Coordinate
   * - Time translation
     - Energy
     - (explicit time)
   * - Space translation
     - Linear momentum
     - Position
   * - Rotation about axis
     - Angular momentum
     - Angle

Usage Examples
--------------

Detecting Cyclic Coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import NoetherAnalyzer, detect_cyclic_coordinates
   import sympy as sp

   # Free particle in 2D
   m = sp.Symbol('m', positive=True)
   x_dot = sp.Symbol('x_dot', real=True)
   y_dot = sp.Symbol('y_dot', real=True)
   
   # L = (1/2)m(ẋ² + ẏ²)  - no x, y dependence!
   L = sp.Rational(1, 2) * m * (x_dot**2 + y_dot**2)
   
   cyclic = detect_cyclic_coordinates(L, ['x', 'y'])
   # Returns: ['x', 'y'] - both cyclic, so p_x and p_y are conserved

Central Force Problem
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import NoetherAnalyzer
   import sympy as sp

   analyzer = NoetherAnalyzer()
   
   # Central force in polar coordinates
   m = sp.Symbol('m', positive=True)
   k = sp.Symbol('k', positive=True)
   r = analyzer.get_symbol('r')
   phi = analyzer.get_symbol('phi')  # angle
   r_dot = analyzer.get_symbol('r_dot')
   phi_dot = analyzer.get_symbol('phi_dot')
   
   # L = (1/2)m(ṙ² + r²φ̇²) - V(r)
   # Note: φ does not appear explicitly!
   L = sp.Rational(1, 2) * m * (r_dot**2 + r**2 * phi_dot**2) - k/r
   
   # Detect cyclic coordinates
   cyclic = analyzer.detect_cyclic_coordinates(L, ['r', 'phi'])
   print(cyclic)  # ['phi'] - angle is cyclic
   
   # Get conserved quantity (angular momentum)
   p_phi = analyzer.compute_conjugate_momentum(L, 'phi')
   print(p_phi)  # m * r² * φ̇ = L (angular momentum)

Energy Conservation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import NoetherAnalyzer
   import sympy as sp

   analyzer = NoetherAnalyzer()
   
   # For time-independent Lagrangian, energy is conserved
   m = sp.Symbol('m', positive=True)
   k = sp.Symbol('k', positive=True)
   x = analyzer.get_symbol('x')
   x_dot = analyzer.get_symbol('x_dot')
   
   L = sp.Rational(1, 2) * m * x_dot**2 - sp.Rational(1, 2) * k * x**2
   
   # Compute energy (Hamiltonian)
   E = analyzer.compute_energy(L, ['x'])
   print(E)  # (1/2)m*ẋ² + (1/2)k*x² = T + V

Finding All Symmetries
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import NoetherAnalyzer
   import sympy as sp

   analyzer = NoetherAnalyzer()
   
   # Isotropic 2D harmonic oscillator
   m, k = sp.symbols('m k', positive=True)
   x = analyzer.get_symbol('x')
   y = analyzer.get_symbol('y')
   x_dot = analyzer.get_symbol('x_dot')
   y_dot = analyzer.get_symbol('y_dot')
   
   L = sp.Rational(1, 2) * m * (x_dot**2 + y_dot**2) - \
       sp.Rational(1, 2) * k * (x**2 + y**2)
   
   symmetries = analyzer.find_all_symmetries(L, ['x', 'y'])
   
   for sym in symmetries:
       print(f"{sym.symmetry_type}: {sym.conserved_quantity}")
   
   # Output:
   # TIME_TRANSLATION: Energy (T + V)
   # ROTATION: Angular momentum (x*p_y - y*p_x)

API Reference
-------------

Classes
~~~~~~~

.. py:class:: NoetherAnalyzer

   Analyzer for symmetries and conservation laws.
   
   .. py:method:: detect_cyclic_coordinates(lagrangian, coordinates)
   
      Find all cyclic (ignorable) coordinates.
      
      :returns: List of cyclic coordinate names
   
   .. py:method:: compute_conjugate_momentum(lagrangian, coordinate)
   
      Compute :math:`p_i = \partial L / \partial \dot{q}_i`.
      
      :returns: Sympy expression for momentum
   
   .. py:method:: compute_energy(lagrangian, coordinates)
   
      Compute energy :math:`E = \sum p_i \dot{q}_i - L`.
      
      :returns: Energy expression
   
   .. py:method:: find_all_symmetries(lagrangian, coordinates)
   
      Detect all symmetries and corresponding conserved quantities.
      
      :returns: List of SymmetryInfo objects

.. py:class:: ConservedQuantity

   Represents a conserved quantity from Noether's theorem.
   
   .. py:attribute:: name
   
      Human-readable name (e.g., "angular_momentum")
   
   .. py:attribute:: expression
   
      Sympy expression for the conserved quantity
   
   .. py:attribute:: symmetry
   
      The symmetry that gives rise to this conservation law

Enums
~~~~~

.. py:class:: SymmetryType

   - ``TIME_TRANSLATION``: Energy conservation
   - ``SPACE_TRANSLATION``: Linear momentum conservation
   - ``ROTATION``: Angular momentum conservation
   - ``BOOST``: Center of mass motion

See Also
--------

- :doc:`canonical` - Canonical transformations preserve Poisson brackets
- :doc:`central_forces` - Angular momentum in central force problems
