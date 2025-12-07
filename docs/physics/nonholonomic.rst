Non-Holonomic Constraints
=========================

The nonholonomic module provides tools for systems with velocity-dependent constraints that cannot be integrated to position constraints.

Overview
--------

Non-holonomic constraints involve velocities and cannot be written as :math:`f(q) = 0`. The module implements:

- **Pfaffian Constraints**: Linear in velocities :math:`\sum a_i(q)\dot{q}_i = 0`
- **Rolling Without Slipping**: Wheels, balls, cylinders
- **Knife-Edge Constraint**: Chaplygin sleigh dynamics
- **Appell's Equations**: Alternative formulation using acceleration energy
- **Maggi's Equations**: Projection method eliminating multipliers

Theory
------

Pfaffian Constraints
~~~~~~~~~~~~~~~~~~~~

A non-holonomic constraint has the form:

.. math::

   \sum_i a_i(q) \dot{q}_i + b(q, t) = 0

If this cannot be integrated to :math:`f(q) = 0`, it is truly non-holonomic.

d'Alembert-Lagrange Equations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With constraints :math:`A \dot{q} = 0`, equations of motion become:

.. math::

   \frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = \sum_j \lambda_j a_{ji}

where :math:`\lambda_j` are Lagrange multipliers.

Appell's Equations
~~~~~~~~~~~~~~~~~~

Using the Gibbs-Appell function (acceleration energy):

.. math::

   S = \frac{1}{2} \sum_{i,j} M_{ij} \ddot{q}_i \ddot{q}_j

Appell's equations are:

.. math::

   \frac{\partial S}{\partial \ddot{q}_j} = Q_j

This is often simpler for non-holonomic systems.

Usage Examples
--------------

Rolling Without Slipping
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import NonholonomicSystem, rolling_constraint
   import sympy as sp

   system = NonholonomicSystem()
   
   # Ball rolling on surface
   m = sp.Symbol('m', positive=True)
   R = sp.Symbol('R', positive=True)
   
   x = system.get_symbol('x')
   theta = system.get_symbol('theta')
   x_dot = system.get_symbol('x_dot')
   theta_dot = system.get_symbol('theta_dot')
   
   # Kinetic energy
   I = sp.Rational(2, 5) * m * R**2  # Solid sphere
   T = sp.Rational(1, 2) * m * x_dot**2 + sp.Rational(1, 2) * I * theta_dot**2
   
   system.set_lagrangian(T)
   
   # Rolling constraint: ẋ = R·θ̇
   lam = system.add_rolling_constraint('x', 'theta', R)
   
   # Derive equations
   eom = system.derive_equations_of_motion(['x', 'theta'])

Chaplygin Sleigh (Knife-Edge)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import NonholonomicSystem, knife_edge_constraint
   import sympy as sp

   system = NonholonomicSystem()
   
   # Chaplygin sleigh: blade can only move along its direction
   m = sp.Symbol('m', positive=True)
   I = sp.Symbol('I', positive=True)  # Moment of inertia
   
   x = system.get_symbol('x')
   y = system.get_symbol('y')
   theta = system.get_symbol('theta')  # Orientation
   x_dot = system.get_symbol('x_dot')
   y_dot = system.get_symbol('y_dot')
   theta_dot = system.get_symbol('theta_dot')
   
   # Kinetic energy
   T = sp.Rational(1, 2) * m * (x_dot**2 + y_dot**2) + \
       sp.Rational(1, 2) * I * theta_dot**2
   
   system.set_lagrangian(T)
   
   # Knife-edge constraint: velocity along blade direction
   # ẏ·cos(θ) - ẋ·sin(θ) = 0
   lam = system.add_knife_edge_constraint('x', 'y', 'theta')
   
   eom = system.derive_equations_of_motion(['x', 'y', 'theta'])

Constraint Matrix Form
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import NonholonomicSystem
   import sympy as sp

   system = NonholonomicSystem()
   
   R = sp.Symbol('R', positive=True)
   
   # Add multiple constraints
   system.add_rolling_constraint('x1', 'theta1', R)
   system.add_rolling_constraint('y1', 'phi1', R)
   
   # Get constraint matrix A and vector b
   # Constraints: A·q̇ + b = 0
   A, b = system.get_constraint_matrix(['x1', 'theta1', 'y1', 'phi1'])
   
   print("Constraint matrix A:")
   print(A)

Using Appell's Equations
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import AppellEquations
   import sympy as sp

   appell = AppellEquations()
   
   m = sp.Symbol('m', positive=True)
   x_dot = appell.get_symbol('x_dot')
   y_dot = appell.get_symbol('y_dot')
   
   # Kinetic energy
   T = sp.Rational(1, 2) * m * (x_dot**2 + y_dot**2)
   
   # Compute Gibbs-Appell function (acceleration energy)
   S = appell.compute_acceleration_energy(T, ['x', 'y'])
   
   print(f"Gibbs-Appell function: S = {S}")
   # S = (1/2)*m*(ẍ² + ÿ²)
   
   # Derive Appell's equations
   Q = {'x': sp.Symbol('F_x'), 'y': sp.Symbol('F_y')}
   eom = appell.derive_appell_equations(S, Q, ['x', 'y'])

Convenience Functions
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import rolling_constraint, knife_edge_constraint
   import sympy as sp

   R = sp.Symbol('R', positive=True)
   
   # Create rolling constraint object directly
   roll = rolling_constraint('x', 'theta', R)
   print(f"Constraint: ẋ - R·θ̇ = 0")
   print(f"Coefficients: {roll.coefficients}")
   
   # Create knife-edge constraint object
   knife = knife_edge_constraint('x', 'y', 'theta')
   print(f"Knife-edge: -ẋ·sin(θ) + ẏ·cos(θ) = 0")

API Reference
-------------

Classes
~~~~~~~

.. py:class:: NonholonomicSystem

   System with non-holonomic (velocity) constraints.
   
   .. py:method:: set_lagrangian(L)
   
      Set the system Lagrangian.
   
   .. py:method:: add_constraint(constraint)
   
      Add a NonholonomicConstraint object.
      
      :returns: Lagrange multiplier symbol
   
   .. py:method:: add_rolling_constraint(linear, angular, radius)
   
      Add rolling without slipping: v = R·ω.
   
   .. py:method:: add_knife_edge_constraint(x, y, theta)
   
      Add knife-edge constraint: ẏ·cos(θ) - ẋ·sin(θ) = 0.
   
   .. py:method:: derive_equations_of_motion(coordinates)
   
      Derive d'Alembert-Lagrange equations with constraints.
   
   .. py:method:: get_constraint_matrix(coordinates)
   
      Get constraint matrix A and vector b.

.. py:class:: NonholonomicConstraint

   Represents a single non-holonomic constraint.
   
   :param coefficients: Dict mapping coordinate to coefficient a_i(q)
   :param inhomogeneous: The b(q,t) term (default 0)
   :param name: Optional constraint name
   
   .. py:method:: as_matrix_form(coordinates)
   
      Convert to matrix form A·q̇ + b = 0.

.. py:class:: AppellEquations

   Appell's equations using acceleration energy.
   
   .. py:method:: compute_acceleration_energy(T, coordinates)
   
      Compute Gibbs-Appell function S from kinetic energy.
   
   .. py:method:: derive_appell_equations(S, forces, coordinates)
   
      Derive ∂S/∂q̈ = Q equations.

.. py:class:: MaggiEquations

   Maggi's projection method.
   
   .. py:method:: project_equations(euler_lagrange, constraint_matrix, coordinates)
   
      Project EL equations onto constraint manifold.

Functions
~~~~~~~~~

.. py:function:: rolling_constraint(linear, angular, radius)

   Create rolling without slipping constraint.

.. py:function:: knife_edge_constraint(x, y, theta)

   Create knife-edge (Chaplygin sleigh) constraint.

See Also
--------

- :doc:`constraint_physics` - Holonomic constraints with Baumgarte stabilization
- :doc:`rigid_body` - Rolling rigid bodies
