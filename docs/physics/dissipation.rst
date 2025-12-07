Dissipation & Non-Conservative Forces
======================================

The dissipation module provides tools for modeling energy dissipation and non-conservative forces in mechanical systems.

Overview
--------

Real mechanical systems experience energy loss through friction, drag, and other dissipative mechanisms. The dissipation module implements:

- **Rayleigh Dissipation Function**: Quadratic velocity-dependent energy loss
- **Friction Models**: Viscous, Coulomb, and Stribeck friction
- **Generalized Forces**: Arbitrary non-conservative forces
- **Modified Euler-Lagrange Equations**: With dissipation terms

Theory
------

The standard Euler-Lagrange equation:

.. math::

   \frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = 0

becomes, with dissipation:

.. math::

   \frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} + \frac{\partial \mathcal{F}}{\partial \dot{q}_i} = Q_i

where :math:`\mathcal{F}` is the Rayleigh dissipation function and :math:`Q_i` are generalized forces.

Rayleigh Dissipation Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Rayleigh dissipation function represents energy loss quadratic in velocities:

.. math::

   \mathcal{F} = \frac{1}{2} \sum_{i,j} b_{ij} \dot{q}_i \dot{q}_j

The power dissipated is :math:`P = 2\mathcal{F}`.

Usage Examples
--------------

Basic Rayleigh Dissipation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import (
       RayleighDissipation,
       DissipativeLagrangianMechanics
   )
   import sympy as sp

   # Create dissipation function
   dissipation = RayleighDissipation()
   
   # Add damping coefficient
   b = sp.Symbol('b', positive=True)
   x_dot = dissipation.get_symbol('x_dot')
   
   # F = (1/2) * b * x_dot^2
   dissipation.set_dissipation_function(sp.Rational(1, 2) * b * x_dot**2)
   
   # Get dissipative force
   force = dissipation.get_dissipative_force('x')
   # Returns: -b * x_dot

Friction Models
~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import FrictionModel, FrictionType

   # Viscous friction: F = -b*v
   viscous = FrictionModel(FrictionType.VISCOUS, coefficient=0.5)
   
   # Coulomb friction: F = -μ*N*sign(v)
   coulomb = FrictionModel(FrictionType.COULOMB, coefficient=0.3, normal_force=10.0)
   
   # Stribeck friction (stick-slip)
   stribeck = FrictionModel(
       FrictionType.STRIBECK,
       static_coefficient=0.4,
       kinetic_coefficient=0.3,
       stribeck_velocity=0.01
   )
   
   # Evaluate friction force
   force = viscous.evaluate(velocity=2.0)

Damped Harmonic Oscillator
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import DissipativeLagrangianMechanics
   import sympy as sp

   # Create dissipative system
   system = DissipativeLagrangianMechanics()
   
   # Define symbols
   m = sp.Symbol('m', positive=True)
   k = sp.Symbol('k', positive=True)
   b = sp.Symbol('b', positive=True)
   x = system.get_symbol('x')
   x_dot = system.get_symbol('x_dot')
   
   # Lagrangian: L = T - V
   L = sp.Rational(1, 2) * m * x_dot**2 - sp.Rational(1, 2) * k * x**2
   system.set_lagrangian(L)
   
   # Rayleigh dissipation function
   F = sp.Rational(1, 2) * b * x_dot**2
   system.set_dissipation(F)
   
   # Derive equations with damping
   eom = system.derive_equations_of_motion(['x'])
   # Result: m*x_ddot + b*x_dot + k*x = 0

API Reference
-------------

Classes
~~~~~~~

.. py:class:: RayleighDissipation

   Rayleigh dissipation function for quadratic velocity-dependent energy loss.
   
   .. py:method:: set_dissipation_function(F)
   
      Set the dissipation function :math:`\mathcal{F}`.
   
   .. py:method:: get_dissipative_force(coordinate)
   
      Get the dissipative force :math:`-\partial\mathcal{F}/\partial\dot{q}`.

.. py:class:: FrictionModel(friction_type, coefficient, ...)

   Models various friction types.
   
   .. py:method:: evaluate(velocity)
   
      Evaluate friction force at given velocity.

.. py:class:: DissipativeLagrangianMechanics

   Lagrangian mechanics with dissipation.
   
   .. py:method:: set_dissipation(F)
   
      Set the Rayleigh dissipation function.
   
   .. py:method:: derive_equations_of_motion(coordinates)
   
      Derive modified Euler-Lagrange equations with dissipation.

See Also
--------

- :doc:`lagrangian_mechanics` - Standard Lagrangian mechanics
- :doc:`stability` - Stability analysis of damped systems
