Collision Dynamics
==================

The collisions module provides tools for analyzing elastic, inelastic, and perfectly inelastic collisions between particles.

Overview
--------

The module implements:

- **1D and 3D Collisions**: Full momentum and energy conservation
- **Coefficient of Restitution**: Model energy loss (0 = perfectly inelastic, 1 = elastic)
- **Center of Mass Frame**: Transform to simplified reference frame
- **Impulse Calculations**: Force × time during impact
- **Symbolic Solver**: Derive collision formulas analytically

Theory
------

Conservation Laws
~~~~~~~~~~~~~~~~~

For a two-body collision, **momentum is always conserved**:

.. math::

   m_1 \mathbf{v}_1 + m_2 \mathbf{v}_2 = m_1 \mathbf{v}_1' + m_2 \mathbf{v}_2'

Coefficient of Restitution
~~~~~~~~~~~~~~~~~~~~~~~~~~

The coefficient of restitution :math:`e` relates relative velocities before and after collision:

.. math::

   e = -\frac{(\mathbf{v}_1' - \mathbf{v}_2') \cdot \hat{n}}{(\mathbf{v}_1 - \mathbf{v}_2) \cdot \hat{n}}

where :math:`\hat{n}` is the collision normal.

- :math:`e = 1`: Elastic (kinetic energy conserved)
- :math:`0 < e < 1`: Inelastic (energy lost)
- :math:`e = 0`: Perfectly inelastic (bodies stick together)

Energy Loss
~~~~~~~~~~~

For inelastic collisions, energy lost is:

.. math::

   \Delta KE = \frac{1}{2} \mu (1 - e^2) v_{rel}^2

where :math:`\mu = m_1 m_2 / (m_1 + m_2)` is the reduced mass.

Usage Examples
--------------

1D Elastic Collision
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import elastic_collision_1d

   # Equal masses: velocities exchange
   m1, m2 = 1.0, 1.0
   v1, v2 = 2.0, 0.0
   
   v1_final, v2_final = elastic_collision_1d(m1, v1, m2, v2)
   
   print(f"v1' = {v1_final}")  # 0.0
   print(f"v2' = {v2_final}")  # 2.0

1D Inelastic Collision
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import inelastic_collision_1d

   m1, m2 = 1.0, 2.0
   v1, v2 = 3.0, 0.0
   e = 0.5  # 50% energy retained
   
   v1_final, v2_final = inelastic_collision_1d(m1, v1, m2, v2, e=e)
   
   # Verify momentum conservation
   p_before = m1*v1 + m2*v2
   p_after = m1*v1_final + m2*v2_final
   print(f"Momentum conserved: {abs(p_before - p_after) < 1e-10}")  # True

Perfectly Inelastic Collision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import perfectly_inelastic_1d

   m1, m2 = 1.0, 3.0
   v1, v2 = 4.0, 0.0
   
   # Bodies stick together
   v_final = perfectly_inelastic_1d(m1, v1, m2, v2)
   
   print(f"Combined velocity: {v_final}")  # (1*4 + 3*0)/(1+3) = 1.0

3D Collision with CollisionSolver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import CollisionSolver, Particle
   import numpy as np

   solver = CollisionSolver()
   
   # Two particles in 3D
   p1 = Particle(
       mass=1.0,
       position=np.array([0, 0, 0]),
       velocity=np.array([1, 0, 0])
   )
   p2 = Particle(
       mass=2.0,
       position=np.array([1, 0, 0]),
       velocity=np.array([0, 0, 0])
   )
   
   result = solver.solve(p1, p2, e=0.8)
   
   print(f"p1 final velocity: {result.v1_final}")
   print(f"p2 final velocity: {result.v2_final}")
   print(f"Energy lost: {result.energy_loss:.4f}")
   print(f"Collision type: {result.collision_type}")

Center of Mass Frame
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import CollisionSolver, Particle
   import numpy as np

   solver = CollisionSolver()
   
   p1 = Particle(mass=1.0, position=np.zeros(3), velocity=np.array([2, 0, 0]))
   p2 = Particle(mass=1.0, position=np.ones(3), velocity=np.array([0, 0, 0]))
   
   v1_cm, v2_cm = solver.center_of_mass_frame(p1, p2)
   
   # In CM frame, momenta are equal and opposite
   p_total_cm = p1.mass * v1_cm + p2.mass * v2_cm
   print(f"Total CM momentum: {p_total_cm}")  # [0, 0, 0]

Impulse Calculations
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import ImpulseCalculator
   import numpy as np

   # Calculate impulse from velocity change
   mass = 2.0
   delta_v = np.array([3.0, 0.0, 0.0])
   
   impulse = ImpulseCalculator.impulse_momentum(mass, delta_v)
   print(f"Impulse J = {impulse}")  # [6, 0, 0]
   
   # Calculate impulse from average force
   F_avg = np.array([1000.0, 0.0, 0.0])
   dt = 0.01  # 10 ms collision
   
   impulse = ImpulseCalculator.impulse_from_force(F_avg, dt)
   print(f"Impulse J = {impulse}")  # [10, 0, 0]

Symbolic Collision Formulas
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import SymbolicCollisionSolver
   
   solver = SymbolicCollisionSolver()
   
   # Derive elastic collision formulas
   result = solver.solve_1d_elastic()
   
   print("1D Elastic Collision Formulas:")
   print(f"  v1' = {result['v1_final']}")
   print(f"  v2' = {result['v2_final']}")
   
   # v1' = ((m1-m2)*v1 + 2*m2*v2) / (m1+m2)
   # v2' = ((m2-m1)*v2 + 2*m1*v1) / (m1+m2)
   
   # Energy loss formula
   delta_KE = solver.energy_loss()
   print(f"\nEnergy loss: ΔKE = {delta_KE}")
   # ΔKE = (1/2)*μ*(1-e²)*v_rel²

API Reference
-------------

Classes
~~~~~~~

.. py:class:: CollisionSolver

   Numerical solver for 2-body collisions.
   
   .. py:method:: solve(particle1, particle2, e=1.0, normal=None)
   
      Solve collision and return final velocities.
      
      :param e: Coefficient of restitution
      :param normal: Collision normal (auto-computed if None)
      :returns: CollisionResult
   
   .. py:method:: solve_1d(m1, v1, m2, v2, e=1.0)
   
      Simplified 1D solver.
      
      :returns: Tuple (v1_final, v2_final)
   
   .. py:method:: center_of_mass_frame(particle1, particle2)
   
      Transform to CM reference frame.
   
   .. py:method:: reduced_mass(m1, m2)
   
      Compute :math:`\mu = m_1 m_2 / (m_1 + m_2)`.

.. py:class:: Particle

   Particle state for collision.
   
   :param mass: Particle mass
   :param position: Position vector (numpy array)
   :param velocity: Velocity vector (numpy array)
   
   .. py:attribute:: momentum
   
      p = m·v
   
   .. py:attribute:: kinetic_energy
   
      KE = ½mv²

.. py:class:: SymbolicCollisionSolver

   Symbolic derivation of collision formulas.
   
   .. py:method:: solve_1d_elastic()
   
      Derive 1D elastic collision formulas.
   
   .. py:method:: solve_1d_inelastic(e=None)
   
      Derive formulas with restitution coefficient.
   
   .. py:method:: energy_loss()
   
      Derive energy loss expression.

.. py:class:: CollisionResult

   Result of collision calculation.
   
   .. py:attribute:: v1_final, v2_final
   
      Final velocities
   
   .. py:attribute:: impulse
   
      Impulse vector during collision
   
   .. py:attribute:: energy_loss
   
      Kinetic energy dissipated
   
   .. py:attribute:: collision_type
   
      CollisionType enum

Functions
~~~~~~~~~

.. py:function:: elastic_collision_1d(m1, v1, m2, v2)

   Compute 1D elastic collision final velocities.

.. py:function:: inelastic_collision_1d(m1, v1, m2, v2, e)

   Compute 1D inelastic collision with given restitution.

.. py:function:: perfectly_inelastic_1d(m1, v1, m2, v2)

   Compute final velocity when bodies stick together.

See Also
--------

- :doc:`scattering` - Large-angle deflections in central force fields
- :doc:`variable_mass` - Collisions with mass ejection
