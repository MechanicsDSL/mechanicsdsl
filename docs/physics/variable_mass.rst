Variable Mass Systems
=====================

The variable mass module provides tools for analyzing systems where mass changes over time, such as rockets, conveyor belts, and falling chains.

Overview
--------

When mass is added to or removed from a system, Newton's second law takes a modified form. The module implements:

- **Tsiolkovsky Rocket Equation**: Ideal delta-v calculation
- **Rocket Simulation**: Numerical integration with thrust and gravity
- **Conveyor Belt Problems**: Force to maintain belt motion
- **Falling Chain**: Impact forces on a surface
- **Multistage Rockets**: Optimal staging analysis

Theory
------

Generalized Newton's Law
~~~~~~~~~~~~~~~~~~~~~~~~

For a system with changing mass:

.. math::

   \mathbf{F}_{ext} = m \frac{d\mathbf{v}}{dt} + \mathbf{v}_{rel} \frac{dm}{dt}

where :math:`\mathbf{v}_{rel}` is the velocity of mass entering or leaving the system.

Tsiolkovsky Rocket Equation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a rocket in free space (no gravity, no drag):

.. math::

   \Delta v = v_e \ln\left(\frac{m_0}{m_f}\right)

where:

- :math:`v_e` = exhaust velocity
- :math:`m_0` = initial mass (rocket + fuel)
- :math:`m_f` = final mass (rocket only)

With gravity:

.. math::

   \Delta v = v_e \ln\left(\frac{m_0}{m_f}\right) - g \cdot t_{burn}

Specific Impulse
~~~~~~~~~~~~~~~~

Specific impulse :math:`I_{sp}` relates exhaust velocity to efficiency:

.. math::

   I_{sp} = \frac{v_e}{g_0} \quad \text{(in seconds)}

Higher :math:`I_{sp}` means more efficient propellant utilization.

Usage Examples
--------------

Tsiolkovsky Ideal Delta-V
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import tsiolkovsky_delta_v
   import numpy as np

   # Rocket parameters
   m_initial = 1000  # kg (rocket + fuel)
   m_final = 200     # kg (rocket only)
   v_exhaust = 3000  # m/s
   
   delta_v = tsiolkovsky_delta_v(m_initial, m_final, v_exhaust)
   
   print(f"Δv = {delta_v:.0f} m/s")
   # Δv = 3000 * ln(5) ≈ 4828 m/s

RocketEquation Class
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import RocketEquation, RocketParameters

   # Define rocket
   params = RocketParameters(
       initial_mass=1000.0,     # kg
       fuel_mass=800.0,         # kg
       exhaust_velocity=3000.0, # m/s
       mass_flow_rate=10.0      # kg/s
   )
   
   rocket = RocketEquation()
   
   # Compute key values
   delta_v = rocket.ideal_delta_v(params)
   burn_time = rocket.burn_time(params)
   isp = rocket.specific_impulse(params)
   
   print(f"Ideal Δv: {delta_v:.0f} m/s")
   print(f"Burn time: {burn_time:.0f} s")
   print(f"Specific impulse: {isp:.0f} s")

Rocket Simulation
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import RocketEquation, RocketParameters
   import matplotlib.pyplot as plt

   params = RocketParameters(
       initial_mass=1000.0,
       fuel_mass=800.0,
       exhaust_velocity=3000.0,
       mass_flow_rate=10.0
   )
   
   rocket = RocketEquation()
   
   # Simulate with gravity
   result = rocket.simulate(params, t_span=(0, 100), g=9.81, num_points=100)
   
   # Plot results
   fig, axes = plt.subplots(3, 1, figsize=(10, 8))
   
   axes[0].plot(result['time'], result['velocity'])
   axes[0].set_ylabel('Velocity (m/s)')
   
   axes[1].plot(result['time'], result['altitude'])
   axes[1].set_ylabel('Altitude (m)')
   
   axes[2].plot(result['time'], result['mass'])
   axes[2].set_xlabel('Time (s)')
   axes[2].set_ylabel('Mass (kg)')
   
   plt.tight_layout()

Delta-V with Gravity Loss
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import RocketEquation, RocketParameters

   params = RocketParameters(
       initial_mass=1000.0,
       fuel_mass=800.0,
       exhaust_velocity=3000.0,
       mass_flow_rate=10.0
   )
   
   rocket = RocketEquation()
   
   dv_ideal = rocket.ideal_delta_v(params)
   dv_gravity = rocket.delta_v_with_gravity(params, g=9.81)
   
   print(f"Ideal Δv: {dv_ideal:.0f} m/s")
   print(f"With gravity: {dv_gravity:.0f} m/s")
   print(f"Gravity loss: {dv_ideal - dv_gravity:.0f} m/s")
   # Gravity loss = g × t_burn = 9.81 × 80 ≈ 785 m/s

Required Mass Ratio
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import required_mass_ratio
   import numpy as np

   # What mass ratio needed for orbital velocity?
   delta_v = 8000  # m/s (approximate LEO velocity)
   v_exhaust = 3500  # m/s (typical kerosene/LOX)
   
   ratio = required_mass_ratio(delta_v, v_exhaust)
   
   print(f"Required mass ratio: {ratio:.2f}")
   # For Δv = 8000, v_e = 3500: ratio ≈ 9.7
   # Only ~10% of launch mass reaches orbit!

Conveyor Belt Force
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import VariableMassSystem

   system = VariableMassSystem()
   
   # Sand falling onto moving belt
   belt_velocity = 2.0  # m/s
   mass_rate = 5.0      # kg/s
   
   force = system.conveyor_belt_force(belt_velocity, mass_rate)
   
   print(f"Force to maintain belt speed: {force:.1f} N")
   # F = v × dm/dt = 2 × 5 = 10 N

Falling Chain
~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import VariableMassSystem
   import numpy as np

   system = VariableMassSystem()
   
   # Chain falling from height onto table
   chain_length = 2.0     # m
   linear_density = 0.5   # kg/m
   height_fallen = 1.0    # m (of chain that has fallen)
   g = 9.81               # m/s²
   
   force = system.falling_chain(
       chain_length=chain_length,
       linear_density=linear_density,
       height=height_fallen,
       g=g
   )
   
   print(f"Force on table: {force:.2f} N")
   # Force = weight on table + impact force

Multistage Rocket (Symbolic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import SymbolicVariableMass

   symbolic = SymbolicVariableMass()
   
   # Two-stage rocket
   total_dv = symbolic.multistage_rocket(stages=2)
   
   print("Total Δv for 2-stage rocket:")
   print(f"  Δv = {total_dv}")
   # Δv = v_e1·ln(m_10/m_1f) + v_e2·ln(m_20/m_2f)

API Reference
-------------

Classes
~~~~~~~

.. py:class:: RocketEquation

   Tsiolkovsky rocket equation solver.
   
   .. py:method:: ideal_delta_v(params)
   
      Compute :math:`\Delta v = v_e \ln(m_0/m_f)`.
   
   .. py:method:: burn_time(params)
   
      Compute :math:`t = m_{fuel} / \dot{m}`.
   
   .. py:method:: delta_v_with_gravity(params, g)
   
      Compute Δv accounting for gravity loss.
   
   .. py:method:: simulate(params, t_span, g, num_points)
   
      Numerically simulate rocket trajectory.
   
   .. py:method:: specific_impulse(params, g0=9.81)
   
      Compute :math:`I_{sp} = v_e / g_0`.

.. py:class:: RocketParameters

   Parameters for rocket simulation.
   
   :param initial_mass: Total initial mass
   :param fuel_mass: Fuel mass
   :param exhaust_velocity: Exhaust velocity
   :param mass_flow_rate: Mass ejection rate
   :param thrust: Optional thrust force
   :param external_force: Optional force function F(t, state)
   
   .. py:attribute:: dry_mass
   
      Mass without fuel
   
   .. py:attribute:: mass_ratio
   
      :math:`m_0 / m_f`

.. py:class:: VariableMassSystem

   General variable mass system solver.
   
   .. py:method:: conveyor_belt_force(belt_velocity, mass_rate)
   
      Force to accelerate material onto belt.
   
   .. py:method:: falling_chain(chain_length, linear_density, height, g)
   
      Force on table from falling chain.
   
   .. py:method:: derive_eom(mass_rate, relative_velocity, external_force)
   
      Derive general EOM for variable mass.

.. py:class:: SymbolicVariableMass

   Symbolic analysis of variable mass systems.
   
   .. py:method:: rocket_equation_symbolic()
   
      Derive rocket formulas symbolically.
   
   .. py:method:: multistage_rocket(stages)
   
      Derive total Δv for n-stage rocket.

Functions
~~~~~~~~~

.. py:function:: tsiolkovsky_delta_v(m_initial, m_final, exhaust_velocity)

   Compute ideal rocket Δv.

.. py:function:: required_mass_ratio(delta_v, exhaust_velocity)

   Compute mass ratio for target Δv.

.. py:function:: specific_impulse_to_exhaust_velocity(isp, g0=9.81)

   Convert Isp to exhaust velocity.

See Also
--------

- :doc:`collisions` - Momentum transfer during impacts
