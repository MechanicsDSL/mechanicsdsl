Electromagnetic Physics
======================

The electromagnetic domain provides tools for simulating charged particle dynamics
in electric and magnetic fields.

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

The electromagnetic module implements:

- **Lorentz force dynamics** for charged particles
- **Cyclotron motion** analysis and exact solutions
- **Magnetic traps** (mirror machines, magnetic bottles)
- **E×B drifts** in crossed fields

All implementations follow the Lagrangian formulation:

.. math::

   L = \frac{1}{2}m\mathbf{v}^2 - q\phi + q\mathbf{v} \cdot \mathbf{A}

where :math:`\phi` is the scalar potential and :math:`\mathbf{A}` is the vector potential.

Quick Start
-----------

.. code-block:: python

   from mechanics_dsl.domains.electromagnetic import ChargedParticle, CyclotronMotion
   import numpy as np

   # Create a charged particle (mass=1, charge=1 in natural units)
   particle = ChargedParticle(mass=1.0, charge=1.0)

   # Set up a uniform magnetic field along z
   particle.set_uniform_magnetic_field(Bz=1.0)

   # Derive equations of motion
   eom = particle.derive_equations_of_motion()
   print(eom)  # {'x_ddot': vy, 'y_ddot': -vx, 'z_ddot': 0}

   # Analyze cyclotron motion
   cyclotron = CyclotronMotion(particle)
   omega_c = particle.cyclotron_frequency(B_magnitude=1.0)  # ωc = qB/m
   r_L = particle.larmor_radius(v_perp=0.5, B_magnitude=1.0)  # rL = mv⊥/(qB)

Classes
-------

ChargedParticle
^^^^^^^^^^^^^^^

.. autoclass:: mechanics_dsl.domains.electromagnetic.ChargedParticle
   :members:
   :show-inheritance:

**Example: Particle in crossed E×B fields**

.. code-block:: python

   from mechanics_dsl.domains.electromagnetic import ChargedParticle

   particle = ChargedParticle(mass=9.11e-31, charge=1.6e-19)  # Electron
   
   # E along y, B along z → drift along x
   particle.set_uniform_electric_field(Ey=1000.0)  # 1000 V/m
   particle.set_uniform_magnetic_field(Bz=0.1)     # 0.1 T
   
   # E×B drift velocity
   v_drift = 1000.0 / 0.1  # = 10,000 m/s in x-direction

CyclotronMotion
^^^^^^^^^^^^^^^

.. autoclass:: mechanics_dsl.domains.electromagnetic.CyclotronMotion
   :members:
   :show-inheritance:

**Example: Computing exact cyclotron trajectory**

.. code-block:: python

   import numpy as np
   from mechanics_dsl.domains.electromagnetic import ChargedParticle, CyclotronMotion

   particle = ChargedParticle(mass=1.0, charge=1.0)
   cyclotron = CyclotronMotion(particle)

   t = np.linspace(0, 10, 1000)
   trajectory = cyclotron.exact_trajectory(
       v0=(1.0, 0.0, 0.5),  # Initial velocity
       r0=(0.0, 0.0, 0.0),  # Start at origin
       B=1.0,                # Magnetic field magnitude
       t_array=t
   )

   # trajectory contains 'x', 'y', 'z', 'vx', 'vy', 'vz', 't'

DipoleTrap
^^^^^^^^^^

.. autoclass:: mechanics_dsl.domains.electromagnetic.DipoleTrap
   :members:
   :show-inheritance:

**Example: Analyzing magnetic mirror confinement**

.. code-block:: python

   from mechanics_dsl.domains.electromagnetic import DipoleTrap
   import numpy as np

   # Magnetic mirror with B0=1 T, scale length L=1 m
   trap = DipoleTrap(B0=1.0, L=1.0)

   # Field at z=1: B(1) = B0(1 + 1) = 2 T
   print(trap.magnetic_field(z=1.0))  # 2.0

   # Mirror ratio
   R = trap.mirror_ratio(z_mirror=1.0)  # R = 2

   # Loss cone angle
   theta_loss = trap.loss_cone_angle(z_mirror=1.0)
   print(f"Loss cone: {np.degrees(theta_loss):.1f}°")  # 45°

   # Is a particle with 60° pitch angle trapped?
   print(trap.is_trapped(np.radians(60), z_mirror=1.0))  # True

Physical Constants
------------------

The module uses SI units by default. Key relationships:

- **Cyclotron frequency**: :math:`\omega_c = \frac{qB}{m}`
- **Larmor radius**: :math:`r_L = \frac{mv_\perp}{qB}`
- **E×B drift velocity**: :math:`v_E = \frac{E}{B}`
- **Magnetic moment**: :math:`\mu = \frac{mv_\perp^2}{2B}` (adiabatic invariant)

Convenience Functions
---------------------

.. code-block:: python

   from mechanics_dsl.domains.electromagnetic import (
       uniform_crossed_fields,
       calculate_drift_velocity
   )

   # Create particle in crossed fields quickly
   particle = uniform_crossed_fields(E=100.0, B=1.0)

   # Calculate E×B drift
   v_drift = calculate_drift_velocity(E=100.0, B=1.0)  # 100 m/s

See Also
--------

- :doc:`relativistic` - For relativistic charged particles
- :doc:`../physics/central_forces` - For orbital dynamics
