General Relativity
==================

The general relativity domain provides tools for calculations in curved spacetime,
including black holes, gravitational lensing, and cosmology.

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

The general relativity module implements:

- **Schwarzschild metric** for non-rotating black holes
- **Kerr metric** for rotating black holes
- **Geodesic solver** for particle/photon trajectories
- **Gravitational lensing** calculations
- **FLRW cosmology** for the expanding universe

Quick Start
-----------

.. code-block:: python

   from mechanics_dsl.domains.general_relativity import (
       SchwarzschildMetric, GravitationalLensing, FLRWCosmology,
       SOLAR_MASS
   )
   
   # Black hole analysis
   bh = SchwarzschildMetric(mass=10 * SOLAR_MASS)
   rs = bh.schwarzschild_radius()  # ~30 km
   r_isco = bh.isco_radius()       # 3 × rs
   T_hawking = bh.hawking_temperature()
   
   # Gravitational lensing
   lens = GravitationalLensing(mass=SOLAR_MASS)
   alpha = lens.deflection_angle(impact_parameter=7e8)  # 1.75 arcsec
   
   # Cosmology
   cosmos = FLRWCosmology(H0=70, Omega_m=0.3, Omega_Lambda=0.7)
   age = cosmos.age()

Classes
-------

SchwarzschildMetric
^^^^^^^^^^^^^^^^^^^

The Schwarzschild metric describes non-rotating, spherically symmetric black holes:

.. math::

   ds^2 = -\left(1 - \frac{r_s}{r}\right)c^2 dt^2 + \left(1 - \frac{r_s}{r}\right)^{-1} dr^2 + r^2 d\Omega^2

where :math:`r_s = 2GM/c^2` is the Schwarzschild radius.

**Key methods:**

- ``schwarzschild_radius()``: Event horizon radius
- ``isco_radius()``: Innermost stable circular orbit (3rs)
- ``photon_sphere_radius()``: Light orbit radius (1.5rs)
- ``hawking_temperature()``: Black hole temperature
- ``gravitational_redshift(r)``: Photon redshift from radius r

KerrMetric
^^^^^^^^^^

The Kerr metric describes rotating black holes with spin parameter :math:`a = J/(Mc)`.

**Key features:**

- ``outer_horizon()``: Outer event horizon
- ``inner_horizon()``: Inner (Cauchy) horizon
- ``ergosphere_radius(theta)``: Static limit surface
- ``frame_dragging_rate(r)``: Spacetime rotation rate
- ``isco_radius(prograde)``: ISCO for prograde/retrograde orbits

GravitationalLensing
^^^^^^^^^^^^^^^^^^^^

Light deflection by massive objects.

**Deflection angle** (weak field):

.. math::

   \alpha = \frac{4GM}{c^2 b} = \frac{2r_s}{b}

For the Sun at grazing incidence: α ≈ 1.75 arcseconds.

FLRWCosmology
^^^^^^^^^^^^^

Friedmann-Lemaître-Robertson-Walker cosmological model:

.. math::

   ds^2 = -c^2 dt^2 + a(t)^2 \left[\frac{dr^2}{1-kr^2} + r^2 d\Omega^2\right]

**Key methods:**

- ``hubble_parameter(z)``: H(z) at redshift z
- ``age()``: Age of the universe
- ``comoving_distance(z)``: Distance to redshift z
- ``luminosity_distance(z)``: For standard candles

Physical Constants
------------------

.. code-block:: python

   from mechanics_dsl.domains.general_relativity import (
       SPEED_OF_LIGHT,      # 299792458 m/s
       GRAVITATIONAL_CONSTANT,  # 6.674e-11 m³/(kg·s²)
       SOLAR_MASS           # 1.989e30 kg
   )

See Also
--------

- :doc:`relativistic` - Special relativity
- :doc:`quantum` - Quantum mechanics

