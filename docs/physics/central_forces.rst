Central Forces & Orbital Mechanics
===================================

The central forces module provides specialized tools for analyzing motion in central force fields, including orbital mechanics and the Kepler problem.

Overview
--------

A central force depends only on the distance from a fixed point:

.. math::

   \mathbf{F} = F(r) \hat{r}

The module implements:

- **Effective Potential**: Reduce 2D motion to 1D radial problem
- **Orbit Classification**: Bounded, unbounded, circular, elliptical
- **Turning Points**: Find perihelion and aphelion
- **Kepler Problem**: Specialized solver for gravitational orbits
- **Precession**: Calculate apsidal precession for non-Keplerian potentials

Theory
------

Effective Potential
~~~~~~~~~~~~~~~~~~~

For a particle with angular momentum :math:`L` in a central potential :math:`V(r)`:

.. math::

   V_{\text{eff}}(r) = V(r) + \frac{L^2}{2mr^2}

The radial motion is equivalent to 1D motion in this effective potential.

Orbit Classification
~~~~~~~~~~~~~~~~~~~~

Given energy :math:`E` and effective potential :math:`V_{\text{eff}}`:

- **Bounded**: :math:`E < V_{\text{eff}}(\infty)`, motion confined between turning points
- **Unbounded**: :math:`E > V_{\text{eff}}(\infty)`, particle escapes to infinity
- **Circular**: :math:`E = V_{\text{eff,min}}`, particle stays at fixed radius

Kepler Orbits
~~~~~~~~~~~~~

For gravitational potential :math:`V = -k/r`:

.. math::

   r(\phi) = \frac{p}{1 + e \cos(\phi - \phi_0)}

where:

- :math:`p = L^2/(mk)` is the semi-latus rectum
- :math:`e = \sqrt{1 + 2EL^2/(mk^2)}` is the eccentricity

Usage Examples
--------------

Effective Potential Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import EffectivePotential
   import sympy as sp

   # Create effective potential analyzer
   eff = EffectivePotential()
   
   # Gravitational potential
   r = sp.Symbol('r', positive=True)
   k = sp.Symbol('k', positive=True)
   m = sp.Symbol('m', positive=True)
   L = sp.Symbol('L', positive=True)
   
   V = -k / r
   
   V_eff = eff.compute(V, L, m, 'r')
   # V_eff = -k/r + L²/(2mr²)
   
   # Find minimum (circular orbit radius)
   r_circular = eff.find_minimum(V_eff, 'r')
   # r_circular = L² / (mk)

Finding Turning Points
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import CentralForceAnalyzer
   import numpy as np

   analyzer = CentralForceAnalyzer()
   
   # Define potential and parameters
   def V(r):
       return -1.0 / r  # Gravitational (k=1)
   
   # Find turning points for given E and L
   E = -0.5  # Negative = bound orbit
   L = 1.0
   m = 1.0
   
   turning_points = analyzer.find_turning_points(V, E, L, m)
   
   print(f"Perihelion: r_min = {turning_points.r_min:.4f}")
   print(f"Aphelion: r_max = {turning_points.r_max:.4f}")

Orbit Classification
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import CentralForceAnalyzer, OrbitType

   analyzer = CentralForceAnalyzer()
   
   def V(r):
       return -1.0 / r
   
   # Bound orbit (E < 0)
   orbit_type = analyzer.classify_orbit(V, E=-0.5, L=1.0, m=1.0)
   print(orbit_type)  # OrbitType.BOUNDED
   
   # Unbound orbit (E > 0)
   orbit_type = analyzer.classify_orbit(V, E=0.5, L=1.0, m=1.0)
   print(orbit_type)  # OrbitType.UNBOUNDED
   
   # Parabolic (E = 0)
   orbit_type = analyzer.classify_orbit(V, E=0.0, L=1.0, m=1.0)
   print(orbit_type)  # OrbitType.PARABOLIC

Kepler Problem Solver
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import KeplerProblem

   # Create Kepler problem
   kepler = KeplerProblem(G=1.0, M=1.0, m=1.0)
   
   # From initial conditions
   r0 = 1.0
   v0 = 1.0
   
   elements = kepler.compute_orbital_elements(r0, v0, phi0=0)
   
   print(f"Semi-major axis: a = {elements.semi_major_axis:.4f}")
   print(f"Eccentricity: e = {elements.eccentricity:.4f}")
   print(f"Period: T = {elements.period:.4f}")
   
   # Compute position at time t
   t = 1.0
   r, phi = kepler.position_at_time(t, elements)

Computing Orbital Period
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import KeplerProblem
   import numpy as np

   kepler = KeplerProblem(G=6.674e-11, M=1.989e30, m=5.972e24)  # Sun-Earth
   
   # Kepler's third law: T² = (4π²/GM) a³
   a = 1.496e11  # 1 AU in meters
   
   T = kepler.compute_period(a)
   print(f"Orbital period: {T / (365.25 * 24 * 3600):.4f} years")
   # Output: ~1.0 year

API Reference
-------------

Classes
~~~~~~~

.. py:class:: EffectivePotential

   Compute and analyze effective potentials.
   
   .. py:method:: compute(V, L, m, r_var)
   
      Compute :math:`V_{\text{eff}} = V + L^2/(2mr^2)`.
   
   .. py:method:: find_minimum(V_eff, r_var)
   
      Find radius of minimum (circular orbit).

.. py:class:: CentralForceAnalyzer

   Analyzer for central force motion.
   
   .. py:method:: find_turning_points(V, E, L, m)
   
      Find radii where :math:`E = V_{\text{eff}}(r)`.
      
      :returns: TurningPoints namedtuple with r_min, r_max
   
   .. py:method:: classify_orbit(V, E, L, m)
   
      Classify orbit based on energy and effective potential.
      
      :returns: OrbitType enum

.. py:class:: KeplerProblem

   Specialized solver for Keplerian (gravitational) orbits.
   
   .. py:method:: compute_orbital_elements(r, v, phi)
   
      Compute orbital elements from position and velocity.
      
      :returns: OrbitalElements namedtuple
   
   .. py:method:: position_at_time(t, elements)
   
      Compute position using Kepler's equation.
   
   .. py:method:: compute_period(a)
   
      Compute orbital period from semi-major axis.

Enums
~~~~~

.. py:class:: OrbitType

   - ``BOUNDED``: Elliptical orbit (E < 0)
   - ``UNBOUNDED``: Hyperbolic orbit (E > 0)
   - ``PARABOLIC``: Escape trajectory (E = 0)
   - ``CIRCULAR``: Circular orbit (E = V_eff_min)
   - ``COLLISION``: Orbit intersects origin

See Also
--------

- :doc:`scattering` - Rutherford scattering in central force fields
- :doc:`symmetry` - Angular momentum conservation
