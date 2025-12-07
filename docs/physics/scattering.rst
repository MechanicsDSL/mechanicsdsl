Scattering Theory
=================

The scattering module provides tools for analyzing particle deflection in central force fields, including Rutherford scattering and cross-section calculations.

Overview
--------

Scattering describes how particles are deflected when passing through a force field. The module implements:

- **Rutherford Scattering**: Classical Coulomb scattering
- **Impact Parameter**: Relationship to scattering angle
- **Differential Cross-Section**: Angular distribution of scattered particles
- **Total Cross-Section**: Integrated scattering probability
- **Hard Sphere Scattering**: Billiard-ball model

Theory
------

Scattering Geometry
~~~~~~~~~~~~~~~~~~~

A particle approaches with:

- **Impact parameter** :math:`b`: perpendicular distance from force center to asymptotic trajectory
- **Energy** :math:`E`: kinetic energy at infinity

After scattering, it deflects by **scattering angle** :math:`\theta`.

Rutherford Scattering
~~~~~~~~~~~~~~~~~~~~~

For Coulomb potential :math:`V(r) = k/r`, the scattering angle is:

.. math::

   \tan\left(\frac{\theta}{2}\right) = \frac{k}{2Eb}

The **differential cross-section** (area per solid angle) is:

.. math::

   \frac{d\sigma}{d\Omega} = \left(\frac{k}{4E}\right)^2 \frac{1}{\sin^4(\theta/2)}

This diverges at :math:`\theta \to 0` (forward scattering) because Coulomb potential extends to infinity.

Hard Sphere Scattering
~~~~~~~~~~~~~~~~~~~~~~

For a hard sphere of radius :math:`R`:

.. math::

   \theta = \pi - 2\arcsin(b/R) \quad \text{for } b \leq R

Total cross-section is simply :math:`\sigma = \pi R^2`.

Usage Examples
--------------

Coulomb Scattering Angle
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import ScatteringAnalyzer
   import numpy as np

   analyzer = ScatteringAnalyzer()
   
   # Rutherford scattering
   k = 1.0      # Coulomb constant × charges
   E = 1.0      # Energy
   b = 1.0      # Impact parameter
   
   result = analyzer.coulomb_scattering(energy=E, impact_parameter=b, k=k)
   
   print(f"Scattering angle: θ = {np.degrees(result.scattering_angle):.2f}°")
   print(f"Closest approach: r_min = {result.closest_approach:.4f}")

Head-On Collision
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import ScatteringAnalyzer
   import numpy as np

   analyzer = ScatteringAnalyzer()
   
   # Head-on (b = 0): complete backscatter
   result = analyzer.coulomb_scattering(energy=1.0, impact_parameter=0.0, k=1.0)
   
   print(f"Scattering angle: {np.degrees(result.scattering_angle)}°")
   # Output: 180° (full reflection)

Rutherford Cross-Section
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import rutherford_cross_section
   import numpy as np

   k = 1.0
   E = 1.0
   
   # Differential cross-section at various angles
   angles = [30, 60, 90, 120, 150]
   
   print("Rutherford Cross-Section:")
   print("-" * 40)
   for deg in angles:
       theta = np.radians(deg)
       dcs = rutherford_cross_section(k, E, theta)
       print(f"θ = {deg:3d}°: dσ/dΩ = {dcs:.4f}")
   
   # Cross-section increases dramatically at small angles

Hard Sphere Scattering
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import ScatteringAnalyzer
   import numpy as np

   analyzer = ScatteringAnalyzer()
   
   R = 1.0  # Sphere radius
   
   # Direct hit
   result = analyzer.hard_sphere_scattering(radius=R, impact_parameter=0.0)
   print(f"b=0: θ = {np.degrees(result.scattering_angle)}°")  # 180°
   
   # Grazing
   result = analyzer.hard_sphere_scattering(radius=R, impact_parameter=0.99)
   print(f"b=0.99R: θ = {np.degrees(result.scattering_angle):.1f}°")  # ~8.1°
   
   # Miss
   result = analyzer.hard_sphere_scattering(radius=R, impact_parameter=2.0)
   print(f"b=2R: θ = {np.degrees(result.scattering_angle)}°")  # 0° (no scattering)

Computing from Potential
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import ScatteringAnalyzer
   import numpy as np

   analyzer = ScatteringAnalyzer()
   
   # Custom potential: screened Coulomb (Yukawa)
   def V(r):
       k = 1.0
       alpha = 0.5  # Screening length
       return k * np.exp(-r/alpha) / r
   
   # Compute scattering angle numerically
   theta = analyzer.compute_scattering_angle(
       potential=V,
       energy=1.0,
       impact_parameter=0.5,
       m=1.0
   )
   
   print(f"Yukawa scattering: θ = {np.degrees(theta):.2f}°")

Symbolic Scattering Formulas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import SymbolicScattering
   import sympy as sp

   symbolic = SymbolicScattering()
   
   # Derive Rutherford formulas
   result = symbolic.rutherford_formula()
   
   print("Rutherford Scattering Formulas:")
   print(f"  Scattering angle: {result['scattering_angle']}")
   print(f"  Differential cross-section: {result['differential_cross_section']}")
   print(f"  Closest approach: {result['closest_approach']}")
   
   # Get orbit equation
   orbit = symbolic.orbit_equation()
   print(f"\nOrbit equation: {orbit}")

Convenience Functions
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import rutherford_angle, rutherford_cross_section
   import numpy as np

   # Quick calculation of scattering angle
   theta = rutherford_angle(energy=1.0, impact_parameter=0.5, k=1.0)
   print(f"θ = {np.degrees(theta):.2f}°")
   
   # Quick cross-section
   dcs = rutherford_cross_section(k=1.0, energy=1.0, theta=np.pi/2)
   print(f"dσ/dΩ(90°) = {dcs:.4f}")

API Reference
-------------

Classes
~~~~~~~

.. py:class:: ScatteringAnalyzer

   Analyzer for particle scattering.
   
   .. py:method:: coulomb_scattering(energy, impact_parameter, k)
   
      Compute Rutherford scattering result.
      
      :returns: ScatteringResult
   
   .. py:method:: hard_sphere_scattering(radius, impact_parameter)
   
      Compute hard sphere scattering.
   
   .. py:method:: compute_scattering_angle(potential, energy, impact_parameter, m)
   
      Numerically compute scattering angle for arbitrary potential.
   
   .. py:method:: rutherford_differential_cross_section(k, energy, theta)
   
      Compute :math:`d\sigma/d\Omega` for Coulomb scattering.
   
   .. py:method:: total_cross_section(potential, params, max_b)
   
      Integrate cross-section over all angles.

.. py:class:: SymbolicScattering

   Symbolic derivation of scattering formulas.
   
   .. py:method:: rutherford_formula()
   
      Derive all Rutherford scattering formulas.
   
   .. py:method:: orbit_equation()
   
      Derive orbit equation :math:`r(\phi)`.

.. py:class:: ScatteringResult

   Result of scattering calculation.
   
   .. py:attribute:: scattering_angle
   
      Deflection angle θ (radians)
   
   .. py:attribute:: closest_approach
   
      Minimum distance to force center
   
   .. py:attribute:: impact_parameter
   
      Input impact parameter

Functions
~~~~~~~~~

.. py:function:: rutherford_angle(energy, impact_parameter, k)

   Quick computation of Rutherford scattering angle.

.. py:function:: rutherford_cross_section(k, energy, theta)

   Quick computation of differential cross-section.

See Also
--------

- :doc:`central_forces` - Central force orbits
- :doc:`collisions` - Short-range particle collisions
