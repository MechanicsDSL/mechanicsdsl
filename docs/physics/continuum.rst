Continuous Systems & Field Mechanics
=====================================

The continuum module provides tools for analyzing continuous mechanical systems like vibrating strings, membranes, and elastic bodies.

Overview
--------

For continuous systems, the Lagrangian becomes a **Lagrangian density** integrated over space. The module implements:

- **Lagrangian Density**: For strings, membranes, and general fields
- **Field Euler-Lagrange Equations**: Derive wave equations and field equations
- **Vibrating String**: Normal modes, Fourier decomposition
- **Vibrating Membrane**: 2D mode shapes and frequencies
- **Stress-Energy Tensor**: Energy density and momentum flux

Theory
------

Lagrangian Density
~~~~~~~~~~~~~~~~~~

For a field :math:`\phi(x, t)`, the Lagrangian density is:

.. math::

   \mathcal{L} = \mathcal{L}\left(\phi, \frac{\partial\phi}{\partial t}, \frac{\partial\phi}{\partial x}\right)

The action is:

.. math::

   S = \int \int \mathcal{L} \, dx \, dt

Field Euler-Lagrange Equation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The variational principle :math:`\delta S = 0` gives:

.. math::

   \frac{\partial\mathcal{L}}{\partial\phi} - \frac{\partial}{\partial t}\frac{\partial\mathcal{L}}{\partial\phi_t} - \frac{\partial}{\partial x}\frac{\partial\mathcal{L}}{\partial\phi_x} = 0

Wave Equation
~~~~~~~~~~~~~

For a string with tension :math:`T` and linear density :math:`\mu`:

.. math::

   \mathcal{L} = \frac{1}{2}\mu\left(\frac{\partial u}{\partial t}\right)^2 - \frac{1}{2}T\left(\frac{\partial u}{\partial x}\right)^2

This gives the wave equation:

.. math::

   \frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}, \quad c = \sqrt{T/\mu}

Usage Examples
--------------

Vibrating String Setup
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import VibratingString
   import numpy as np

   # Create string: 1 meter long, wave speed 100 m/s
   string = VibratingString(length=1.0, wave_speed=100.0)
   
   # Fundamental frequency
   f1 = string.fundamental_frequency()
   print(f"Fundamental: f₁ = {f1} Hz")  # f = c/(2L) = 50 Hz
   
   # Harmonic series
   for n in range(1, 6):
       f_n = string.mode_frequency(n)
       print(f"Mode {n}: f = {f_n} Hz")

Mode Shapes
~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import VibratingString
   import numpy as np
   import matplotlib.pyplot as plt

   string = VibratingString(length=1.0, wave_speed=100.0)
   
   x = np.linspace(0, 1, 100)
   
   plt.figure(figsize=(10, 6))
   for n in range(1, 5):
       shape = string.mode_shape(n, x)
       plt.plot(x, shape, label=f'Mode {n}')
   
   plt.xlabel('Position (m)')
   plt.ylabel('Amplitude')
   plt.legend()
   plt.title('String Normal Mode Shapes')

Fourier Decomposition
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import VibratingString
   import numpy as np

   string = VibratingString(length=1.0, wave_speed=100.0)
   
   # Initial plucked shape: triangle at center
   x = np.linspace(0, 1, 100)
   initial_shape = np.where(x <= 0.5, 2*x, 2*(1-x))
   
   # Compute Fourier coefficients
   coeffs = string.fourier_coefficients(initial_shape, x, n_modes=10)
   
   print("Fourier coefficients:")
   for n, c in enumerate(coeffs, 1):
       print(f"  A_{n} = {c:.4f}")

Time Evolution of String
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import VibratingString
   import numpy as np
   import matplotlib.pyplot as plt

   string = VibratingString(length=1.0, wave_speed=100.0)
   
   x = np.linspace(0, 1, 100)
   t = np.linspace(0, 0.05, 50)
   
   # Initial displacement
   initial = np.sin(np.pi * x)  # First mode only
   coeffs = string.fourier_coefficients(initial, x, n_modes=5)
   
   # Compute solution u(x, t)
   u = string.solution(x, t, coeffs)
   
   # Plot at different times
   for i, t_val in enumerate([0, 10, 25, 40]):
       plt.plot(x, u[:, i], label=f't = {t[t_val]*1000:.1f} ms')
   plt.legend()

Vibrating Membrane
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import VibratingMembrane
   import numpy as np

   # Rectangular membrane
   membrane = VibratingMembrane(length_x=1.0, length_y=1.5, wave_speed=100.0)
   
   # Mode frequencies
   f_11 = membrane.mode_frequency(1, 1)
   f_21 = membrane.mode_frequency(2, 1)
   f_12 = membrane.mode_frequency(1, 2)
   
   print(f"f(1,1) = {f_11/(2*np.pi):.2f} Hz")
   print(f"f(2,1) = {f_21/(2*np.pi):.2f} Hz")
   print(f"f(1,2) = {f_12/(2*np.pi):.2f} Hz")
   
   # Get sorted modes
   modes = membrane.compute_modes(max_m=3, max_n=3)
   print("\nModes sorted by frequency:")
   for m, n, freq in modes[:5]:
       print(f"  ({m},{n}): ω = {freq:.2f}")

Membrane Mode Shapes
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import VibratingMembrane
   import numpy as np
   import matplotlib.pyplot as plt

   membrane = VibratingMembrane(length_x=1.0, length_y=1.0, wave_speed=100.0)
   
   x = np.linspace(0, 1, 50)
   y = np.linspace(0, 1, 50)
   
   fig, axes = plt.subplots(2, 2, figsize=(10, 10))
   
   modes = [(1,1), (2,1), (1,2), (2,2)]
   for ax, (m, n) in zip(axes.flat, modes):
       shape = membrane.mode_shape(m, n, x, y)
       ax.contourf(x, y, shape.T, levels=20, cmap='RdBu')
       ax.set_title(f'Mode ({m},{n})')
       ax.set_aspect('equal')

Lagrangian Density
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import LagrangianDensity
   import sympy as sp

   density = LagrangianDensity()
   
   # String Lagrangian density
   L_string = density.string_lagrangian()
   print(f"String: L = {L_string}")
   # L = (1/2)*μ*u_t² - (1/2)*T*u_x²
   
   # Membrane Lagrangian density
   L_membrane = density.membrane_lagrangian()
   print(f"Membrane: L = {L_membrane}")
   
   # Klein-Gordon (relativistic scalar field)
   L_kg = density.klein_gordon_lagrangian()
   print(f"Klein-Gordon: L = {L_kg}")

Field Euler-Lagrange
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import FieldEulerLagrange
   import sympy as sp

   field_el = FieldEulerLagrange()
   
   # Derive wave equation from string Lagrangian
   wave_eq = field_el.wave_equation_1d()
   print(f"Wave equation: {wave_eq}")
   # u_tt = c²·u_xx

Convenience Functions
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import string_mode_frequencies, wave_speed

   # Compute wave speed from tension and density
   T = 100.0   # N (tension)
   mu = 0.01   # kg/m (linear density)
   
   c = wave_speed(tension=T, density=mu)
   print(f"Wave speed: c = {c:.1f} m/s")  # 100 m/s
   
   # Get first 5 mode frequencies
   L = 1.0  # m
   freqs = string_mode_frequencies(length=L, tension=T, density=mu, n_modes=5)
   print("Mode frequencies:", [f"{f:.1f}" for f in freqs])

API Reference
-------------

Classes
~~~~~~~

.. py:class:: VibratingString(length, wave_speed)

   Solver for vibrating string.
   
   .. py:method:: fundamental_frequency()
   
      Get first harmonic frequency :math:`f_1 = c/(2L)`.
   
   .. py:method:: mode_frequency(n)
   
      Get frequency of mode n: :math:`f_n = nc/(2L)`.
   
   .. py:method:: mode_shape(n, x)
   
      Get mode shape :math:`\sin(n\pi x/L)`.
   
   .. py:method:: compute_modes(n_modes)
   
      Compute list of WaveMode objects.
   
   .. py:method:: fourier_coefficients(initial_shape, x_points, n_modes)
   
      Compute Fourier coefficients for initial displacement.
   
   .. py:method:: solution(x, t, coefficients)
   
      Compute :math:`u(x,t)` via modal superposition.

.. py:class:: VibratingMembrane(length_x, length_y, wave_speed)

   Solver for rectangular vibrating membrane.
   
   .. py:method:: mode_frequency(m, n)
   
      Get frequency :math:`\omega_{mn} = \pi c\sqrt{(m/a)^2 + (n/b)^2}`.
   
   .. py:method:: mode_shape(m, n, x, y)
   
      Get 2D mode shape.
   
   .. py:method:: compute_modes(max_m, max_n)
   
      Get sorted list of (m, n, frequency) tuples.

.. py:class:: LagrangianDensity

   Construct Lagrangian densities.
   
   .. py:method:: string_lagrangian()
   
      String: :math:`\mathcal{L} = \frac{1}{2}\mu u_t^2 - \frac{1}{2}T u_x^2`.
   
   .. py:method:: membrane_lagrangian()
   
      Membrane Lagrangian density.
   
   .. py:method:: klein_gordon_lagrangian()
   
      Relativistic scalar field.

.. py:class:: FieldEulerLagrange

   Derive field equations.
   
   .. py:method:: derive_field_equation(lagrangian, field, coordinates)
   
      Apply field Euler-Lagrange equation.
   
   .. py:method:: wave_equation_1d()
   
      Derive 1D wave equation.

.. py:class:: WaveMode

   Normal mode representation.
   
   .. py:attribute:: mode_number
   
      Mode index
   
   .. py:attribute:: frequency
   
      Angular frequency ω
   
   .. py:attribute:: wavenumber
   
      Wave number k

Functions
~~~~~~~~~

.. py:function:: string_mode_frequencies(length, tension, density, n_modes)

   Compute mode frequencies for a string.

.. py:function:: wave_speed(tension, density)

   Compute :math:`c = \sqrt{T/\mu}`.

See Also
--------

- :doc:`oscillations` - Normal modes of discrete systems
- :doc:`lagrangian_mechanics` - Discrete Lagrangian mechanics
