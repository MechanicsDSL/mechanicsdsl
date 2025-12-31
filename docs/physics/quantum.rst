Quantum Mechanics
=================

The quantum domain provides tools for semiclassical quantum mechanics,
WKB approximation, and exact solutions for standard potentials.

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

The quantum module implements:

- **WKB approximation** for semiclassical analysis
- **Bohr-Sommerfeld quantization** for bound states
- **Quantum harmonic oscillator** exact solutions
- **Infinite square well** (particle in a box)
- **Ehrenfest dynamics** (quantum-classical correspondence)

Quick Start
-----------

.. code-block:: python

   from mechanics_dsl.domains.quantum import (
       QuantumHarmonicOscillator,
       InfiniteSquareWell,
       WKBApproximation
   )
   import numpy as np

   # Quantum harmonic oscillator
   qho = QuantumHarmonicOscillator(mass=1.0, omega=1.0, hbar=1.0)
   
   # Energy levels: E_n = ℏω(n + 1/2)
   E0 = qho.energy_level(0)  # 0.5 (ground state)
   E1 = qho.energy_level(1)  # 1.5
   
   # Wavefunction
   x = np.linspace(-5, 5, 1000)
   psi_0 = qho.wavefunction(x, n=0)

Classes
-------

QuantumHarmonicOscillator
^^^^^^^^^^^^^^^^^^^^^^^^^

The quantum harmonic oscillator has the Hamiltonian:

.. math::

   H = \frac{p^2}{2m} + \frac{1}{2}m\omega^2 x^2

with exact energy eigenvalues :math:`E_n = \hbar\omega(n + \frac{1}{2})`.

.. autoclass:: mechanics_dsl.domains.quantum.QuantumHarmonicOscillator
   :members:
   :show-inheritance:

**Example: Ground state properties**

.. code-block:: python

   from mechanics_dsl.domains.quantum import QuantumHarmonicOscillator
   import numpy as np
   import matplotlib.pyplot as plt

   qho = QuantumHarmonicOscillator(mass=1.0, omega=1.0, hbar=1.0)
   
   # Ground state energy (zero-point energy)
   E0 = qho.zero_point_energy()  # ℏω/2 = 0.5
   
   # Characteristic length scale
   a = qho.characteristic_length()  # √(ℏ/mω) = 1.0
   
   # Classical turning point
   x_classical = qho.classical_amplitude(n=0)
   
   # Plot probability density
   x = np.linspace(-4, 4, 500)
   prob = qho.probability_density(x, n=0)
   
   plt.plot(x, prob)
   plt.xlabel('x')
   plt.ylabel('|ψ₀(x)|²')
   plt.title('Ground State Probability Density')

**Example: Uncertainty principle verification**

.. code-block:: python

   from mechanics_dsl.domains.quantum import QuantumHarmonicOscillator
   
   qho = QuantumHarmonicOscillator(hbar=1.0)
   
   # Ground state has minimum uncertainty
   delta_x_delta_p = qho.uncertainty_product(n=0)  # ℏ/2 = 0.5
   
   # Excited states have larger uncertainty
   for n in range(5):
       product = qho.uncertainty_product(n)
       print(f"n={n}: ΔxΔp = {product:.1f}ℏ")

InfiniteSquareWell
^^^^^^^^^^^^^^^^^^

Particle in a box with walls at x=0 and x=L:

.. math::

   E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}, \quad n = 1, 2, 3, \ldots

.. autoclass:: mechanics_dsl.domains.quantum.InfiniteSquareWell
   :members:
   :show-inheritance:

**Example: Energy level spacing**

.. code-block:: python

   from mechanics_dsl.domains.quantum import InfiniteSquareWell
   
   well = InfiniteSquareWell(length=1.0, mass=1.0, hbar=1.0)
   
   # First few energy levels
   for n in range(1, 6):
       E = well.energy_level(n)
       print(f"E_{n} = {E:.3f}")
   
   # Note: E_n ∝ n²

WKBApproximation
^^^^^^^^^^^^^^^^

The WKB (Wentzel-Kramers-Brillouin) approximation is valid when the 
de Broglie wavelength varies slowly compared to the potential.

**Bohr-Sommerfeld quantization condition:**

.. math::

   \oint p \, dx = \left(n + \frac{1}{2}\right) h

.. autoclass:: mechanics_dsl.domains.quantum.WKBApproximation
   :members:
   :show-inheritance:

**Example: Finding energy levels with WKB**

.. code-block:: python

   from mechanics_dsl.domains.quantum import WKBApproximation
   
   # Harmonic oscillator potential
   def V(x):
       return 0.5 * x**2
   
   wkb = WKBApproximation(potential=V, mass=1.0, hbar=1.0)
   
   # Find ground state energy
   E0 = wkb.find_energy_level(n=0, E_range=(0.1, 2.0), x_range=(-5, 5))
   print(f"WKB ground state: E_0 = {E0:.3f}")  # ≈ 0.5
   
   # Compare with exact: E_n = (n + 0.5)ℏω = 0.5

EhrenfestDynamics
^^^^^^^^^^^^^^^^^

Ehrenfest theorem states that expectation values follow classical equations
for quadratic potentials:

.. math::

   \frac{d\langle x \rangle}{dt} = \frac{\langle p \rangle}{m}

.. math::

   \frac{d\langle p \rangle}{dt} = -\left\langle \frac{dV}{dx} \right\rangle

.. autoclass:: mechanics_dsl.domains.quantum.EhrenfestDynamics
   :members:
   :show-inheritance:

**Example: Wave packet evolution**

.. code-block:: python

   from mechanics_dsl.domains.quantum import EhrenfestDynamics
   import numpy as np
   
   # Harmonic oscillator
   def V(x): return 0.5 * x**2
   def dV(x): return x
   
   dynamics = EhrenfestDynamics(potential=V, potential_derivative=dV, mass=1.0)
   
   # Initial conditions: x₀ = 1, p₀ = 0
   result = dynamics.propagate(x0=1.0, p0=0.0, t_span=(0, 10))
   
   # <x>(t) oscillates like classical oscillator
   print(f"Initial <x> = {result['x_exp'][0]:.2f}")
   print(f"Final <x> = {result['x_exp'][-1]:.2f}")

Physical Constants
------------------

.. code-block:: python

   from mechanics_dsl.domains.quantum import HBAR, PLANCK_H
   
   print(f"ℏ = {HBAR:.6e} J·s")        # 1.054571817e-34
   print(f"h = {PLANCK_H:.6e} J·s")     # 6.62607015e-34

Convenience Functions
---------------------

.. code-block:: python

   from mechanics_dsl.domains.quantum import (
       de_broglie_wavelength,
       compton_wavelength,
       heisenberg_minimum
   )
   
   # de Broglie wavelength: λ = h/p = 2πℏ/p
   lam = de_broglie_wavelength(momentum=1e-24)  # For electron-like particle
   
   # Compton wavelength: λ_C = h/(mc)
   lam_C = compton_wavelength(mass=9.11e-31)  # Electron Compton wavelength
   
   # Heisenberg uncertainty minimum
   min_uncertainty = heisenberg_minimum(hbar=1.0)  # ℏ/2

See Also
--------

- :doc:`relativistic` - For relativistic quantum extensions
- :doc:`../physics/oscillations` - For classical oscillator comparison
