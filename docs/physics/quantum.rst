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
   # Heisenberg uncertainty minimum
   min_uncertainty = heisenberg_minimum(hbar=1.0)  # ℏ/2

Quantum Tunneling
-----------------

The ``QuantumTunneling`` class provides exact and WKB-based tunneling calculations.

**Rectangular Barrier (Exact)**

For a barrier of height V₀ and width a:

.. math::

   T = \frac{1}{1 + \frac{V_0^2 \sinh^2(\kappa a)}{4E(V_0 - E)}}

where :math:`\kappa = \sqrt{2m(V_0 - E)}/\hbar`.

.. code-block:: python

   from mechanics_dsl.domains.quantum import QuantumTunneling
   
   tunneling = QuantumTunneling(mass=1.0, hbar=1.0)
   
   # Transmission through rectangular barrier
   T = tunneling.rectangular_barrier(E=1.0, V0=2.0, width=1.0)
   print(f"Transmission probability: {T:.4f}")
   
   # WKB for arbitrary potentials
   def gaussian_barrier(x):
       return 2.0 * np.exp(-x**2)
   
   T_wkb = tunneling.wkb_transmission(E=0.5, potential=gaussian_barrier, 
                                       x1=-2, x2=2)

**Alpha Decay (Gamow Factor)**

.. code-block:: python

   from mechanics_dsl.domains.quantum import alpha_decay_rate
   
   # Uranium-238 alpha decay (E_alpha ≈ 4.2 MeV)
   E_alpha = 4.2e6 * 1.602e-19  # Convert MeV to Joules
   Z_thorium = 90  # Daughter nucleus
   
   decay_rate = alpha_decay_rate(E_alpha, Z_daughter=Z_thorium)

Finite Square Well
------------------

The ``FiniteSquareWell`` solves the transcendental eigenvalue equations:

.. math::

   \text{Even parity:} \quad k \tan(ka/2) = \kappa

.. math::

   \text{Odd parity:} \quad -k \cot(ka/2) = \kappa

where :math:`k = \sqrt{2m(E+V_0)}/\hbar` and :math:`\kappa = \sqrt{-2mE}/\hbar`.

.. code-block:: python

   from mechanics_dsl.domains.quantum import FiniteSquareWell
   
   well = FiniteSquareWell(depth=10.0, width=2.0, mass=1.0, hbar=1.0)
   
   # Find all bound states
   states = well.find_bound_states()
   for state in states:
       print(f"n={state.n}: E = {state.energy:.4f}")
   
   # Scattering transmission (E > 0)
   T = well.transmission_coefficient(E=5.0)

Hydrogen Atom
-------------

Exact energy levels for hydrogen-like atoms:

.. math::

   E_n = -\frac{Z^2 \times 13.6 \text{ eV}}{n^2}

.. code-block:: python

   from mechanics_dsl.domains.quantum import HydrogenAtom
   
   H = HydrogenAtom(Z=1)
   
   # Ground state energy
   E1 = H.energy_level(n=1)  # -13.6 eV
   
   # Bohr radius
   r1 = H.bohr_radius_n(n=1)  # 5.29e-11 m
   
   # Lyman-alpha wavelength (2→1)
   lam_alpha = H.transition_wavelength(n_initial=2, n_final=1)  # 121.6 nm
   
   # Balmer series (visible spectrum)
   balmer = H.spectral_series(n_final=2, n_max=7)
   
   # Helium ion (Z=2)
   He_plus = HydrogenAtom(Z=2)
   E1_He = He_plus.energy_level(n=1)  # -54.4 eV

Step Potential & Delta Barrier
------------------------------

.. code-block:: python

   from mechanics_dsl.domains.quantum import StepPotential, DeltaFunctionBarrier
   
   # Step potential
   step = StepPotential(height=5.0)
   R, T = step.reflection_transmission(E=10.0)
   print(f"R = {R:.3f}, T = {T:.3f}")
   
   # Delta function barrier: V(x) = λδ(x)
   barrier = DeltaFunctionBarrier(strength=1.0)
   T = barrier.transmission(E=0.5)
   
   # Attractive delta well has one bound state
   well = DeltaFunctionBarrier(strength=-1.0)
   E_bound = well.bound_state_energy()  # -0.5

See Also
--------

- :doc:`relativistic` - For relativistic quantum extensions
- :doc:`../physics/oscillations` - For classical oscillator comparison

