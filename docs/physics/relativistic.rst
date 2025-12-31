Special Relativistic Mechanics
==============================

The relativistic domain provides tools for special relativistic dynamics,
Lorentz transformations, and four-vector calculations.

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

The relativistic module implements:

- **Relativistic particle dynamics** with proper Lagrangian/Hamiltonian
- **Four-vectors** with Lorentz invariants
- **Lorentz transformations** (boosts, velocity addition)
- **Relativistic collisions** and invariant mass calculations

The relativistic Lagrangian is:

.. math::

   L = -mc^2\sqrt{1 - \frac{v^2}{c^2}} - V(\mathbf{r})

Quick Start
-----------

.. code-block:: python

   from mechanics_dsl.domains.relativistic import RelativisticParticle, gamma
   
   # Create a relativistic particle
   particle = RelativisticParticle(mass=1.0)
   particle.set_parameter('c', 1.0)  # Natural units
   
   # Calculate Lorentz factor at v = 0.8c
   g = particle.lorentz_factor(0.8)  # γ = 5/3 ≈ 1.667
   
   # Relativistic momentum: p = γmv
   p = particle.relativistic_momentum(0.8)  # p ≈ 1.333
   
   # Total energy: E = γmc²
   E = particle.relativistic_energy(0.8)  # E ≈ 1.667
   
   # Kinetic energy: T = (γ-1)mc²
   T = particle.kinetic_energy(0.8)  # T ≈ 0.667

Classes
-------

RelativisticParticle
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: mechanics_dsl.domains.relativistic.RelativisticParticle
   :members:
   :show-inheritance:

**Example: Relativistic dynamics with external field**

.. code-block:: python

   import sympy as sp
   from mechanics_dsl.domains.relativistic import RelativisticParticle
   
   particle = RelativisticParticle(mass=1.0)
   
   # Add a potential V(x) = kx²/2
   x = sp.Symbol('x')
   particle.set_potential(0.5 * x**2)
   
   # Get Lagrangian and Hamiltonian
   L = particle.define_lagrangian()
   H = particle.define_hamiltonian()

FourVector
^^^^^^^^^^

.. autoclass:: mechanics_dsl.domains.relativistic.FourVector
   :members:

**Example: Spacetime intervals**

.. code-block:: python

   from mechanics_dsl.domains.relativistic import FourVector
   
   # Event at (ct=5, x=3, y=0, z=0)
   event = FourVector(ct=5.0, x=3.0, y=0.0, z=0.0)
   
   # Lorentz invariant: s² = (ct)² - x² - y² - z²
   s2 = event.invariant()  # 25 - 9 = 16 (timelike)
   
   print(event.is_timelike())   # True (s² > 0)
   print(event.is_spacelike())  # False
   print(event.is_lightlike())  # False
   
   # Light ray from origin
   light = FourVector(ct=5.0, x=3.0, y=4.0, z=0.0)
   print(light.invariant())     # 0 (null/lightlike)
   print(light.is_lightlike())  # True

LorentzTransform
^^^^^^^^^^^^^^^^

.. autoclass:: mechanics_dsl.domains.relativistic.LorentzTransform
   :members:

**Example: Lorentz boost**

.. code-block:: python

   from mechanics_dsl.domains.relativistic import FourVector, LorentzTransform
   
   # Event in lab frame
   event = FourVector(ct=10.0, x=5.0, y=0.0, z=0.0)
   
   # Boost to frame moving at v = 0.6c in x-direction
   boosted = LorentzTransform.boost_x(event, v=0.6, c=1.0)
   print(f"ct' = {boosted.ct:.3f}, x' = {boosted.x:.3f}")
   
   # Invariant is preserved!
   assert abs(event.invariant() - boosted.invariant()) < 1e-10

**Example: Relativistic velocity addition**

.. code-block:: python

   from mechanics_dsl.domains.relativistic import LorentzTransform
   
   # Spaceship at 0.5c fires missile at 0.5c
   v_combined = LorentzTransform.velocity_addition(0.5, 0.5, c=1.0)
   print(f"Combined velocity: {v_combined}c")  # 0.8c, not 1.0c!
   
   # Even 0.9c + 0.9c gives less than c
   v = LorentzTransform.velocity_addition(0.9, 0.9, c=1.0)
   print(f"0.9c + 0.9c = {v:.4f}c")  # 0.9945c

Key Formulas
------------

**Lorentz Factor**

.. math::

   \gamma = \frac{1}{\sqrt{1 - v^2/c^2}}

**Energy-Momentum Relation**

.. math::

   E^2 = (pc)^2 + (mc^2)^2

**Velocity Addition**

.. math::

   u = \frac{v_1 + v_2}{1 + v_1 v_2 / c^2}

**Time Dilation**

.. math::

   \Delta t = \gamma \Delta \tau

**Length Contraction**

.. math::

   L = L_0 / \gamma

Convenience Functions
---------------------

.. code-block:: python

   from mechanics_dsl.domains.relativistic import gamma, beta, rapidity
   
   # Quick Lorentz factor
   g = gamma(0.8, c=1.0)  # 5/3
   
   # β = v/c
   b = beta(0.6, c=1.0)  # 0.6
   
   # Rapidity η = arctanh(v/c)
   # Rapidities add linearly: η₁₂ = η₁ + η₂
   eta = rapidity(0.6, c=1.0)

See Also
--------

- :doc:`electromagnetic` - For charged particle dynamics
- :doc:`quantum` - For quantum mechanics
