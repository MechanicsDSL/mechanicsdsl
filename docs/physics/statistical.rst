Statistical Mechanics
=====================

The statistical mechanics domain provides tools for calculating thermodynamic
properties from microscopic physics.

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

The statistical mechanics module implements:

- **Boltzmann distribution** for thermal equilibrium
- **Ideal gas** calculations
- **Ising model** for magnetic systems
- **Fermi-Dirac** distribution for fermions
- **Bose-Einstein** distribution for bosons

Quick Start
-----------

.. code-block:: python

   from mechanics_dsl.domains.statistical import (
       BoltzmannDistribution, IdealGas, IsingModel,
       FermiDirac, BoseEinstein
   )
   
   # Boltzmann distribution
   boltz = BoltzmannDistribution(temperature=300)
   v_rms = boltz.rms_speed(mass=4.65e-26)  # N2 molecules
   
   # Ideal gas
   gas = IdealGas(n_moles=1.0, temperature=273.15, volume=0.0224)
   P = gas.pressure()  # ~1 atm
   
   # Ising model
   ising = IsingModel(L=20, dimension=2, temperature=2.5)
   ising.initialize_random()
   ising.monte_carlo_sweep()
   M = ising.magnetization_density()

Classes
-------

BoltzmannDistribution
^^^^^^^^^^^^^^^^^^^^^

Thermal equilibrium probability distribution:

.. math::

   P(E) \propto g(E) \exp\left(-\frac{E}{k_B T}\right)

**Key methods:**

- ``boltzmann_factor(E)``: exp(-βE)
- ``maxwell_speed_distribution(v, m)``: Speed distribution
- ``most_probable_speed(m)``: v_p = √(2kT/m)
- ``rms_speed(m)``: v_rms = √(3kT/m)

IdealGas
^^^^^^^^

Ideal gas equation of state: PV = NkT

**Methods:**

- ``pressure()``: Calculate from T, V, N
- ``internal_energy(f)``: U = (f/2)NkT
- ``heat_capacity_V()``: C_V = (f/2)Nk
- ``entropy()``: Sackur-Tetrode formula

IsingModel
^^^^^^^^^^

Ising model Hamiltonian:

.. math::

   H = -J \sum_{\langle i,j \rangle} s_i s_j - h \sum_i s_i

**Features:**

- 1D, 2D, 3D lattices
- Metropolis Monte Carlo dynamics
- Critical temperature calculation (2D exact)

FermiDirac & BoseEinstein
^^^^^^^^^^^^^^^^^^^^^^^^^

Quantum statistics:

.. math::

   f_{FD}(E) = \frac{1}{e^{(E-\mu)/(k_B T)} + 1}

.. math::

   n_{BE}(E) = \frac{1}{e^{(E-\mu)/(k_B T)} - 1}

Physical Constants
------------------

.. code-block:: python

   from mechanics_dsl.domains.statistical import (
       BOLTZMANN_CONSTANT,  # 1.38e-23 J/K
       AVOGADRO_NUMBER,     # 6.02e23 /mol
       GAS_CONSTANT,        # 8.314 J/(mol·K)
       PLANCK_CONSTANT      # 6.626e-34 J·s
   )

See Also
--------

- :doc:`thermodynamics` - Macroscopic thermodynamics
- :doc:`quantum` - Quantum mechanics

