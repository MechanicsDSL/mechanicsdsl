Thermodynamics
==============

The thermodynamics domain provides tools for heat engines, equations of state,
and phase transitions.

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

The thermodynamics module implements:

- **Heat engines**: Carnot, Otto, Diesel cycles
- **Equations of state**: Ideal gas, van der Waals
- **Phase transitions**: Clausius-Clapeyron equation
- **Heat capacity**: Debye and Einstein models

Quick Start
-----------

.. code-block:: python

   from mechanics_dsl.domains.thermodynamics import (
       CarnotEngine, OttoCycle, VanDerWaalsGas, PhaseTransition
   )
   
   # Carnot engine
   engine = CarnotEngine(T_hot=500, T_cold=300)
   eta = engine.efficiency()  # 0.4 (40%)
   
   # Otto cycle (gasoline engine)
   otto = OttoCycle(compression_ratio=10, gamma=1.4)
   eta_otto = otto.efficiency()  # ~60%
   
   # Van der Waals gas
   co2 = VanDerWaalsGas(a=0.364, b=4.27e-5)
   P = co2.pressure(V=0.001, T=300)

Classes
-------

CarnotEngine
^^^^^^^^^^^^

Ideal reversible heat engine with maximum efficiency:

.. math::

   \eta = 1 - \frac{T_{cold}}{T_{hot}}

**Methods:**

- ``efficiency()``: Carnot efficiency
- ``work_output(Q_hot)``: Work from heat input
- ``cop_refrigerator()``: Cooling coefficient
- ``cop_heat_pump()``: Heating coefficient

OttoCycle
^^^^^^^^^

Gasoline engine cycle with adiabatic compression/expansion:

.. math::

   \eta = 1 - \frac{1}{r^{\gamma-1}}

where r is the compression ratio.

DieselCycle
^^^^^^^^^^^

Compression ignition engine with isobaric combustion.

VanDerWaalsGas
^^^^^^^^^^^^^^

Real gas equation of state:

.. math::

   \left(P + \frac{a}{V^2}\right)(V - b) = RT

**Methods:**

- ``pressure(V, T)``: Calculate pressure
- ``critical_point()``: Returns (P_c, V_c, T_c)
- ``compressibility_factor(V, T)``: Z = PV/(nRT)

PhaseTransition
^^^^^^^^^^^^^^^

Phase boundary calculations using Clausius-Clapeyron:

.. math::

   \frac{dP}{dT} = \frac{L}{T \Delta V}

HeatCapacity
^^^^^^^^^^^^

**Debye model** for solids:

.. math::

   C_V = 9nR \left(\frac{T}{\theta_D}\right)^3 \int_0^{\theta_D/T} \frac{x^4 e^x}{(e^x-1)^2} dx

**Einstein model** for optical phonons.

Maxwell Relations
-----------------

The module includes all four Maxwell relations derived from thermodynamic potentials.

Physical Constants
------------------

.. code-block:: python

   from mechanics_dsl.domains.thermodynamics import (
       R_GAS,      # 8.314 J/(mol·K)
       BOLTZMANN   # 1.38e-23 J/K
   )

See Also
--------

- :doc:`statistical` - Statistical mechanics
- :doc:`../physics/fluids` - Fluid dynamics

