Fluid Dynamics (SPH)
====================

MechanicsDSL implements a mesh-free Lagrangian fluid solver using **Smoothed Particle Hydrodynamics (SPH)**.

Governing Equations
-------------------
Unlike grid-based CFD (Eulerian), SPH discretizes the fluid into particles carrying properties (mass, density, pressure).

**Density Summation (Poly6 Kernel):**

.. math::

    \rho_i = \sum_j m_j W(|\mathbf{r}_{ij}|, h)

**Momentum Equation (Navier-Stokes):**

.. math::

    \frac{d\mathbf{v}_i}{dt} = -\sum_j m_j \left( \frac{P_i}{\rho_i^2} + \frac{P_j}{\rho_j^2} \right) \nabla W_{ij} + \mathbf{g}

Tait Equation of State
----------------------
To simulate water as a weakly compressible fluid without solving a Poisson equation, we use the Tait EOS:

.. math::

    P = B \left( \left( \frac{\rho}{\rho_0} \right)^\gamma - 1 \right)

Where :math:`\gamma = 7` creates a "stiff" pressure response that resists compression.

Compiler Implementation
-----------------------
When ``\fluid`` is detected, the compiler switches strategies:
1.  **Generates Particles**: Using ``ParticleGenerator`` based on ``\region`` geometry.
2.  **Switches Integrator**: Uses **Velocity Verlet** instead of RK4 for symplectic stability.
3.  **Spatial Hashing**: Generates C++ code for an :math:`O(N)` neighbor search grid.
