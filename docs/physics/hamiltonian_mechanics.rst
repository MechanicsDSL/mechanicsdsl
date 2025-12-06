Hamiltonian Mechanics
=====================

For chaotic systems or long-duration simulations, the **Hamiltonian** formulation offers superior energy conservation properties compared to Lagrangian mechanics.

Legendre Transform
------------------
The compiler performs a symbolic Legendre transform to convert user-defined Lagrangians into Hamiltonians:

.. math::

    \mathcal{H}(q, p, t) = \sum_i \dot{q}_i p_i - \mathcal{L}

    p_i = \frac{\partial \mathcal{L}}{\partial \dot{q}_i}

Symplectic Integration
----------------------
The generated C++ code for Hamiltonian systems uses symplectic integrators which preserve the phase-space volume, ensuring that energy drift is bounded even over millions of time steps. This is critical for orbital mechanics (e.g., the Figure-8 solution).
