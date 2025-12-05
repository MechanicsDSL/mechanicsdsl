Lagrangian Mechanics
====================

The primary abstraction of MechanicsDSL is the **Lagrangian** :math:`\mathcal{L}`, defined as the difference between kinetic and potential energy.

.. math::

    \mathcal{L}(q, \dot{q}, t) = T(q, \dot{q}) - V(q)

Equations of Motion
-------------------
The compiler automatically derives the equations of motion using the Euler-Lagrange equation:

.. math::

    \frac{d}{dt} \left( \frac{\partial \mathcal{L}}{\partial \dot{q}_i} \right) - \frac{\partial \mathcal{L}}{\partial q_i} = Q_i

Where :math:`Q_i` represents generalized non-conservative forces (like drag or friction).

Symbolic Derivation Pipeline
----------------------------
1.  **Velocity Jacobian**: The compiler identifies the dependence of :math:`T` on :math:`\dot{q}`.
2.  **Hessian Matrix**: It computes the mass matrix :math:`M_{ij} = \frac{\partial^2 \mathcal{L}}{\partial \dot{q}_i \partial \dot{q}_j}`.
3.  **Linear Solve**: It symbolically solves the linear system :math:`M \ddot{q} = \mathbf{F}` to isolate accelerations.

Code Example
------------
.. code-block:: latex

    \system{double_pendulum}
    \lagrangian{ 0.5*(m1+m2)*l1^2*\dot{t1}^2 + ... }
