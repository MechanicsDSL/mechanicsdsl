Physics Background
==================

MechanicsDSL automates the **Lagrangian** and **Hamiltonian** formalisms of classical mechanics.

Lagrangian Mechanics
--------------------

Lagrangian mechanics describes a system using its generalized coordinates :math:`q_i` and their time derivatives :math:`\dot{q}_i`. The central quantity is the **Lagrangian**:

.. math::

    L(q, \dot{q}, t) = T - V

where :math:`T` is the kinetic energy and :math:`V` is the potential energy. The equations of motion are derived from the Principle of Least Action, leading to the **Euler-Lagrange equations**:

.. math::

    \frac{d}{dt} \left( \frac{\partial L}{\partial \dot{q}_i} \right) - \frac{\partial L}{\partial q_i} = Q_i

where :math:`Q_i` represents non-conservative generalized forces.

Hamiltonian Mechanics
---------------------

MechanicsDSL can also perform the Legendre transform to convert a Lagrangian system into the Hamiltonian formalism. The **Hamiltonian** is defined as:

.. math::

    H(q, p, t) = \sum_i p_i \dot{q}_i - L

where :math:`p_i = \frac{\partial L}{\partial \dot{q}_i}` are the conjugate momenta. The dynamics are given by **Hamilton's equations**:

.. math::

    \dot{q}_i = \frac{\partial H}{\partial p_i}, \quad \dot{p}_i = -\frac{\partial H}{\partial q_i}

This formulation is particularly useful for symplectic integration and analyzing phase space flow.
