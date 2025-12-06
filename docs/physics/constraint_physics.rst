Constraint Handling
===================

MechanicsDSL supports **Holonomic Constraints** of the form :math:`f(q_1, ..., q_n, t) = 0`.

Lagrange Multipliers
--------------------
When a ``\constraint{}`` is defined, the compiler augments the Lagrangian:

.. math::

    \mathcal{L}' = \mathcal{L} + \sum_k \lambda_k f_k(q, t)

This introduces new unknowns (:math:`\lambda_k`, the forces of constraint). The compiler solves for these multipliers simultaneously with the accelerations, effectively projecting the dynamics onto the constraint manifold.

Example: Bead on a Wire
-----------------------
.. code-block:: latex

    \constraint{x^2 + y^2 - R^2} % Constrain to circle
