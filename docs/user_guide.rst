User Guide
==========

The MechanicsDSL syntax is inspired by LaTeX but simplified for machine parsing. Every command starts with a backslash ``\``.

Defining Variables
------------------

Variables represent the generalized coordinates of your system (degrees of freedom).

.. code-block:: latex

    \var{name}{description}{unit}

**Example:**

.. code-block:: latex

    \var{theta}{Angle}{rad}
    \var{x}{Position}{m}

Defining Parameters
-------------------

Parameters are constants that define the physical properties of the system.

.. code-block:: latex

    \parameter{name}{value}{unit}

**Example:**

.. code-block:: latex

    \parameter{g}{9.81}{m/s^2}
    \parameter{m}{1.0}{kg}

The Lagrangian
--------------

The Lagrangian :math:`L = T - V` defines the system's dynamics. You can use standard math operations (`+`, `-`, `*`, `/`, `^`) and functions (`\sin`, `\cos`, `\exp`, etc.).

Time derivatives are denoted by ``\dot{x}``.

.. code-block:: latex

    \lagrangian{\frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2}

Forces and Damping
------------------

Non-conservative forces (like friction or drive) can be added using ``\force`` or ``\damping``. These terms appear on the right-hand side of the Euler-Lagrange equations.

.. code-block:: latex

    \damping{-b * \dot{x}}
    \force{F0 * \cos{omega * t}}

Initial Conditions
------------------

Set the starting state for the simulation.

.. code-block:: latex

    \initial{x=1.0, x_dot=0.0}

Solver Configuration
--------------------

You can specify the numerical method explicitly, though the engine defaults to adaptive selection.

.. code-block:: latex

    \solve{LSODA}
