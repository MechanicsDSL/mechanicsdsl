DSL Syntax Reference
====================

This document provides a complete reference for the MechanicsDSL domain-specific 
language syntax. The DSL uses LaTeX-inspired notation to describe physical systems
in natural mathematical form.

Overview
--------

The DSL consists of **commands** that define various aspects of a physical system:

- System identification
- Variable and parameter definitions
- Energy functions (Lagrangian, Hamiltonian)
- Constraints and forces
- Initial conditions
- Simulation and output directives

All commands use the LaTeX backslash syntax: ``\command{arguments}``.

Comments
--------

Comments start with ``%`` and continue to the end of the line:

.. code-block:: latex

   % This is a comment
   \system{pendulum}  % Inline comment

System Definition
-----------------

\system{name}
~~~~~~~~~~~~~

Defines the name of the physical system. Must be the first command.

.. code-block:: latex

   \system{double_pendulum}

**Arguments:**

- ``name``: System identifier (alphanumeric, underscores allowed)

Variable Definitions
--------------------

\defvar{name}{type}{unit}
~~~~~~~~~~~~~~~~~~~~~~~~~

Defines a generalized coordinate or velocity.

.. code-block:: latex

   \defvar{theta}{angle}{rad}
   \defvar{x}{position}{m}
   \defvar{q1}{generalized}{dimensionless}

**Arguments:**

- ``name``: Variable name (becomes a symbol in equations)
- ``type``: Semantic type (``angle``, ``position``, ``generalized``, etc.)
- ``unit``: Physical unit (informational, not enforced)

**Automatic Derivatives:**

For each variable ``q``, the system automatically creates:

- ``q_dot`` (first time derivative, velocity)
- ``q_ddot`` (second time derivative, acceleration)

Parameter Definitions
---------------------

\parameter{name}{value}{unit}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defines a constant physical parameter.

.. code-block:: latex

   \parameter{m}{1.5}{kg}
   \parameter{l}{0.8}{m}
   \parameter{g}{9.81}{m/s^2}
   \parameter{k}{100.0}{N/m}

**Arguments:**

- ``name``: Parameter identifier
- ``value``: Numerical value (float or integer)
- ``unit``: Physical unit

Energy Functions
----------------

\lagrangian{expression}
~~~~~~~~~~~~~~~~~~~~~~~

Defines the Lagrangian L = T - V for the system.

.. code-block:: latex

   % Simple pendulum
   \lagrangian{
       \frac{1}{2} m l^2 \dot{theta}^2 - m g l (1 - \cos(theta))
   }
   
   % Spring-mass system
   \lagrangian{
       \frac{1}{2} m \dot{x}^2 - \frac{1}{2} k x^2
   }
   
   % Double pendulum (complex)
   \lagrangian{
       \frac{1}{2} (m_1 + m_2) l_1^2 \dot{theta_1}^2 +
       \frac{1}{2} m_2 l_2^2 \dot{theta_2}^2 +
       m_2 l_1 l_2 \dot{theta_1} \dot{theta_2} \cos(theta_1 - theta_2) +
       (m_1 + m_2) g l_1 \cos(theta_1) +
       m_2 g l_2 \cos(theta_2)
   }

\hamiltonian{expression}
~~~~~~~~~~~~~~~~~~~~~~~~

Defines the Hamiltonian H = T + V for the system (alternative to Lagrangian).

.. code-block:: latex

   \hamiltonian{
       \frac{p_theta^2}{2 m l^2} + m g l (1 - \cos(theta))
   }

Mathematical Notation
---------------------

Expressions support standard mathematical notation:

Derivatives
~~~~~~~~~~~

.. code-block:: latex

   \dot{theta}      % First derivative (velocity)
   \ddot{theta}     % Second derivative (acceleration)
   theta_dot        % Alternative notation
   theta_ddot       % Alternative notation

Fractions
~~~~~~~~~

.. code-block:: latex

   \frac{1}{2}      % One half
   \frac{p^2}{2m}   % p² / 2m

Trigonometric Functions
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: latex

   \sin(theta)
   \cos(theta)
   \tan(theta)
   \asin(x)
   \acos(x)
   \atan(x)
   \atan2(y, x)

Other Functions
~~~~~~~~~~~~~~~

.. code-block:: latex

   \sqrt{x}         % Square root
   \exp(x)          % Exponential
   \log(x)          % Natural logarithm
   \abs{x}          % Absolute value
   x^2              % Power
   x^{3/2}          % Fractional power

Greek Letters
~~~~~~~~~~~~~

.. code-block:: latex

   \theta, \phi, \psi    % Angles
   \omega                 % Angular velocity
   \alpha, \beta          % Generic parameters
   \lambda                % Lagrange multipliers
   \rho                   % Density
   \mu                    % Friction coefficient

Operators
~~~~~~~~~

.. code-block:: latex

   +, -, *, /       % Basic arithmetic
   ^                % Exponentiation
   ( )              % Grouping

Constraints
-----------

\constraint{expression}
~~~~~~~~~~~~~~~~~~~~~~~

Defines a holonomic constraint g(q) = 0.

.. code-block:: latex

   % Pendulum on a circle
   \constraint{x^2 + y^2 - l^2}
   
   % Rod constraint in double pendulum
   \constraint{
       (x_2 - x_1)^2 + (y_2 - y_1)^2 - l_2^2
   }

\nonholonomic{expression}
~~~~~~~~~~~~~~~~~~~~~~~~~

Defines a non-holonomic (velocity-dependent) constraint.

.. code-block:: latex

   % Rolling without slipping
   \nonholonomic{v - r \omega}
   
   % Knife edge constraint
   \nonholonomic{\dot{x} \sin(theta) - \dot{y} \cos(theta)}

Forces
------

\force{coordinate}{expression}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adds an external non-conservative force to a coordinate.

.. code-block:: latex

   % Applied torque
   \force{theta}{tau}
   
   % Time-dependent force
   \force{x}{F_0 \cos(\omega t)}
   
   % Position-dependent force
   \force{x}{-k_nonlinear x^3}

\damping{coefficient}
~~~~~~~~~~~~~~~~~~~~~

Adds linear damping (viscous friction) to all coordinates.

.. code-block:: latex

   \damping{0.1}  % Damping coefficient

   % Per-coordinate damping
   \damping{theta}{0.05}
   \damping{x}{0.2}

Initial Conditions
------------------

\initial{assignments}
~~~~~~~~~~~~~~~~~~~~~

Sets initial conditions for simulation.

.. code-block:: latex

   % Position and velocity
   \initial{theta=0.5, theta_dot=0}
   
   % Multiple coordinates
   \initial{theta_1=1.0, theta_1_dot=0, theta_2=0.5, theta_2_dot=0}
   
   % Using pi
   \initial{theta=\frac{\pi}{4}, theta_dot=0}

Fluid Dynamics
--------------

\fluid{name}
~~~~~~~~~~~~

Defines an SPH fluid region.

.. code-block:: latex

   \fluid{water}
   \region{0, 0, 0.5, 0.5}
   \parameter{h}{0.05}{m}          % Smoothing length
   \parameter{rho_0}{1000}{kg/m^3} % Rest density
   \parameter{mu}{0.001}{Pa.s}     % Viscosity

\boundary{name}
~~~~~~~~~~~~~~~

Defines a boundary region for fluids.

.. code-block:: latex

   \boundary{container}
   \region{-0.1, -0.1, 1.1, 0}  % Bottom wall
   \region{-0.1, -0.1, 0, 1.1}  % Left wall

\region{x1, y1, x2, y2}
~~~~~~~~~~~~~~~~~~~~~~~

Defines a rectangular region (for fluids or boundaries).

.. code-block:: latex

   \region{0, 0, 1, 0.5}  % Rectangle from (0,0) to (1, 0.5)

Simulation Control
------------------

\solve{method}
~~~~~~~~~~~~~~

Specifies the numerical integration method.

.. code-block:: latex

   \solve{RK45}      % Runge-Kutta 4-5 (default)
   \solve{LSODA}     % Automatic stiff/non-stiff
   \solve{Radau}     % Implicit for stiff systems
   \solve{BDF}       % Backward differentiation
   \solve{DOP853}    % High-order Runge-Kutta

Output Commands
---------------

\animate{target}
~~~~~~~~~~~~~~~~

Requests animation of specific outputs.

.. code-block:: latex

   \animate{pendulum}       % Animate pendulum motion
   \animate{phase_space}    % Animate phase portrait
   \animate{energy}         % Animate energy components

\export{filename}
~~~~~~~~~~~~~~~~~

Export results or generated code.

.. code-block:: latex

   \export{results.csv}     % Export simulation data
   \export{simulation.cpp}  % Generate C++ code
   \export{sim.wasm}        % Generate WebAssembly

\import{filename}
~~~~~~~~~~~~~~~~~

Import external definitions.

.. code-block:: latex

   \import{common_parameters.mdsl}

Custom Operators
----------------

\define{\op{name}(args) = expression}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define custom mathematical operators.

.. code-block:: latex

   % Define kinetic energy operator
   \define{\op{KE}(m, v) = \frac{1}{2} m v^2}
   
   % Use in Lagrangian
   \lagrangian{\op{KE}(m, \dot{x}) - \frac{1}{2} k x^2}

Coordinate Transforms
---------------------

\transform{type}{definitions}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define coordinate transformations.

.. code-block:: latex

   % Polar to Cartesian
   \transform{cartesian}{
       x = r \cos(\theta)
       y = r \sin(\theta)
   }
   
   % Spherical coordinates
   \transform{spherical}{
       x = r \sin(\phi) \cos(\theta)
       y = r \sin(\phi) \sin(\theta)
       z = r \cos(\phi)
   }

Complete Example
----------------

Here's a comprehensive example demonstrating multiple features:

.. code-block:: latex

   % Damped Driven Pendulum
   \system{damped_driven_pendulum}
   
   % Generalized coordinate
   \defvar{theta}{angle}{rad}
   
   % Physical parameters
   \parameter{m}{0.5}{kg}
   \parameter{l}{0.4}{m}
   \parameter{g}{9.81}{m/s^2}
   \parameter{gamma}{0.1}{1/s}     % Damping coefficient
   \parameter{F_0}{1.2}{N.m}       % Driving amplitude
   \parameter{omega_d}{2.0}{rad/s} % Driving frequency
   
   % Lagrangian (kinetic - potential)
   \lagrangian{
       \frac{1}{2} m l^2 \dot{theta}^2 - m g l (1 - \cos(theta))
   }
   
   % Damping term
   \damping{gamma}
   
   % Periodic driving force
   \force{theta}{F_0 \cos(\omega_d t)}
   
   % Initial conditions
   \initial{theta=0.1, theta_dot=0}
   
   % Use LSODA for potentially stiff equations
   \solve{LSODA}
   
   % Output
   \animate{pendulum}
   \export{results.csv}

Syntax Errors
-------------

Common errors and solutions:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Error
     - Solution
   * - ``Unexpected token``
     - Check for missing backslashes or braces
   * - ``Undefined variable``
     - Ensure ``\defvar`` is called before use
   * - ``Undefined parameter``
     - Ensure ``\parameter`` is called before use
   * - ``Unbalanced braces``
     - Count opening and closing braces
   * - ``Invalid expression``
     - Check mathematical syntax
