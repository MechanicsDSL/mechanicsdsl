Quick Start Guide
=================

This guide will help you get started with MechanicsDSL in just a few minutes.
By the end, you'll understand the basic workflow and be ready to model your own
physical systems.

Prerequisites
-------------

Before you begin, make sure you have:

* Python 3.8 or later
* NumPy, SciPy, SymPy, and Matplotlib installed
* Basic knowledge of classical mechanics (helpful but not required)

Your First Simulation: Simple Pendulum
--------------------------------------

Let's start with the classic example—a simple pendulum.

Step 1: Import and Create Compiler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl import PhysicsCompiler, setup_logging
   
   # Optional: Enable detailed logging
   setup_logging(level='INFO')
   
   # Create the compiler instance
   compiler = PhysicsCompiler()

Step 2: Write Your DSL Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The DSL uses LaTeX-inspired syntax. Here's a complete pendulum definition:

.. code-block:: python

   source = r'''
   % Define the system name
   \system{simple_pendulum}
   
   % Define the generalized coordinate (angle)
   \defvar{theta}{angle}{rad}
   
   % Define physical parameters
   \parameter{m}{1.0}{kg}      % mass at the end
   \parameter{l}{1.0}{m}       % pendulum length
   \parameter{g}{9.81}{m/s^2}  % gravitational acceleration
   
   % Define the Lagrangian: L = T - V
   % T = (1/2) * m * l^2 * theta_dot^2  (kinetic energy)
   % V = m * g * l * (1 - cos(theta))   (potential energy)
   \lagrangian{
       \frac{1}{2} m l^2 \dot{theta}^2 - m g l (1 - \cos(theta))
   }
   
   % Set initial conditions
   \initial{theta=0.5, theta_dot=0}
   '''

Step 3: Compile and Simulate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compile the DSL source
   result = compiler.compile(source)
   
   if result['success']:
       print("Compilation successful!")
       print(f"System: {result['system_name']}")
       print(f"Coordinates: {result['coordinates']}")
       
       # Run the simulation for 10 seconds
       solution = compiler.simulate((0, 10), num_points=1000)
       
       if solution['success']:
           print(f"Simulation complete: {len(solution['t'])} time points")

Step 4: Visualize the Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create an animated visualization
   compiler.visualize(solution)
   
   # Or plot energy conservation
   compiler.plot_energy(solution)
   
   # Show all plots
   import matplotlib.pyplot as plt
   plt.show()

Complete Working Example
------------------------

Here's everything together as a copy-paste ready script:

.. code-block:: python

   #!/usr/bin/env python3
   """
   MechanicsDSL Quick Start: Simple Pendulum
   """
   from mechanics_dsl import PhysicsCompiler
   import matplotlib.pyplot as plt
   
   # Create compiler
   compiler = PhysicsCompiler()
   
   # Define pendulum system
   source = r'''
   \system{simple_pendulum}
   \defvar{theta}{angle}{rad}
   \parameter{m}{1.0}{kg}
   \parameter{l}{1.0}{m}  
   \parameter{g}{9.81}{m/s^2}
   \lagrangian{\frac{1}{2} m l^2 \dot{theta}^2 - m g l (1 - \cos(theta))}
   \initial{theta=0.5, theta_dot=0}
   '''
   
   # Compile and run
   result = compiler.compile(source)
   solution = compiler.simulate((0, 10))
   
   # Visualize
   compiler.visualize(solution)
   plt.show()

Understanding the Output
------------------------

When you run the example, you'll see:

1. **Console Output**: Compilation status, equation derivation steps, and simulation progress
2. **Animation**: A 3D visualization of the pendulum swinging
3. **Energy Plot** (if called): Kinetic, potential, and total energy over time

The Solution Dictionary
~~~~~~~~~~~~~~~~~~~~~~~

The ``solution`` dictionary contains:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Key
     - Type
     - Description
   * - ``success``
     - bool
     - Whether simulation completed successfully
   * - ``t``
     - np.ndarray
     - Time points array
   * - ``y``
     - np.ndarray
     - State vector at each time (shape: 2n × m)
   * - ``coordinates``
     - list
     - Names of generalized coordinates
   * - ``nfev``
     - int
     - Number of function evaluations

Accessing State Variables
~~~~~~~~~~~~~~~~~~~~~~~~~

For a single coordinate system like the pendulum:

.. code-block:: python

   import numpy as np
   
   t = solution['t']
   theta = solution['y'][0]      # Position (angle)
   theta_dot = solution['y'][1]  # Velocity (angular velocity)
   
   # Find maximum angle
   print(f"Max angle: {np.max(np.abs(theta)):.4f} rad")
   
   # Find period (for small oscillations)
   l, g = 1.0, 9.81
   T_theory = 2 * np.pi * np.sqrt(l / g)
   print(f"Theoretical period: {T_theory:.4f} s")

Next Steps
----------

Now that you've completed your first simulation, try:

1. **Change Parameters**: Modify ``m``, ``l``, or ``g`` and observe the effects
2. **Different Initial Conditions**: Try larger angles or non-zero initial velocity
3. **Add Damping**: Model a damped pendulum with the ``\damping`` command
4. **Double Pendulum**: Add a second mass for chaotic dynamics
5. **Export Results**: Save to CSV or generate C++ code

See the :doc:`tutorials` for more detailed walkthroughs.

Common Pitfalls
---------------

.. warning::

   **Syntax Errors**: Make sure to escape backslashes in Python strings 
   using raw strings (``r''``) or double backslashes (``\\``).

.. warning::

   **Units Matter**: While MechanicsDSL doesn't enforce unit consistency,
   using inconsistent units will give incorrect results.

.. tip::

   **Large Angles**: For large initial angles (> 1 rad), use adaptive
   solvers by setting ``detect_stiff=True`` in ``simulate()``.
