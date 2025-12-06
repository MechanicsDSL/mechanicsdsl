Visualization API Reference
===========================

The ``mechanics_dsl.visualization`` package provides tools for creating
animations, plots, and phase space visualizations of simulation results.

.. contents:: Contents
   :local:
   :depth: 2

Animator
--------

.. py:class:: mechanics_dsl.visualization.Animator

   Animation handler for mechanical system simulations.

   **Example:**

   .. code-block:: python

      from mechanics_dsl.visualization import Animator
      
      animator = Animator(trail_length=200, fps=30)
      
      # Animate pendulum
      anim = animator.animate_pendulum(solution, length=1.0)
      
      # Save to file
      animator.save("pendulum.gif", dpi=100)

   **Methods:**

   .. py:method:: __init__(trail_length: int = None, fps: int = None)

      Initialize animator.

      :param trail_length: Number of trail points (default: config value)
      :param fps: Frames per second (default: config value)

   .. py:method:: setup_figure(xlim, ylim, title) -> Tuple[Figure, Axes]

      Create and configure figure for animation.

   .. py:method:: animate_pendulum(solution, length=1.0, title='Pendulum')

      Create pendulum animation from simulation.

      :param solution: Simulation result dictionary
      :param length: Pendulum length for visualization
      :returns: matplotlib FuncAnimation object

   .. py:method:: animate_particles(positions, title='Particles')

      Animate particle positions over time.

      :param positions: List of (x_array, y_array) for each frame
      :returns: matplotlib FuncAnimation object

   .. py:method:: save(filename: str, dpi: int = 100) -> bool

      Save animation to file (mp4, gif, etc.).


Plotter
-------

.. py:class:: mechanics_dsl.visualization.Plotter

   Plotting utilities for simulation analysis.

   **Example:**

   .. code-block:: python

      from mechanics_dsl.visualization import Plotter
      
      plotter = Plotter()
      
      # Time series plot
      fig = plotter.plot_time_series(solution, variables=['theta'])
      
      # 2D trajectory
      fig = plotter.plot_trajectory_2d(solution, x_var='x', y_var='y')
      
      # Energy conservation
      fig = plotter.plot_energy(solution, kinetic, potential)
      
      plotter.show()

   **Methods:**

   .. py:method:: plot_time_series(solution, variables=None, title='Time Series')

      Plot state variables vs time.

      :param solution: Simulation result
      :param variables: List of variables to plot (default: all)
      :returns: matplotlib Figure

   .. py:method:: plot_trajectory_2d(solution, x_var='x', y_var='y', title='Trajectory')

      Plot 2D trajectory.

      :returns: matplotlib Figure

   .. py:method:: plot_energy(solution, kinetic, potential, title='Energy')

      Plot energy components and conservation error.

      :returns: matplotlib Figure

   .. py:method:: save_figure(fig, filename, dpi=150)

      Save figure to file.

   .. py:staticmethod:: show()

      Display all figures.


PhaseSpaceVisualizer
--------------------

.. py:class:: mechanics_dsl.visualization.PhaseSpaceVisualizer

   Phase space and Poincaré section visualization.

   **Example:**

   .. code-block:: python

      from mechanics_dsl.visualization import PhaseSpaceVisualizer
      
      viz = PhaseSpaceVisualizer()
      
      # Phase portrait (q vs q_dot)
      fig = viz.plot_phase_portrait(solution, coordinate_index=0)
      
      # 3D phase space
      fig = viz.plot_phase_portrait_3d(solution)
      
      # Poincaré section
      fig = viz.plot_poincare_section(solution, section_var=0, section_value=0)

   **Methods:**

   .. py:method:: plot_phase_portrait(solution, coordinate_index=0, title='Phase Portrait')

      Plot 2D phase space trajectory (q vs q̇).

      :param solution: Simulation result
      :param coordinate_index: Which coordinate to plot
      :returns: matplotlib Figure

   .. py:method:: plot_phase_portrait_3d(solution, coords=(0,0,1), title='3D Phase Space')

      Plot 3D phase space trajectory.

      :param coords: Tuple of (coord1_idx, coord1_type, coord2_idx)
      :returns: matplotlib Figure

   .. py:method:: plot_poincare_section(solution, section_var, section_value, plot_vars)

      Plot Poincaré section (stroboscopic map).

      :param section_var: State variable index for section condition
      :param section_value: Value where section is taken
      :param plot_vars: Which variables to plot
      :returns: matplotlib Figure


MechanicsVisualizer (Legacy)
----------------------------

.. py:class:: mechanics_dsl.visualization.MechanicsVisualizer

   Legacy visualization class for backward compatibility. Includes methods
   for pendulum animation, phase space plots, and energy analysis.

   .. note::

      For new code, prefer the modular ``Animator``, ``Plotter``, and
      ``PhaseSpaceVisualizer`` classes.

   **Methods:**

   - ``animate_pendulum()``: Animate pendulum systems
   - ``animate_oscillator()``: Animate harmonic oscillators
   - ``animate_fluid_from_csv()``: Animate SPH particles from CSV
   - ``plot_energy()``: Energy conservation plots
   - ``plot_phase_space()``: Phase portrait plots
   - ``save_animation_to_file()``: Save animations
