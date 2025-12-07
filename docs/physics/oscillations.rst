Normal Modes & Oscillations
===========================

The oscillations module provides tools for analyzing small oscillations around equilibrium and computing normal modes.

Overview
--------

For systems near stable equilibrium, small oscillations can be described by coupled linear equations. The module implements:

- **Mass Matrix Extraction**: From kinetic energy :math:`T = \frac{1}{2}\dot{q}^T M \dot{q}`
- **Stiffness Matrix Extraction**: From potential energy :math:`V = \frac{1}{2}q^T K q`
- **Normal Mode Computation**: Solve generalized eigenvalue problem
- **Modal Analysis**: Frequencies, mode shapes, participation factors
- **Coupled Oscillator Systems**: Build and analyze multi-DOF systems

Theory
------

Small Oscillations
~~~~~~~~~~~~~~~~~~

Near a stable equilibrium, the equations of motion become:

.. math::

   M \ddot{q} + K q = 0

where :math:`M` is the mass matrix (symmetric, positive definite) and :math:`K` is the stiffness matrix (symmetric, positive semi-definite for stable equilibrium).

Normal Modes
~~~~~~~~~~~~

Assume harmonic solutions :math:`q(t) = a \cos(\omega t)`. Substituting gives the generalized eigenvalue problem:

.. math::

   K a = \omega^2 M a

The eigenvalues :math:`\omega_n^2` are squared natural frequencies, and eigenvectors :math:`a_n` are mode shapes.

Modal Decomposition
~~~~~~~~~~~~~~~~~~~

Any motion can be expressed as superposition of normal modes:

.. math::

   q(t) = \sum_n c_n a_n \cos(\omega_n t + \phi_n)

Usage Examples
--------------

Extracting Mass and Stiffness Matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import (
       extract_mass_matrix,
       extract_stiffness_matrix
   )
   import sympy as sp

   # Two coupled oscillators
   m = sp.Symbol('m', positive=True)
   k = sp.Symbol('k', positive=True)
   x1 = sp.Symbol('x1', real=True)
   x2 = sp.Symbol('x2', real=True)
   x1_dot = sp.Symbol('x1_dot', real=True)
   x2_dot = sp.Symbol('x2_dot', real=True)
   
   # Kinetic energy
   T = sp.Rational(1, 2) * m * (x1_dot**2 + x2_dot**2)
   
   # Potential energy (springs between wall-m1-m2-wall)
   V = sp.Rational(1, 2) * k * (x1**2 + (x2 - x1)**2 + x2**2)
   
   M = extract_mass_matrix(T, ['x1_dot', 'x2_dot'])
   K = extract_stiffness_matrix(V, ['x1', 'x2'])
   
   print("Mass matrix:")
   print(M)  # [[m, 0], [0, m]]
   
   print("Stiffness matrix:")
   print(K)  # [[2k, -k], [-k, 2k]]

Computing Normal Modes
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import NormalModeAnalyzer, compute_normal_modes
   import numpy as np

   # Numerical mass and stiffness matrices
   m_val = 1.0
   k_val = 1.0
   
   M = np.array([[m_val, 0], [0, m_val]])
   K = np.array([[2*k_val, -k_val], [-k_val, 2*k_val]])
   
   modes = compute_normal_modes(M, K)
   
   for mode in modes:
       print(f"ω = {mode.frequency:.4f}, shape = {mode.shape}")
   
   # Output:
   # ω = 1.0000, shape = [0.707, 0.707]   (in-phase mode)
   # ω = 1.7321, shape = [0.707, -0.707]  (out-of-phase mode)

Full Modal Analysis
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import NormalModeAnalyzer
   import sympy as sp

   analyzer = NormalModeAnalyzer()
   
   # Define Lagrangian
   m = sp.Symbol('m', positive=True)
   k = sp.Symbol('k', positive=True)
   
   x1 = analyzer.get_symbol('x1')
   x2 = analyzer.get_symbol('x2')
   x1_dot = analyzer.get_symbol('x1_dot')
   x2_dot = analyzer.get_symbol('x2_dot')
   
   L = sp.Rational(1, 2) * m * (x1_dot**2 + x2_dot**2) - \
       sp.Rational(1, 2) * k * (x1**2 + (x2 - x1)**2 + x2**2)
   
   result = analyzer.analyze(L, ['x1', 'x2'], parameters={'m': 1.0, 'k': 1.0})
   
   print(f"Frequencies: {result.frequencies}")
   print(f"Mode shapes: {result.mode_shapes}")
   print(f"Mass matrix:\n{result.mass_matrix}")
   print(f"Stiffness matrix:\n{result.stiffness_matrix}")

Coupled Pendulums
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import CoupledOscillatorSystem
   import numpy as np

   # Create coupled pendulum system
   system = CoupledOscillatorSystem(n_dof=2)
   
   # Set masses
   system.set_mass(0, 1.0)
   system.set_mass(1, 1.0)
   
   # Set spring constants
   system.set_spring(0, None, 1.0)    # Wall to mass 0
   system.set_spring(0, 1, 0.2)       # Mass 0 to mass 1 (weak coupling)
   system.set_spring(1, None, 1.0)    # Mass 1 to wall
   
   # Compute modes
   modes = system.compute_modes()
   
   # Beating phenomenon: energy oscillates between pendulums
   # at frequency Δω = ω₂ - ω₁
   delta_omega = modes[1].frequency - modes[0].frequency
   T_beat = 2 * np.pi / delta_omega
   print(f"Beat period: {T_beat:.2f}")

Modal Response to Initial Displacement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import NormalModeAnalyzer
   import numpy as np

   analyzer = NormalModeAnalyzer()
   
   # After computing modes...
   M = np.array([[1, 0], [0, 1]])
   K = np.array([[2, -1], [-1, 2]])
   
   result = analyzer.compute_modes_numerical(M, K)
   
   # Initial displacement: only first mass displaced
   q0 = np.array([1.0, 0.0])
   v0 = np.array([0.0, 0.0])
   
   # Decompose into modal coordinates
   modal_coords = analyzer.modal_decomposition(q0, v0, result)
   
   print(f"Mode 1 amplitude: {modal_coords[0]:.4f}")
   print(f"Mode 2 amplitude: {modal_coords[1]:.4f}")

API Reference
-------------

Classes
~~~~~~~

.. py:class:: NormalModeAnalyzer

   Analyzer for normal modes of oscillating systems.
   
   .. py:method:: analyze(lagrangian, coordinates, parameters)
   
      Full modal analysis from Lagrangian.
      
      :returns: ModalAnalysisResult
   
   .. py:method:: compute_modes_numerical(M, K)
   
      Compute modes from numerical M and K matrices.
      
      :returns: List of NormalMode objects
   
   .. py:method:: modal_decomposition(q0, v0, modes)
   
      Decompose initial conditions into modal coordinates.

.. py:class:: CoupledOscillatorSystem

   Builder for coupled oscillator systems.
   
   .. py:method:: set_mass(index, mass)
   
      Set mass of oscillator at index.
   
   .. py:method:: set_spring(i, j, k)
   
      Set spring constant between masses i and j (or to wall if j=None).
   
   .. py:method:: compute_modes()
   
      Build and solve the eigenvalue problem.

.. py:class:: NormalMode

   Represents a single normal mode.
   
   .. py:attribute:: frequency
   
      Natural frequency ω (rad/s)
   
   .. py:attribute:: shape
   
      Mode shape (eigenvector)
   
   .. py:attribute:: participation_factor
   
      Participation in overall response

.. py:class:: ModalAnalysisResult

   Result of complete modal analysis.
   
   .. py:attribute:: frequencies
   
      List of natural frequencies
   
   .. py:attribute:: mode_shapes
   
      List of mode shape vectors
   
   .. py:attribute:: mass_matrix
   
      Mass matrix M
   
   .. py:attribute:: stiffness_matrix
   
      Stiffness matrix K

Functions
~~~~~~~~~

.. py:function:: extract_mass_matrix(T, velocity_vars)

   Extract mass matrix from kinetic energy expression.

.. py:function:: extract_stiffness_matrix(V, position_vars)

   Extract stiffness matrix from potential energy expression.

.. py:function:: compute_normal_modes(M, K)

   Compute normal modes from M and K matrices.

See Also
--------

- :doc:`stability` - Stability analysis at equilibrium
- :doc:`continuum` - Normal modes of continuous systems (strings, membranes)
