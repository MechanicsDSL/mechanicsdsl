Perturbation Theory
===================

The perturbation module provides tools for analyzing near-integrable systems using perturbation expansions.

Overview
--------

When a system is close to an exactly solvable one, perturbation theory gives approximate solutions. The module implements:

- **Hamiltonian Perturbation**: Expand corrections in small parameter ε
- **Lindstedt-Poincaré Method**: Remove secular terms by expanding frequency
- **Method of Averaging**: Average over fast oscillations
- **Multiple Scale Analysis**: Separate fast and slow dynamics

Theory
------

Perturbation Expansion
~~~~~~~~~~~~~~~~~~~~~~

For Hamiltonian :math:`H = H_0 + \epsilon H_1`:

.. math::

   E &= E_0 + \epsilon E_1 + \epsilon^2 E_2 + \ldots \\
   q(t) &= q_0(t) + \epsilon q_1(t) + \ldots

Secular Terms
~~~~~~~~~~~~~

Naive perturbation often produces terms that grow without bound:

.. math::

   x(t) = A\cos(\omega t) + \epsilon \alpha t \sin(\omega t) + \ldots

The :math:`t \sin(\omega t)` term is **secular**—it grows with time.

Lindstedt-Poincaré Method
~~~~~~~~~~~~~~~~~~~~~~~~~

Expand the frequency to eliminate secular terms:

.. math::

   \omega = \omega_0 + \epsilon \omega_1 + \epsilon^2 \omega_2 + \ldots

Choose :math:`\omega_n` to cancel secular terms at each order.

Usage Examples
--------------

Basic Perturbation Expansion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import PerturbationExpander
   import sympy as sp

   expander = PerturbationExpander()
   
   # Harmonic oscillator with anharmonic perturbation
   p = sp.Symbol('p', real=True)
   x = sp.Symbol('x', real=True)
   m = sp.Symbol('m', positive=True)
   omega = sp.Symbol('omega', positive=True)
   alpha = sp.Symbol('alpha', real=True)
   
   # H₀ = p²/(2m) + (1/2)mω²x²  (harmonic)
   H0 = p**2/(2*m) + sp.Rational(1, 2) * m * omega**2 * x**2
   
   # H₁ = αx³  (cubic anharmonicity)
   H1 = alpha * x**3
   
   # Expand to first order
   result = expander.expand_hamiltonian(H0, H1, ['x'], order=1)
   
   print(f"Unperturbed: {result.unperturbed}")
   print(f"First correction: {result.corrections[0]}")

Averaging Over Fast Angle
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import PerturbationExpander, average_over_angle
   import sympy as sp

   # Average sin²(θ) over θ ∈ [0, 2π]
   theta = sp.Symbol('theta', real=True)
   
   expr = sp.sin(theta)**2
   avg = average_over_angle(expr, 'theta')
   
   print(f"<sin²θ> = {avg}")  # 1/2
   
   # Average cos(θ) → 0
   avg_cos = average_over_angle(sp.cos(theta), 'theta')
   print(f"<cosθ> = {avg_cos}")  # 0

Lindstedt-Poincaré Method
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import PerturbationExpander
   import sympy as sp

   expander = PerturbationExpander()
   
   x = sp.Symbol('x', real=True)
   omega0 = sp.Symbol('omega_0', positive=True)
   epsilon = sp.Symbol('epsilon', positive=True)
   
   # Duffing oscillator: ẍ + ω₀²x + εx³ = 0
   # Naive solution has secular terms
   
   # Use L-P to find frequency correction
   result = expander.lindstedt_poincare(
       equation=x**3,  # Perturbation f(x)
       coordinate='x',
       omega0=omega0,
       epsilon=epsilon,
       order=2
   )
   
   print(f"Corrected frequency: {result['frequency']}")
   # ω = ω₀ + ε·ω₁ + ε²·ω₂

Multiple Scale Analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import MultiScaleAnalysis
   import sympy as sp

   msa = MultiScaleAnalysis(order=2)
   
   t = sp.Symbol('t', real=True)
   epsilon = sp.Symbol('epsilon', positive=True)
   
   # Define time scales
   scales = msa.define_time_scales(t, epsilon)
   
   print("Time scales:")
   for i, T in enumerate(scales):
       print(f"  T_{i} = ε^{i} * t")
   
   # T₀ = t (fast)
   # T₁ = εt (slow)
   # T₂ = ε²t (slower)

Separating Secular and Periodic Terms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import PerturbationExpander
   import sympy as sp

   expander = PerturbationExpander()
   
   t = sp.Symbol('t', real=True)
   omega = sp.Symbol('omega', positive=True)
   
   # Solution with both secular and periodic parts
   solution = sp.cos(omega*t) + t*sp.sin(omega*t) + sp.sin(2*omega*t)
   
   secular, periodic = expander.separate_secular_periodic(solution, t)
   
   print(f"Secular: {secular}")    # t*sin(ωt)
   print(f"Periodic: {periodic}")  # cos(ωt) + sin(2ωt)

Method of Averaging
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mechanics_dsl.domains.classical import AveragingMethod
   import sympy as sp

   avg_method = AveragingMethod()
   
   # For ẋ = εf(x, φ) where φ is fast angle
   phi = sp.Symbol('phi', real=True)
   A = sp.Symbol('A', positive=True)
   
   # Right-hand side with oscillatory terms
   rhs = A * sp.sin(phi)**2 + A * sp.cos(phi)
   
   # Average over fast variable
   averaged = avg_method.average_system(rhs, 'phi', period=2*sp.pi)
   
   print(f"Averaged RHS: {averaged}")  # A/2 (sin² → 1/2, cos → 0)

API Reference
-------------

Classes
~~~~~~~

.. py:class:: PerturbationExpander

   Main class for perturbation expansions.
   
   .. py:method:: expand_hamiltonian(H0, H1, coordinates, order)
   
      Expand Hamiltonian :math:`H = H_0 + \epsilon H_1` to given order.
      
      :returns: PerturbationResult
   
   .. py:method:: lindstedt_poincare(equation, coordinate, omega0, epsilon, order)
   
      Apply Lindstedt-Poincaré method to avoid secular terms.
   
   .. py:method:: average_over_fast_angle(expr, angle)
   
      Compute :math:`\langle f \rangle = (1/2\pi)\int_0^{2\pi} f \, d\theta`.
   
   .. py:method:: separate_secular_periodic(solution, time)
   
      Split solution into secular (growing) and periodic (bounded) parts.

.. py:class:: AveragingMethod

   Method of averaging for oscillatory systems.
   
   .. py:method:: average_system(rhs, fast_var, period)
   
      Average ODE right-hand side over fast variable.
   
   .. py:method:: action_angle_averaging(hamiltonian, action, angle)
   
      Average Hamiltonian at fixed action.

.. py:class:: MultiScaleAnalysis(order)

   Multiple time scale perturbation method.
   
   .. py:method:: define_time_scales(t, epsilon)
   
      Create hierarchy of time scales T₀, T₁, ...
   
   .. py:method:: remove_secular_terms(equations, fast_time)
   
      Extract solvability conditions.

Data Classes
~~~~~~~~~~~~

.. py:class:: PerturbationResult

   Result of perturbation expansion.
   
   .. py:attribute:: order
   
      Order of expansion
   
   .. py:attribute:: unperturbed
   
      Zeroth-order solution/energy
   
   .. py:attribute:: corrections
   
      List of correction terms at each order
   
   .. py:method:: total_solution(epsilon)
   
      Compute full solution to specified order.

See Also
--------

- :doc:`canonical` - Canonical perturbation theory uses action-angle variables
- :doc:`oscillations` - Perturbed oscillators and frequency shifts
