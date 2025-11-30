Tutorials
=========

These tutorials guide you from basic usage to advanced chaotic simulations.

1. Getting Started
------------------

Let's simulate a simple free particle to understand the workflow.

.. code-block:: python

    from mechanics_dsl import PhysicsCompiler

    dsl_code = """
    \\system{free_particle}
    \\var{x}{Position}{m}
    \\parameter{m}{1.0}{kg}
    \\lagrangian{\\frac{1}{2} * m * \\dot{x}^2}
    \\initial{x=0.0, x_dot=1.0}
    """

    compiler = PhysicsCompiler()
    result = compiler.compile_dsl(dsl_code)
    solution = compiler.simulate(t_span=(0, 10))

2. The Harmonic Oscillator
--------------------------

A mass on a spring is the fundamental building block of mechanics.

.. literalinclude:: ../examples/02_harmonic_oscillator.py
   :language: python
   :linenos:

3. The Double Pendulum (Chaos)
------------------------------

Demonstrating the power of automatic symbolic derivation for complex systems.

.. literalinclude:: ../examples/05_double_pendulum.py
   :language: python
   :linenos:

4. Constrained Systems
----------------------

Using Lagrange multipliers to enforce constraints (e.g., a bead on a wire).

.. code-block:: latex

    \system{rolling_ball}
    ...
    \constraint{x - R * theta}  % Rolling without slipping condition
