Tutorials
=========

This section provides step-by-step tutorials covering various aspects of
MechanicsDSL, from basic mechanics to advanced fluid simulations.

.. contents:: Contents
   :local:
   :depth: 1

Tutorial 1: Harmonic Oscillator
-------------------------------

The simplest mechanical system—a mass on a spring.

**Learning Goals:**

- Basic DSL syntax
- Defining Lagrangians
- Running simulations
- Plotting results

**System Setup:**

A mass m attached to a spring with constant k:

.. math::

   L = \frac{1}{2}m\dot{x}^2 - \frac{1}{2}kx^2

**Implementation:**

.. code-block:: python

   from mechanics_dsl import PhysicsCompiler
   import matplotlib.pyplot as plt
   
   compiler = PhysicsCompiler()
   
   source = r'''
   \system{harmonic_oscillator}
   \defvar{x}{position}{m}
   
   \parameter{m}{1.0}{kg}
   \parameter{k}{10.0}{N/m}
   
   \lagrangian{\frac{1}{2} m \dot{x}^2 - \frac{1}{2} k x^2}
   
   \initial{x=1.0, x_dot=0}
   '''
   
   result = compiler.compile(source)
   solution = compiler.simulate((0, 10))
   
   # Plot
   plt.figure(figsize=(10, 4))
   plt.plot(solution['t'], solution['y'][0], label='Position')
   plt.plot(solution['t'], solution['y'][1], label='Velocity')
   plt.xlabel('Time (s)')
   plt.legend()
   plt.title('Simple Harmonic Oscillator')
   plt.grid(True, alpha=0.3)
   plt.show()

**Expected Output:**

Sinusoidal oscillation with period T = 2π√(m/k) ≈ 1.99 s.


Tutorial 2: Simple Pendulum
---------------------------

A classic physics problem with nonlinear dynamics.

**Learning Goals:**

- Trigonometric functions in Lagrangians
- Phase space visualization
- Energy conservation

**Implementation:**

.. code-block:: python

   source = r'''
   \system{simple_pendulum}
   \defvar{theta}{angle}{rad}
   
   \parameter{m}{1.0}{kg}
   \parameter{l}{1.0}{m}
   \parameter{g}{9.81}{m/s^2}
   
   % L = T - V = (1/2)ml²θ̇² - mgl(1-cos(θ))
   \lagrangian{
       \frac{1}{2} m l^2 \dot{theta}^2 - m g l (1 - \cos(theta))
   }
   
   \initial{theta=2.0, theta_dot=0}  % Large angle
   '''
   
   result = compiler.compile(source)
   solution = compiler.simulate((0, 20))
   
   # Phase space plot
   plt.figure(figsize=(8, 8))
   plt.plot(solution['y'][0], solution['y'][1])
   plt.xlabel('θ (rad)')
   plt.ylabel('θ̇ (rad/s)')
   plt.title('Pendulum Phase Space')
   plt.grid(True, alpha=0.3)
   plt.show()

**Exploration:**

- Try different initial angles
- Near π (inverted position) the dynamics become interesting
- Compare small-angle approximation to exact solution


Tutorial 3: Double Pendulum
---------------------------

A chaotic system demonstrating sensitive dependence on initial conditions.

**Learning Goals:**

- Multiple coordinates
- Complex Lagrangians
- Chaotic dynamics
- Animation

**Implementation:**

.. code-block:: python

   source = r'''
   \system{double_pendulum}
   
   \defvar{theta1}{angle}{rad}
   \defvar{theta2}{angle}{rad}
   
   \parameter{m1}{1.0}{kg}
   \parameter{m2}{1.0}{kg}
   \parameter{l1}{1.0}{m}
   \parameter{l2}{1.0}{m}
   \parameter{g}{9.81}{m/s^2}
   
   \lagrangian{
       \frac{1}{2} (m1 + m2) l1^2 \dot{theta1}^2 +
       \frac{1}{2} m2 l2^2 \dot{theta2}^2 +
       m2 l1 l2 \dot{theta1} \dot{theta2} \cos(theta1 - theta2) +
       (m1 + m2) g l1 \cos(theta1) +
       m2 g l2 \cos(theta2)
   }
   
   \initial{theta1=2.0, theta1_dot=0, theta2=2.0, theta2_dot=0}
   '''
   
   result = compiler.compile(source)
   solution = compiler.simulate((0, 20), num_points=2000)
   
   # Create animation
   compiler.visualize(solution)
   plt.show()

**Chaos Demonstration:**

Try initial conditions differing by 0.001 rad and observe divergence:

.. code-block:: python

   # Run two simulations with tiny difference
   ic1 = {'theta1': 2.000, 'theta2': 2.0}
   ic2 = {'theta1': 2.001, 'theta2': 2.0}
   
   # Compare trajectories after ~10 seconds


Tutorial 4: Damped Driven Pendulum
----------------------------------

Combining dissipation and periodic forcing.

**Learning Goals:**

- Non-conservative forces
- Damping
- Time-dependent forcing
- Limit cycles

**Implementation:**

.. code-block:: python

   source = r'''
   \system{damped_driven_pendulum}
   \defvar{theta}{angle}{rad}
   
   \parameter{m}{1.0}{kg}
   \parameter{l}{1.0}{m}
   \parameter{g}{9.81}{m/s^2}
   \parameter{gamma}{0.5}{1/s}
   \parameter{F0}{1.5}{N.m}
   \parameter{omega_d}{0.667}{rad/s}
   
   \lagrangian{\frac{1}{2} m l^2 \dot{theta}^2 - m g l (1 - \cos(theta))}
   
   \damping{gamma}
   \force{theta}{F0 \cos(omega_d t)}
   
   \initial{theta=0.1, theta_dot=0}
   '''
   
   result = compiler.compile(source)
   solution = compiler.simulate((0, 100), num_points=5000)

**Exploration:**

- Vary F0/omega_d to find chaotic regime
- Create bifurcation diagrams
- Analyze Poincaré sections


Tutorial 5: N-Body Gravitational System
---------------------------------------

Simulate gravitational interactions between multiple bodies.

**Learning Goals:**

- Multiple interacting particles
- Conservation laws
- Orbital mechanics

**The Figure-8 Orbit:**

A remarkable periodic solution to the 3-body problem:

.. code-block:: python

   source = r'''
   \system{figure8_orbit}
   
   \defvar{x1}{position}{m}
   \defvar{y1}{position}{m}
   \defvar{x2}{position}{m}
   \defvar{y2}{position}{m}
   \defvar{x3}{position}{m}
   \defvar{y3}{position}{m}
   
   \parameter{m}{1.0}{kg}
   \parameter{G}{1.0}{N.m^2/kg^2}
   
   % Kinetic energy
   \define{\op{T} = \frac{1}{2} m (\dot{x1}^2 + \dot{y1}^2 + 
                                   \dot{x2}^2 + \dot{y2}^2 + 
                                   \dot{x3}^2 + \dot{y3}^2)}
   
   % Gravitational potential
   \define{\op{V} = -G m^2 (
       1/\sqrt{(x2-x1)^2 + (y2-y1)^2} +
       1/\sqrt{(x3-x1)^2 + (y3-y1)^2} +
       1/\sqrt{(x3-x2)^2 + (y3-y2)^2}
   )}
   
   \lagrangian{\op{T} - \op{V}}
   
   % Special initial conditions for figure-8
   \initial{
       x1=0.97000436, y1=-0.24308753,
       x2=-0.97000436, y2=0.24308753,
       x3=0, y3=0,
       x1_dot=0.4662036850, y1_dot=0.4323657300,
       x2_dot=0.4662036850, y2_dot=0.4323657300,
       x3_dot=-0.93240737, y3_dot=-0.86473146
   }
   '''


Tutorial 6: SPH Fluid Simulation
--------------------------------

Create a dam break simulation using Smoothed Particle Hydrodynamics.

**Learning Goals:**

- Fluid definition syntax
- Boundary conditions
- Particle-based simulation
- Visualization of fluids

**Implementation:**

.. code-block:: python

   source = r'''
   \system{dam_break}
   
   \fluid{water}
   \region{0, 0, 0.3, 0.5}  % Water column
   \parameter{h}{0.03}{m}
   \parameter{rho_0}{1000}{kg/m^3}
   \parameter{mu}{0.001}{Pa.s}
   
   \boundary{container}
   \region{-0.1, -0.1, 1.1, 0}    % Bottom
   \region{-0.1, -0.1, 0, 0.6}    % Left
   \region{1.0, -0.1, 1.1, 0.6}   % Right
   '''
   
   result = compiler.compile(source)
   
   # Run SPH simulation
   solution = compiler.simulate_fluid((0, 2.0))
   
   # Animate
   compiler.visualize_fluid(solution)


Tutorial 7: Constrained Systems
-------------------------------

Handle systems with holonomic constraints.

**Bead on a Wire:**

A bead constrained to move on a parabolic wire:

.. code-block:: python

   source = r'''
   \system{bead_on_wire}
   
   \defvar{x}{position}{m}
   \defvar{y}{position}{m}
   
   \parameter{m}{0.1}{kg}
   \parameter{g}{9.81}{m/s^2}
   \parameter{a}{1.0}{1/m}  % Parabola parameter
   
   \lagrangian{
       \frac{1}{2} m (\dot{x}^2 + \dot{y}^2) - m g y
   }
   
   % Constraint: y = ax²
   \constraint{y - a x^2}
   
   \initial{x=1.0, x_dot=0, y=1.0, y_dot=0}
   '''


Tutorial 8: Hamiltonian Formulation
-----------------------------------

Use Hamilton's equations instead of Lagrange's.

**Learning Goals:**

- Phase space (q, p) coordinates
- Hamilton's equations
- Symplectic integrators

**Implementation:**

.. code-block:: python

   source = r'''
   \system{harmonic_hamiltonian}
   
   \defvar{q}{position}{m}
   
   \parameter{m}{1.0}{kg}
   \parameter{k}{4.0}{N/m}
   
   % H = p²/2m + kq²/2
   \hamiltonian{\frac{p_q^2}{2 m} + \frac{1}{2} k q^2}
   
   \initial{q=1.0, p_q=0}
   '''
   
   result = compiler.compile(source)
   solution = compiler.simulate((0, 10))
   
   # Plot in phase space (q, p)
   plt.plot(solution['y'][0], solution['y'][1])


Next Steps
----------

After completing these tutorials, explore:

- :doc:`advanced` - Advanced techniques and optimizations
- :doc:`api/core` - Full API reference
- :doc:`codegen/overview` - Code generation for deployment
