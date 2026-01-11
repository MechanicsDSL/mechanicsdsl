Kinematics
==========

The kinematics module provides tools for analyzing motion without considering forces.
It focuses on analytical solutions with step-by-step work generation, making it ideal
for introductory physics education and problem-solving.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

The kinematics module supports:

- **Constant acceleration kinematics** (1D and 2D)
- **Projectile motion** with complete trajectory analysis
- **Relative motion** between reference frames
- **Symbolic equation derivation** with show-your-work capabilities

Quick Start
-----------

Basic Projectile Motion
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from mechanics_dsl.domains.kinematics import ProjectileMotion

    # Create a projectile with initial conditions
    proj = ProjectileMotion(
        v0=20,          # Initial velocity (m/s)
        angle=45,       # Launch angle (degrees)
        height=10       # Initial height (m)
    )

    # Calculate key quantities
    print(f"Maximum height: {proj.max_height():.2f} m")
    print(f"Range: {proj.range():.2f} m")
    print(f"Time of flight: {proj.time_of_flight():.2f} s")

Kinematic Solver
^^^^^^^^^^^^^^^^

.. code-block:: python

    from mechanics_dsl.domains.kinematics import KinematicsSolver

    # Create solver and solve for unknowns
    solver = KinematicsSolver()
    solution = solver.solve(
        v0=10,      # Initial velocity (m/s)
        a=2,        # Acceleration (m/s^2)
        t=5         # Time (s)
    )

    if solution.success:
        print(f"Final velocity: {solution.state.final_velocity} m/s")
        print(f"Displacement: {solution.state.displacement} m")
        
        # Show step-by-step work
        print(solution.show_work())

Module Reference
----------------

Core Classes
^^^^^^^^^^^^

KinematicsSolver
~~~~~~~~~~~~~~~~

The main solver for 1D kinematics problems using the five kinematic equations.

.. code-block:: python

    from mechanics_dsl.domains.kinematics import KinematicsSolver, KinematicState

    solver = KinematicsSolver()
    
    # Solve with known quantities
    solution = solver.solve(
        v0=0,       # Initial velocity
        a=9.81,     # Acceleration
        t=2         # Time
    )
    
    print(solution.state.final_velocity)  # 19.62 m/s
    print(solution.state.displacement)     # 19.62 m

ProjectileMotion
~~~~~~~~~~~~~~~~

Complete projectile motion analysis with trajectory calculations.

.. code-block:: python

    from mechanics_dsl.domains.kinematics import ProjectileMotion

    proj = ProjectileMotion(v0=25, angle=30, height=0)
    
    # Key quantities
    proj.max_height()           # Maximum height above launch
    proj.range()                # Horizontal distance
    proj.time_of_flight()       # Total time in air
    proj.velocity_at_impact()   # Impact speed
    proj.impact_angle()         # Impact angle (degrees)
    
    # Position at any time
    x, y = proj.position_at(t=1.5)
    vx, vy = proj.velocity_at(t=1.5)

1D Motion Classes
^^^^^^^^^^^^^^^^^

UniformMotion
~~~~~~~~~~~~~

Constant velocity motion (zero acceleration).

.. code-block:: python

    from mechanics_dsl.domains.kinematics import UniformMotion

    motion = UniformMotion(velocity=5, initial_position=0)
    
    motion.position_at(t=10)    # 50 m
    motion.time_to_reach(x=100) # 20 s

UniformlyAcceleratedMotion
~~~~~~~~~~~~~~~~~~~~~~~~~~

Constant acceleration motion.

.. code-block:: python

    from mechanics_dsl.domains.kinematics import UniformlyAcceleratedMotion

    motion = UniformlyAcceleratedMotion(
        initial_velocity=0,
        acceleration=2,
        initial_position=0
    )
    
    motion.position_at(t=5)     # 25 m
    motion.velocity_at(t=5)     # 10 m/s

FreeFall
~~~~~~~~

Specialized class for free-fall motion near Earth's surface.

.. code-block:: python

    from mechanics_dsl.domains.kinematics import FreeFall

    fall = FreeFall(initial_height=100, initial_velocity=0)
    
    fall.time_to_ground()       # ~4.52 s
    fall.velocity_at_ground()   # ~44.3 m/s (downward)

VerticalThrow
~~~~~~~~~~~~~

Upward throw motion with gravity.

.. code-block:: python

    from mechanics_dsl.domains.kinematics import VerticalThrow

    throw = VerticalThrow(
        initial_velocity=20,    # Upward velocity (m/s)
        initial_height=0,
        g=9.81
    )
    
    throw.max_height()          # Maximum height reached
    throw.time_to_max()         # Time to reach max height
    throw.total_time()          # Total time until return

2D Motion
^^^^^^^^^

Vector2D
~~~~~~~~

2D vector class for motion analysis.

.. code-block:: python

    from mechanics_dsl.domains.kinematics import Vector2D

    v = Vector2D(3, 4)
    v.magnitude()   # 5.0
    v.angle()       # 53.13 degrees
    v.unit()        # Vector2D(0.6, 0.8)

Motion2D
~~~~~~~~

General 2D motion with constant acceleration.

.. code-block:: python

    from mechanics_dsl.domains.kinematics import Motion2D, Vector2D

    motion = Motion2D(
        initial_position=Vector2D(0, 0),
        initial_velocity=Vector2D(10, 15),
        acceleration=Vector2D(0, -9.81)
    )
    
    pos = motion.position_at(t=2)
    vel = motion.velocity_at(t=2)

Relative Motion
^^^^^^^^^^^^^^^

ReferenceFrame
~~~~~~~~~~~~~~

Define reference frames for relative motion analysis.

.. code-block:: python

    from mechanics_dsl.domains.kinematics import ReferenceFrame, RelativeMotion

    # Ground frame
    ground = ReferenceFrame(name="ground")
    
    # Train moving at 30 m/s relative to ground
    train = ReferenceFrame(
        name="train",
        velocity_relative_to=ground,
        velocity=Vector2D(30, 0)
    )

RelativeMotion
~~~~~~~~~~~~~~

Calculate velocities between reference frames.

.. code-block:: python

    from mechanics_dsl.domains.kinematics import RelativeMotion

    rel = RelativeMotion()
    
    # Ball thrown at 10 m/s in train frame
    ball_in_train = Vector2D(10, 0)
    
    # Ball velocity in ground frame
    ball_in_ground = rel.transform(ball_in_train, train, ground)
    # Vector2D(40, 0) - adds train velocity

Kinematic Equations
-------------------

The module uses the five standard kinematic equations for constant acceleration:

1. **v = v₀ + at** (velocity-time)
2. **x = x₀ + v₀t + ½at²** (position-time)
3. **v² = v₀² + 2a(x - x₀)** (velocity-position)
4. **x = x₀ + ½(v₀ + v)t** (average velocity)
5. **x = x₀ + vt - ½at²** (final velocity form)

These are implemented in the ``equations`` submodule:

.. code-block:: python

    from mechanics_dsl.domains.kinematics.equations import (
        velocity_time,           # Equation 1
        position_time,           # Equation 2
        velocity_squared,        # Equation 3
        position_average,        # Equation 4
        position_final_velocity  # Equation 5
    )

Show Your Work
--------------

A key feature is the ability to generate step-by-step solutions:

.. code-block:: python

    solver = KinematicsSolver()
    solution = solver.solve(v0=10, a=-9.81, x=0, x0=20)
    
    print(solution.show_work())

Output::

    Problem Setup:
    - Initial position: 20 m
    - Final position: 0 m
    - Initial velocity: 10 m/s
    - Acceleration: -9.81 m/s^2
    
    Step 1: Identify known quantities
    - x0 = 20 m, x = 0 m, v0 = 10 m/s, a = -9.81 m/s^2
    
    Step 2: Select appropriate equation
    Using v^2 = v0^2 + 2a(x - x0)
    
    Step 3: Substitute and solve
    v^2 = (10)^2 + 2(-9.81)(0 - 20)
    v^2 = 100 + 392.4
    v = ±22.19 m/s
    
    Taking negative root (downward motion): v = -22.19 m/s

Examples
--------

Marble from Balcony
^^^^^^^^^^^^^^^^^^^

Classic projectile problem: A marble is launched horizontally from a 10m balcony.

.. code-block:: python

    from mechanics_dsl.domains.kinematics import ProjectileMotion

    # Horizontal launch from 10m height
    marble = ProjectileMotion(v0=5, angle=0, height=10)
    
    print(f"Time of flight: {marble.time_of_flight():.2f} s")
    print(f"Horizontal range: {marble.range():.2f} m")
    print(f"Impact velocity: {marble.velocity_at_impact():.2f} m/s")
    print(f"Impact angle: {marble.impact_angle():.1f} degrees")

Optimal Launch Angle
^^^^^^^^^^^^^^^^^^^^

Find the angle that maximizes range for a given initial velocity:

.. code-block:: python

    from mechanics_dsl.domains.kinematics import ProjectileMotion
    import numpy as np

    v0 = 20  # m/s
    max_range = 0
    best_angle = 0

    for angle in range(0, 91):
        proj = ProjectileMotion(v0=v0, angle=angle, height=0)
        r = proj.range()
        if r > max_range:
            max_range = r
            best_angle = angle

    print(f"Best angle: {best_angle}° with range {max_range:.2f} m")
    # Best angle: 45° with range 40.77 m

See Also
--------

- :doc:`lagrangian_mechanics` - For dynamics problems with forces
- :doc:`oscillations` - For periodic motion
- :doc:`central_forces` - For orbital mechanics
