```rst
Fluid Dynamics (SPH)
====================

MechanicsDSL supports Lagrangian fluid simulation using **Smoothed Particle Hydrodynamics (SPH)**. Unlike grid-based methods, SPH is mesh-free, making it ideal for free-surface flows like splashing water, dam breaks, and astrophysics.

Syntax Reference
---------------

Defining Fluids
~~~~~~~~~~~~~~~
Use the ``\fluid`` command to define a body of liquid.

.. code-block:: latex

    \fluid{name}
    \region{shape}{constraints}
    \particle_mass{value}
    \equation_of_state{type}

* **name**: Unique identifier for the fluid.
* **region**: Geometric bounds (see below).
* **particle_mass**: Mass of a single SPH particle (determines density resolution).
* **equation_of_state**: Currently supports ``tait`` (for water-like fluids).

Defining Boundaries
~~~~~~~~~~~~~~~~~~~
Use the ``\boundary`` command to define solid obstacles. Boundary particles exert repulsive forces on fluid particles.

.. code-block:: latex

    \boundary{wall_name}
    \region{line}{...}

Geometric Regions
~~~~~~~~~~~~~~~~~
Regions determine where particles are spawned.

* **Rectangle**: Spawns a grid of particles.
    ``\region{rectangle}{x=0.0 .. 1.0, y=0.0 .. 2.0}``

* **Line**: Useful for thin walls.
    ``\region{line}{x=0.0 .. 1.0, y=0.0}``

Parameters
~~~~~~~~~~
The simulation relies on specific global parameters:

* ``\parameter{h}{val}{m}``: **Smoothing Length**. The most critical parameter. Determines the interaction radius of particles. Smaller ``h`` means higher resolution but higher computational cost.

Under the Hood
--------------
The compiler generates a specialized C++ engine implementing:

1.  **Spatial Hashing**: Accelerates neighbor search from O(N²) to O(N).
2.  **Velocity Verlet Integrator**: Provides symplectic stability for long-term energy conservation.
3.  **Tait Equation of State**: Models water as a weakly compressible fluid: :math:`P = B ((\rho/\rho_0)^\gamma - 1)`.
