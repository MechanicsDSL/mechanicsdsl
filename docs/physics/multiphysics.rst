Multiphysics Coupling
=====================

MechanicsDSL supports coupling between different physics domains within a
single simulation, enabling complex scenarios like floating objects in fluids.

Supported Couplings
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Domain A
     - Domain B
     - Applications
   * - Rigid Body
     - SPH Fluid
     - Floating objects, wave makers
   * - Elastic Body
     - SPH Fluid  
     - Flexible structures in flow
   * - Particle System
     - Rigid Body
     - Granular materials, debris

Rigid Body + Fluid Coupling
---------------------------

Simulate a floating box in water:

.. code-block:: python

   source = r'''
   \system{floating_box}
   
   % Fluid setup
   \fluid{water}
   \region{rectangle}{x=0..2, y=0..0.5}
   \parameter{h}{0.03}{m}
   \parameter{rho_0}{1000}{kg/m^3}
   
   % Rigid body (floating box)
   \rigidbody{box}
   \mass{5.0}{kg}
   \geometry{rectangle}{width=0.3, height=0.2}
   \initial_position{1.0, 0.6}
   \initial_velocity{0, 0}
   
   % Boundary container
   \boundary{walls}
   \region{line}{x=-0.1, y=0..1.2}
   \region{line}{x=2.1, y=0..1.2}
   \region{line}{x=-0.1..2.1, y=-0.1}
   '''

Coupling Algorithm
------------------

The two-way coupling uses the following approach:

1. **Fluid → Body**: Sum pressure forces from nearby particles

   .. math::

      \mathbf{F}_{fluid} = -\sum_i m_i \frac{P_i}{\rho_i^2} \nabla W(\mathbf{r} - \mathbf{r}_i)

2. **Body → Fluid**: Apply boundary conditions via ghost particles

3. **Time stepping**: Sub-cycle fluid (smaller dt) relative to rigid body

Force Computation
-----------------

For each rigid body in contact with fluid:

.. code-block:: python

   # Pseudocode for coupling
   for particle in fluid_particles:
       if particle.near(body):
           # Pressure force on body
           F_pressure = compute_pressure_force(particle, body)
           body.apply_force(F_pressure)
           
           # Reaction on particle
           particle.apply_force(-F_pressure)
   
   # Update body with total forces
   body.integrate(dt)

Buoyancy
--------

Archimedes' principle emerges naturally from the SPH pressure forces.
A body will:

- **Float** if body density < fluid density
- **Sink** if body density > fluid density
- **Reach equilibrium** with correct displaced volume

Example: Wave Maker
-------------------

Create waves using a moving rigid body:

.. code-block:: python

   source = r'''
   \system{wave_maker}
   
   \fluid{water}
   \region{rectangle}{x=0.5..3, y=0..0.4}
   
   % Oscillating paddle
   \rigidbody{paddle}
   \geometry{rectangle}{width=0.05, height=0.5}
   \motion{oscillate}{
       center=(0.2, 0.25),
       amplitude=0.1,
       frequency=1.0
   }
   
   \boundary{tank}
   \region{line}{x=0, y=0..0.6}
   \region{line}{x=3.5, y=0..0.6}
   \region{line}{x=0..3.5, y=0}
   '''

Limitations
-----------

Current multiphysics limitations:

1. **2D only**: 3D coupling is experimental
2. **Simple geometries**: Rectangles, circles, convex polygons
3. **No deformation**: Rigid bodies don't deform
4. **Explicit coupling**: May require small time steps for stability

Future Work
-----------

Planned enhancements:

- 3D fluid-structure interaction
- Deformable bodies (FEM)
- Thermal coupling
- Chemical reactions
