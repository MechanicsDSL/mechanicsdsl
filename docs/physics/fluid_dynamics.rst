Fluid Dynamics (SPH)
====================

MechanicsDSL implements a mesh-free Lagrangian fluid solver using **Smoothed Particle Hydrodynamics (SPH)**.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

SPH is a particle-based method for simulating fluid dynamics. Unlike grid-based 
Computational Fluid Dynamics (CFD), SPH discretizes the fluid into particles 
that carry properties (mass, density, pressure). This makes it particularly 
well-suited for:

- Free-surface flows (waves, splashes)
- Multi-phase flows
- Large deformations
- Complex geometries

Quick Start
-----------

Basic Dam Break Simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from mechanics_dsl import PhysicsCompiler
    
    source = r"""
    \system{dam_break}
    
    # Define fluid region (water column)
    \region{fluid_zone}{rectangle}
        \constraint{x}{0, 0.5}
        \constraint{y}{0, 1.0}
    
    # Define boundary (walls)
    \region{walls}{line}
        \constraint{x}{0, 2.0}
        \constraint{y}{0, 0}
    
    # Fluid with water properties
    \fluid{water}{fluid_zone}
        \density{1000}      # kg/m^3
        \viscosity{0.001}   # Pa.s
        \smoothing{0.02}    # SPH kernel radius
    
    # Boundary particles
    \boundary{walls}
    
    # Gravity
    \parameter{g}{9.81}
    """
    
    compiler = PhysicsCompiler()
    result = compiler.compile_dsl(source)
    
    if result['success']:
        # Run SPH simulation
        solution = compiler.simulate_sph(t_span=(0, 2.0), dt=0.0001)
        
        # Export to CSV for visualization
        compiler.export_particles('dam_break.csv')

Governing Equations
-------------------

Density Summation
^^^^^^^^^^^^^^^^^

The density at each particle is computed using kernel interpolation:

**Poly6 Kernel:**

.. math::

    \\rho_i = \\sum_j m_j W(|\\mathbf{r}_{ij}|, h)

Where:
- :math:`m_j` is the mass of neighboring particle j
- :math:`W` is the smoothing kernel function
- :math:`h` is the smoothing length (kernel radius)
- :math:`\\mathbf{r}_{ij} = \\mathbf{r}_i - \\mathbf{r}_j`

**Implementation in MechanicsDSL:**

.. code-block:: python

    from mechanics_dsl.domains.fluids import SPHKernels
    
    # Poly6 kernel (good for density, interpolation)
    W = SPHKernels.poly6(r, h)
    
    # Spiky kernel (good for pressure gradients)
    grad_W = SPHKernels.spiky_gradient(r, h)
    
    # Viscosity kernel (good for viscous forces)
    lap_W = SPHKernels.viscosity_laplacian(r, h)

Momentum Equation
^^^^^^^^^^^^^^^^^

The acceleration of each particle follows the Navier-Stokes equations:

.. math::

    \\frac{d\\mathbf{v}_i}{dt} = -\\frac{1}{\\rho_i}\\nabla P + \\nu \\nabla^2 \\mathbf{v} + \\mathbf{g}

In SPH form (pressure + viscosity + gravity):

.. math::

    \\frac{d\\mathbf{v}_i}{dt} = -\\sum_j m_j \\left( \\frac{P_i}{\\rho_i^2} + \\frac{P_j}{\\rho_j^2} \\right) \\nabla W_{ij} + \\nu \\sum_j m_j \\frac{\\mathbf{v}_{ij}}{\\rho_j} \\nabla^2 W_{ij} + \\mathbf{g}

Equation of State
^^^^^^^^^^^^^^^^^

For weakly compressible SPH (WCSPH), pressure is computed from density using 
the **Tait Equation of State**:

.. math::

    P = B \\left( \\left( \\frac{\\rho}{\\rho_0} \\right)^\\gamma - 1 \\right)

Where:
- :math:`B = c_s^2 \\rho_0 / \\gamma` is the bulk modulus
- :math:`c_s` is the artificial speed of sound (typically 10× max fluid velocity)
- :math:`\\rho_0` is the reference density
- :math:`\\gamma = 7` for water (creates stiff pressure response)

SPH Kernels
-----------

MechanicsDSL implements several standard SPH kernels:

.. list-table:: Available Kernels
   :header-rows: 1
   :widths: 20 40 40

   * - Kernel
     - Best For
     - Properties
   * - Poly6
     - Density, general interpolation
     - Smooth, non-negative
   * - Spiky
     - Pressure forces
     - Non-zero gradient at r=0
   * - Viscosity
     - Viscous forces
     - Always positive Laplacian
   * - Cubic Spline
     - General purpose
     - Good stability

Boundary Handling
-----------------

Several boundary methods are supported:

**Dummy Particles**
    Static particles placed outside the domain that exert repulsive forces.

**Lennard-Jones Repulsion**
    Penalty force that increases sharply near boundaries:
    
    .. math::
    
        \\mathbf{F}_{boundary} = D \\left( \\left(\\frac{r_0}{r}\\right)^{12} - \\left(\\frac{r_0}{r}\\right)^{6} \\right) \\hat{\\mathbf{n}}

**Dynamic Boundary Condition (DBC)**
    Boundary particles are treated like fluid particles but with zero velocity.

Compiler Implementation
-----------------------

When ``\\fluid`` is detected, the compiler switches strategies:

1. **Generates Particles**: Using ``ParticleGenerator`` based on ``\\region`` geometry.
2. **Switches Integrator**: Uses **Velocity Verlet** instead of RK4 for symplectic stability.
3. **Spatial Hashing**: Generates C++ code for an :math:`O(N)` neighbor search grid.

ParticleGenerator
^^^^^^^^^^^^^^^^^

.. code-block:: python

    from mechanics_dsl.compiler_pkg import ParticleGenerator
    from mechanics_dsl.parser import RegionDef
    
    # Define a rectangular fluid region
    region = RegionDef('rectangle', {'x': (0, 1), 'y': (0, 0.5)})
    
    # Generate particles with 0.02m spacing
    particles = ParticleGenerator.generate(region, spacing=0.02)
    
    print(f"Generated {len(particles)} particles")
    # Output: Generated 1250 particles

Code Generation
^^^^^^^^^^^^^^^

SPH simulations can be exported to high-performance C++ code:

.. code-block:: python

    from mechanics_dsl.codegen import CppGenerator
    
    # Generate optimized C++ with OpenMP parallelization
    generator = CppGenerator(
        enable_openmp=True,
        enable_simd=True
    )
    
    cpp_code = generator.generate_sph(
        particles=compiler.fluid_particles,
        boundaries=compiler.boundary_particles,
        parameters={'h': 0.02, 'rho0': 1000, 'c_s': 20}
    )
    
    with open('sph_solver.cpp', 'w') as f:
        f.write(cpp_code)

Visualization
-------------

Animating Fluid Simulations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from mechanics_dsl.visualization import MechanicsVisualizer
    
    viz = MechanicsVisualizer()
    
    # Animate from CSV data (exported particle positions)
    anim = viz.animate_fluid_from_csv(
        'dam_break.csv',
        title='Dam Break Simulation'
    )
    
    # Save as video
    viz.save_animation_to_file(anim, 'dam_break.mp4', fps=30)

The CSV format expected:

.. code-block:: text

    t,id,x,y,rho
    0.000,0,0.01,0.01,1000.0
    0.000,1,0.03,0.01,1000.0
    ...
    0.001,0,0.01,0.012,1001.2
    ...

Performance Tips
----------------

1. **Smoothing Length**: Set :math:`h \\approx 1.2-1.5 \\times` particle spacing
2. **Time Step**: Use CFL condition: :math:`\\Delta t < 0.4 h / c_s`
3. **Particle Count**: Start with 1000-10000 particles for testing
4. **Neighbor Search**: Use spatial hashing for :math:`O(N)` complexity

See Also
--------

- :doc:`continuum` - Continuum mechanics formulation
- :doc:`multiphysics` - Coupled fluid-structure interactions
