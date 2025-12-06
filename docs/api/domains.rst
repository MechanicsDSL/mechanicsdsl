Domains API Reference
=====================

The ``mechanics_dsl.domains`` package provides domain-specific physics implementations
following a common interface defined by the abstract ``PhysicsDomain`` base class.

.. contents:: Contents
   :local:
   :depth: 2

Base Classes
------------

PhysicsDomain
~~~~~~~~~~~~~

.. py:class:: mechanics_dsl.domains.PhysicsDomain

   Abstract base class that all physics domains must inherit from. Defines the
   common interface for Lagrangian/Hamiltonian derivation and equation solving.

   **Attributes:**

   - ``name``: Human-readable domain name
   - ``coordinates``: List of generalized coordinates
   - ``parameters``: Dictionary of physical parameters
   - ``equations``: Derived equations of motion
   - ``is_compiled``: Whether equations have been derived

   **Abstract Methods:**

   .. py:method:: define_lagrangian() -> sp.Expr
      :abstractmethod:

      Define the Lagrangian L = T - V.

   .. py:method:: define_hamiltonian() -> sp.Expr
      :abstractmethod:

      Define the Hamiltonian H = T + V.

   .. py:method:: derive_equations_of_motion() -> Dict[str, sp.Expr]
      :abstractmethod:

      Derive equations of motion.

   .. py:method:: get_state_variables() -> List[str]
      :abstractmethod:

      Get list of state variables for ODE system.

   **Common Methods:**

   .. py:method:: set_parameter(name: str, value: float) -> None

      Set a single physical parameter.

   .. py:method:: set_parameters(params: Dict[str, float]) -> None

      Set multiple parameters at once.

   .. py:method:: add_coordinate(name: str) -> None

      Add a generalized coordinate.

   .. py:method:: validate_parameters() -> Tuple[bool, List[str]]

      Validate that all required parameters are set.

   .. py:method:: get_conserved_quantities() -> Dict[str, sp.Expr]

      Get expressions for conserved quantities.


Classical Mechanics
-------------------

The ``mechanics_dsl.domains.classical`` package provides implementations for
traditional point-mass and rigid body mechanics.

LagrangianMechanics
~~~~~~~~~~~~~~~~~~~

.. py:class:: mechanics_dsl.domains.classical.LagrangianMechanics

   Lagrangian mechanics using the Euler-Lagrange equations.

   **Example:**

   .. code-block:: python

      from mechanics_dsl.domains.classical import LagrangianMechanics
      import sympy as sp
      
      # Create domain
      domain = LagrangianMechanics("simple_pendulum")
      domain.add_coordinate("theta")
      domain.set_parameters({"m": 1.0, "l": 1.0, "g": 9.81})
      
      # Define energy
      theta = domain.get_symbol("theta")
      theta_dot = domain.get_symbol("theta_dot")
      m, l, g = sp.symbols("m l g", positive=True)
      
      T = sp.Rational(1, 2) * m * l**2 * theta_dot**2
      V = m * g * l * (1 - sp.cos(theta))
      
      domain.set_kinetic_energy(T)
      domain.set_potential_energy(V)
      
      # Derive equations
      equations = domain.derive_equations_of_motion()
      print(equations)

   **Methods:**

   .. py:method:: get_symbol(name: str, **assumptions) -> sp.Symbol

      Get or create a SymPy symbol with caching.

   .. py:method:: set_kinetic_energy(expr: sp.Expr) -> None

      Set the kinetic energy T.

   .. py:method:: set_potential_energy(expr: sp.Expr) -> None

      Set the potential energy V.

   .. py:method:: set_lagrangian(expr: sp.Expr) -> None

      Set the Lagrangian directly (alternative to T and V).

   .. py:method:: derive_equations_of_motion() -> Dict[str, sp.Expr]

      Derive equations using d/dt(∂L/∂q̇) - ∂L/∂q = 0.


HamiltonianMechanics
~~~~~~~~~~~~~~~~~~~~

.. py:class:: mechanics_dsl.domains.classical.HamiltonianMechanics

   Hamiltonian mechanics using Hamilton's equations.

   **Example:**

   .. code-block:: python

      from mechanics_dsl.domains.classical import HamiltonianMechanics
      
      domain = HamiltonianMechanics("harmonic_oscillator")
      domain.add_coordinate("q")
      
      # Define Hamiltonian H = p²/2m + kq²/2
      q = domain.get_symbol("q")
      p = domain.get_symbol("p_q")
      m, k = sp.symbols("m k", positive=True)
      
      H = p**2 / (2*m) + k * q**2 / 2
      domain.set_hamiltonian(H)
      
      # Get Hamilton's equations: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
      equations = domain.derive_equations_of_motion()

   **Methods:**

   .. py:method:: set_hamiltonian(expr: sp.Expr) -> None

      Set the Hamiltonian expression.

   .. py:method:: derive_equations_of_motion() -> Dict[str, sp.Expr]

      Derive Hamilton's equations.


ConstraintHandler
~~~~~~~~~~~~~~~~~

.. py:class:: mechanics_dsl.domains.classical.ConstraintHandler

   Handles mechanical constraints using Lagrange multipliers.

   **Example:**

   .. code-block:: python

      from mechanics_dsl.domains.classical import ConstraintHandler
      import sympy as sp
      
      handler = ConstraintHandler()
      
      # Pendulum constraint: x² + y² = l²
      x, y, l = sp.symbols("x y l")
      constraint = x**2 + y**2 - l**2
      
      lambda_sym = handler.add_holonomic_constraint(constraint)
      
      # Augment Lagrangian
      L_original = ...
      L_augmented = handler.augment_lagrangian(L_original)

   **Methods:**

   .. py:method:: add_holonomic_constraint(constraint: sp.Expr) -> sp.Symbol

      Add constraint g(q) = 0. Returns Lagrange multiplier symbol.

   .. py:method:: add_nonholonomic_constraint(constraint: sp.Expr) -> None

      Add velocity-dependent constraint.

   .. py:method:: augment_lagrangian(lagrangian: sp.Expr) -> sp.Expr

      Create L' = L + Σ(λᵢgᵢ).


RigidBodyDynamics
~~~~~~~~~~~~~~~~~

.. py:class:: mechanics_dsl.domains.classical.RigidBodyDynamics

   Rigid body dynamics with rotational degrees of freedom.

   **Methods:**

   .. py:method:: set_inertia_tensor(I: sp.Matrix) -> None

      Set the 3×3 inertia tensor.


Fluid Dynamics
--------------

The ``mechanics_dsl.domains.fluids`` package provides SPH fluid simulation.

SPHFluid
~~~~~~~~

.. py:class:: mechanics_dsl.domains.fluids.SPHFluid

   Smoothed Particle Hydrodynamics fluid simulation.

   **Example:**

   .. code-block:: python

      from mechanics_dsl.domains.fluids import SPHFluid
      
      # Create fluid with parameters
      fluid = SPHFluid(
          smoothing_length=0.05,
          rest_density=1000.0,
          gas_constant=2000.0,
          viscosity=1.0
      )
      
      # Add particles in a grid
      for i in range(20):
          for j in range(20):
              fluid.add_particle(
                  x=0.1 + i * 0.02,
                  y=0.1 + j * 0.02,
                  mass=0.01
              )
      
      # Simulate
      for _ in range(1000):
          fluid.step(dt=0.001)
      
      # Get positions
      x, y = fluid.get_positions()

   **Attributes:**

   - ``smoothing_length``: SPH kernel width (h)
   - ``rest_density``: Reference density (ρ₀)
   - ``gas_constant``: Stiffness for equation of state
   - ``viscosity``: Dynamic viscosity coefficient
   - ``gravity``: Gravity vector [gx, gy]

   **Methods:**

   .. py:method:: add_particle(x, y, mass=1.0, vx=0, vy=0, particle_type='fluid')

      Add a particle to the simulation.

   .. py:method:: compute_density_pressure() -> None

      Compute density and pressure for all particles.

   .. py:method:: compute_forces() -> List[np.ndarray]

      Compute pressure, viscosity, and gravity forces.

   .. py:method:: step(dt: float) -> None

      Advance simulation by one timestep.

   .. py:method:: get_positions() -> Tuple[np.ndarray, np.ndarray]

      Get (x, y) arrays of particle positions.


BoundaryConditions
~~~~~~~~~~~~~~~~~~

.. py:class:: mechanics_dsl.domains.fluids.BoundaryConditions

   Boundary condition handler for fluid simulations.

   **Example:**

   .. code-block:: python

      from mechanics_dsl.domains.fluids import BoundaryConditions
      
      bc = BoundaryConditions(
          domain_min=(0.0, 0.0),
          domain_max=(1.0, 1.0)
      )
      
      # Add walls
      bc.add_wall(0, 0, 1, 0, wall_type='no_slip')  # Bottom
      bc.add_wall(0, 0, 0, 1, wall_type='no_slip')  # Left
      
      # Generate boundary particles
      boundary_particles = bc.generate_boundary_particles(spacing=0.02)

   **Methods:**

   .. py:method:: add_wall(x1, y1, x2, y2, wall_type='no_slip')

      Add a wall segment. Types: 'no_slip', 'free_slip', 'open'.

   .. py:method:: enforce_box_boundary(position, velocity, restitution=0.5)

      Enforce box boundaries with reflection.

   .. py:method:: enforce_periodic(position) -> np.ndarray

      Apply periodic boundary conditions.

   .. py:method:: generate_boundary_particles(spacing: float) -> List[Dict]

      Generate boundary particles along walls.
