"""
Tests for MechanicsDSL fluids domain.

Tests SPHFluid and BoundaryConditions classes for fluid simulation.
"""

import pytest
import numpy as np

from mechanics_dsl.domains.fluids import SPHFluid, BoundaryConditions


class TestSPHFluidInit:
    """Tests for SPHFluid initialization."""

    def test_init_default_values(self):
        """Test SPHFluid initialization with default values."""
        fluid = SPHFluid()
        assert fluid.smoothing_length == 0.1
        assert fluid.rest_density == 1000.0
        assert fluid.gas_constant == 2000.0
        assert fluid.viscosity == 1.0

    def test_init_custom_smoothing_length(self):
        """Test SPHFluid with custom smoothing length."""
        fluid = SPHFluid(smoothing_length=0.05)
        assert fluid.smoothing_length == 0.05

    def test_init_custom_rest_density(self):
        """Test SPHFluid with custom rest density."""
        fluid = SPHFluid(rest_density=500.0)
        assert fluid.rest_density == 500.0

    def test_init_custom_gas_constant(self):
        """Test SPHFluid with custom gas constant."""
        fluid = SPHFluid(gas_constant=3000.0)
        assert fluid.gas_constant == 3000.0

    def test_init_custom_viscosity(self):
        """Test SPHFluid with custom viscosity."""
        fluid = SPHFluid(viscosity=0.5)
        assert fluid.viscosity == 0.5

    def test_init_empty_particles(self):
        """Test that fluid starts with no particles."""
        fluid = SPHFluid()
        assert len(fluid.particles) == 0

    def test_init_empty_boundary_particles(self):
        """Test that fluid starts with no boundary particles."""
        fluid = SPHFluid()
        assert len(fluid.boundary_particles) == 0

    def test_init_gravity(self):
        """Test that gravity is set correctly."""
        fluid = SPHFluid()
        np.testing.assert_array_almost_equal(fluid.gravity, [0.0, -9.81])


class TestSPHFluidAddParticle:
    """Tests for SPHFluid.add_particle method."""

    def test_add_single_particle(self):
        """Test adding a single particle."""
        fluid = SPHFluid()
        fluid.add_particle(0.5, 0.5)
        assert len(fluid.particles) == 1

    def test_add_multiple_particles(self):
        """Test adding multiple particles."""
        fluid = SPHFluid()
        fluid.add_particle(0.1, 0.1)
        fluid.add_particle(0.2, 0.2)
        fluid.add_particle(0.3, 0.3)
        assert len(fluid.particles) == 3

    def test_add_particle_position(self):
        """Test that particle position is set correctly."""
        fluid = SPHFluid()
        fluid.add_particle(0.7, 0.3)
        assert fluid.particles[0]['x'] == 0.7
        assert fluid.particles[0]['y'] == 0.3

    def test_add_particle_velocity(self):
        """Test adding particle with custom velocity."""
        fluid = SPHFluid()
        fluid.add_particle(0.5, 0.5, vx=1.0, vy=-0.5)
        assert fluid.particles[0]['vx'] == 1.0
        assert fluid.particles[0]['vy'] == -0.5

    def test_add_particle_mass(self):
        """Test adding particle with custom mass."""
        fluid = SPHFluid()
        fluid.add_particle(0.5, 0.5, mass=2.5)
        assert fluid.particles[0]['mass'] == 2.5

    def test_add_boundary_particle(self):
        """Test adding boundary particle."""
        fluid = SPHFluid()
        fluid.add_particle(0.0, 0.0, particle_type='boundary')
        assert len(fluid.boundary_particles) == 1
        assert len(fluid.particles) == 0


class TestSPHFluidKernels:
    """Tests for SPH kernel functions."""

    def test_kernel_poly6_at_origin(self):
        """Test poly6 kernel at r=0."""
        fluid = SPHFluid(smoothing_length=0.1)
        value = fluid.kernel_poly6(0.0, 0.1)
        assert value > 0

    def test_kernel_poly6_beyond_h(self):
        """Test poly6 kernel beyond smoothing length."""
        fluid = SPHFluid(smoothing_length=0.1)
        value = fluid.kernel_poly6(0.2, 0.1)
        assert value == 0.0

    def test_kernel_spiky_at_origin(self):
        """Test spiky gradient at very small r."""
        fluid = SPHFluid(smoothing_length=0.1)
        grad = fluid.kernel_spiky_grad(np.array([0.0, 0.0]), 0.1)
        np.testing.assert_array_almost_equal(grad, [0.0, 0.0])

    def test_kernel_viscosity_beyond_h(self):
        """Test viscosity laplacian beyond smoothing length."""
        fluid = SPHFluid(smoothing_length=0.1)
        value = fluid.kernel_viscosity_laplacian(0.2, 0.1)
        assert value == 0.0


class TestSPHFluidSimulation:
    """Tests for SPH simulation methods."""

    def test_compute_density_pressure(self):
        """Test density and pressure computation."""
        fluid = SPHFluid(smoothing_length=0.1)
        fluid.add_particle(0.5, 0.5)
        fluid.add_particle(0.55, 0.5)
        fluid.compute_density_pressure()
        # Particles should have non-zero density
        assert fluid.particles[0]['density'] > 0

    def test_compute_forces(self):
        """Test force computation."""
        fluid = SPHFluid(smoothing_length=0.1)
        fluid.add_particle(0.5, 0.5)
        fluid.compute_density_pressure()
        forces = fluid.compute_forces()
        assert len(forces) == 1
        assert isinstance(forces[0], np.ndarray)

    def test_step_updates_position(self):
        """Test that step updates particle positions."""
        fluid = SPHFluid(smoothing_length=0.1)
        fluid.add_particle(0.5, 0.5, vy=1.0)
        initial_y = fluid.particles[0]['y']
        fluid.step(0.01)
        # Position should change due to velocity
        assert fluid.particles[0]['y'] != initial_y

    def test_get_positions(self):
        """Test getting particle positions."""
        fluid = SPHFluid()
        fluid.add_particle(0.1, 0.2)
        fluid.add_particle(0.3, 0.4)
        x, y = fluid.get_positions()
        np.testing.assert_array_almost_equal(x, [0.1, 0.3])
        np.testing.assert_array_almost_equal(y, [0.2, 0.4])

    def test_get_positions_empty(self):
        """Test getting positions with no particles."""
        fluid = SPHFluid()
        x, y = fluid.get_positions()
        assert len(x) == 0
        assert len(y) == 0


class TestBoundaryConditionsInit:
    """Tests for BoundaryConditions initialization."""

    def test_init_default_domain(self):
        """Test BoundaryConditions with default domain."""
        bc = BoundaryConditions()
        np.testing.assert_array_almost_equal(bc.domain_min, [0.0, 0.0])
        np.testing.assert_array_almost_equal(bc.domain_max, [1.0, 1.0])

    def test_init_custom_domain(self):
        """Test BoundaryConditions with custom domain."""
        bc = BoundaryConditions(domain_min=(-1.0, -1.0), domain_max=(2.0, 2.0))
        np.testing.assert_array_almost_equal(bc.domain_min, [-1.0, -1.0])
        np.testing.assert_array_almost_equal(bc.domain_max, [2.0, 2.0])

    def test_init_empty_walls(self):
        """Test that BC starts with no walls."""
        bc = BoundaryConditions()
        assert len(bc.walls) == 0


class TestBoundaryConditionsWalls:
    """Tests for BoundaryConditions wall methods."""

    def test_add_wall(self):
        """Test adding a wall."""
        bc = BoundaryConditions()
        bc.add_wall(0.0, 0.0, 1.0, 0.0)
        assert len(bc.walls) == 1

    def test_add_multiple_walls(self):
        """Test adding multiple walls."""
        bc = BoundaryConditions()
        bc.add_wall(0.0, 0.0, 1.0, 0.0)  # Bottom
        bc.add_wall(0.0, 0.0, 0.0, 1.0)  # Left
        bc.add_wall(1.0, 0.0, 1.0, 1.0)  # Right
        bc.add_wall(0.0, 1.0, 1.0, 1.0)  # Top
        assert len(bc.walls) == 4

    def test_wall_has_normal(self):
        """Test that wall has computed normal."""
        bc = BoundaryConditions()
        bc.add_wall(0.0, 0.0, 1.0, 0.0)
        assert 'normal' in bc.walls[0]
        assert len(bc.walls[0]['normal']) == 2


class TestBoundaryConditionsEnforce:
    """Tests for boundary enforcement methods."""

    def test_enforce_box_left_boundary(self):
        """Test enforcement of left boundary."""
        bc = BoundaryConditions()
        pos = np.array([-0.1, 0.5])
        vel = np.array([-1.0, 0.0])
        new_pos, new_vel = bc.enforce_box_boundary(pos, vel)
        assert new_pos[0] >= 0.0
        assert new_vel[0] > 0  # Reflected

    def test_enforce_box_right_boundary(self):
        """Test enforcement of right boundary."""
        bc = BoundaryConditions()
        pos = np.array([1.1, 0.5])
        vel = np.array([1.0, 0.0])
        new_pos, new_vel = bc.enforce_box_boundary(pos, vel)
        assert new_pos[0] <= 1.0
        assert new_vel[0] < 0  # Reflected

    def test_enforce_box_bottom_boundary(self):
        """Test enforcement of bottom boundary."""
        bc = BoundaryConditions()
        pos = np.array([0.5, -0.1])
        vel = np.array([0.0, -1.0])
        new_pos, new_vel = bc.enforce_box_boundary(pos, vel)
        assert new_pos[1] >= 0.0
        assert new_vel[1] > 0  # Reflected

    def test_enforce_box_restitution(self):
        """Test restitution coefficient."""
        bc = BoundaryConditions()
        pos = np.array([-0.1, 0.5])
        vel = np.array([-2.0, 0.0])
        new_pos, new_vel = bc.enforce_box_boundary(pos, vel, restitution=0.5)
        assert new_vel[0] == pytest.approx(1.0)  # 0.5 * 2.0

    def test_enforce_periodic_wrap_left(self):
        """Test periodic BC wrapping from left."""
        bc = BoundaryConditions()
        pos = np.array([-0.1, 0.5])
        new_pos = bc.enforce_periodic(pos)
        assert new_pos[0] >= 0.0
        assert new_pos[0] <= 1.0

    def test_enforce_periodic_wrap_right(self):
        """Test periodic BC wrapping from right."""
        bc = BoundaryConditions()
        pos = np.array([1.1, 0.5])
        new_pos = bc.enforce_periodic(pos)
        assert new_pos[0] >= 0.0
        assert new_pos[0] <= 1.0


class TestBoundaryParticleGeneration:
    """Tests for boundary particle generation."""

    def test_generate_boundary_particles(self):
        """Test generating boundary particles."""
        bc = BoundaryConditions()
        bc.add_wall(0.0, 0.0, 1.0, 0.0)
        particles = bc.generate_boundary_particles(spacing=0.1)
        assert len(particles) > 0

    def test_boundary_particles_have_positions(self):
        """Test that boundary particles have position data."""
        bc = BoundaryConditions()
        bc.add_wall(0.0, 0.0, 1.0, 0.0)
        particles = bc.generate_boundary_particles(spacing=0.1)
        for p in particles:
            assert 'x' in p
            assert 'y' in p

    def test_boundary_particles_type(self):
        """Test that boundary particles have correct type."""
        bc = BoundaryConditions()
        bc.add_wall(0.0, 0.0, 1.0, 0.0)
        particles = bc.generate_boundary_particles(spacing=0.1)
        for p in particles:
            assert p['type'] == 'boundary'
