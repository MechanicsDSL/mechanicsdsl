"""
Unit tests for MechanicsDSL visualization module.

Tests the MechanicsVisualizer (Animator) class and related visualization functionality.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from mechanics_dsl.visualization import MechanicsVisualizer, Animator, Plotter, PhaseSpaceVisualizer


@pytest.fixture
def visualizer():
    """Create a basic visualizer (Animator) for testing."""
    return MechanicsVisualizer()


@pytest.fixture
def animator():
    """Create an Animator for testing."""
    return Animator()


@pytest.fixture
def simple_solution():
    """Create a simple solution dictionary for testing."""
    return {
        'success': True,
        't': np.linspace(0, 10, 100),
        'y': np.vstack([
            np.sin(np.linspace(0, 10, 100)),
            np.cos(np.linspace(0, 10, 100))
        ]),
        'coordinates': ['theta']
    }


class TestMechanicsVisualizerInit:
    """Tests for MechanicsVisualizer initialization."""

    def test_init_default_values(self):
        """Test visualizer initialization with default values."""
        vis = MechanicsVisualizer()
        assert vis.fig is None
        assert vis.ax is None
        assert vis.animation is None

    def test_init_custom_trail_length(self):
        """Test visualizer with custom trail length."""
        vis = MechanicsVisualizer(trail_length=100)
        assert vis.trail_length == 100

    def test_init_custom_fps(self):
        """Test visualizer with custom FPS."""
        vis = MechanicsVisualizer(fps=30)
        assert vis.fps == 30

    def test_init_both_custom_values(self):
        """Test visualizer with both custom values."""
        vis = MechanicsVisualizer(trail_length=200, fps=60)
        assert vis.trail_length == 200
        assert vis.fps == 60


class TestAnimatorInit:
    """Tests for Animator initialization."""

    def test_animator_creates_instance(self):
        """Test that Animator can be created."""
        anim = Animator()
        assert anim is not None

    def test_animator_has_required_attrs(self):
        """Test animator has required attributes."""
        anim = Animator()
        assert hasattr(anim, 'trail_length')
        assert hasattr(anim, 'fps')
        assert hasattr(anim, 'fig')
        assert hasattr(anim, 'ax')
        assert hasattr(anim, 'animation')


class TestSetupFigure:
    """Tests for Animator.setup_figure method."""

    def test_setup_figure_returns_tuple(self, animator):
        """Test that setup_figure returns fig, ax tuple."""
        result = animator.setup_figure()
        assert isinstance(result, tuple)
        assert len(result) == 2
        plt.close('all')

    def test_setup_figure_sets_fig(self, animator):
        """Test that setup_figure sets self.fig."""
        animator.setup_figure()
        assert animator.fig is not None
        plt.close('all')

    def test_setup_figure_sets_ax(self, animator):
        """Test that setup_figure sets self.ax."""
        animator.setup_figure()
        assert animator.ax is not None
        plt.close('all')

    def test_setup_figure_custom_title(self, animator):
        """Test setup_figure with custom title."""
        animator.setup_figure(title="Custom Title")
        assert animator.fig is not None
        plt.close('all')

    def test_setup_figure_custom_limits(self, animator):
        """Test setup_figure with custom limits."""
        animator.setup_figure(xlim=(-5, 5), ylim=(-3, 3))
        assert animator.ax is not None
        plt.close('all')


class TestAnimatePendulum:
    """Tests for Animator.animate_pendulum method."""

    def test_animate_pendulum_returns_animation(self, animator, simple_solution):
        """Test that animate_pendulum returns an animation object."""
        anim = animator.animate_pendulum(simple_solution, length=1.0)
        assert anim is not None
        plt.close('all')

    def test_animate_pendulum_custom_length(self, animator, simple_solution):
        """Test animate_pendulum with custom length."""
        anim = animator.animate_pendulum(simple_solution, length=2.0)
        assert anim is not None
        plt.close('all')

    def test_animate_pendulum_custom_title(self, animator, simple_solution):
        """Test animate_pendulum with custom title."""
        anim = animator.animate_pendulum(simple_solution, title="My Pendulum")
        assert anim is not None
        plt.close('all')


class TestAnimate:
    """Tests for Animator.animate method."""

    def test_animate_pendulum_system(self, animator, simple_solution):
        """Test animate with pendulum system."""
        params = {'l': 1.0}
        anim = animator.animate(simple_solution, params, system_name="simple_pendulum")
        assert anim is not None
        plt.close('all')

    def test_animate_returns_none_for_failed_solution(self, animator):
        """Test animate returns None for failed solution."""
        anim = animator.animate({'success': False}, {}, "test")
        assert anim is None

    def test_animate_returns_none_for_none_solution(self, animator):
        """Test animate returns None for None solution."""
        anim = animator.animate(None, {}, "test")
        assert anim is None


class TestAnimateParticles:
    """Tests for Animator.animate_particles method."""

    def test_animate_particles_returns_animation(self, animator):
        """Test animate_particles returns an animation."""
        positions = [
            (np.random.rand(10), np.random.rand(10))
            for _ in range(20)
        ]
        anim = animator.animate_particles(positions)
        assert anim is not None
        plt.close('all')

    def test_animate_particles_custom_title(self, animator):
        """Test animate_particles with custom title."""
        positions = [
            (np.array([0.5, 1.0]), np.array([0.5, 1.0]))
            for _ in range(5)
        ]
        anim = animator.animate_particles(positions, title="Fluid Particles")
        assert anim is not None
        plt.close('all')


class TestSaveAnimation:
    """Tests for Animator.save method."""

    def test_save_without_animation_returns_false(self, animator):
        """Test save returns False when no animation exists."""
        result = animator.save("test.gif")
        assert result is False


class TestPlotter:
    """Tests for Plotter class."""

    def test_plotter_creates_instance(self):
        """Test that Plotter can be created."""
        plotter = Plotter()
        assert plotter is not None

    def test_plotter_has_plot_energy(self):
        """Test Plotter has plot_energy method."""
        plotter = Plotter()
        assert hasattr(plotter, 'plot_energy')


class TestPhaseSpaceVisualizer:
    """Tests for PhaseSpaceVisualizer class."""

    def test_phase_space_visualizer_creates_instance(self):
        """Test that PhaseSpaceVisualizer can be created."""
        vis = PhaseSpaceVisualizer()
        assert vis is not None

    def test_phase_space_has_plot_phase_portrait(self):
        """Test PhaseSpaceVisualizer has plot_phase_portrait method."""
        vis = PhaseSpaceVisualizer()
        assert hasattr(vis, 'plot_phase_portrait')
