"""
Comprehensive test suite for MechanicsVisualizer (uses Animator class)
Target: 95%+ code coverage on Codecov
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
import os
import tempfile
from unittest.mock import patch

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from mechanics_dsl.utils import config

# Import the module under test
from mechanics_dsl.visualization import Animator, MechanicsVisualizer, PhaseSpaceVisualizer, Plotter

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def visualizer():
    """Create a visualizer instance for testing"""
    return MechanicsVisualizer(trail_length=50, fps=30)


@pytest.fixture
def animator():
    """Create an animator instance for testing"""
    return Animator(trail_length=50, fps=30)


@pytest.fixture
def plotter():
    """Create a plotter instance for testing"""
    return Plotter()


@pytest.fixture
def phase_viz():
    """Create a phase space visualizer instance for testing"""
    return PhaseSpaceVisualizer()


@pytest.fixture
def basic_solution():
    """Create a basic valid solution dictionary"""
    t = np.linspace(0, 10, 100)
    theta = np.sin(t)
    theta_dot = np.cos(t)
    return {"success": True, "t": t, "y": np.array([theta, theta_dot]), "coordinates": ["theta"]}


@pytest.fixture
def oscillator_solution():
    """Create an oscillator solution"""
    t = np.linspace(0, 10, 100)
    x = np.sin(t)
    v = np.cos(t)
    return {"success": True, "t": t, "y": np.array([x, v]), "coordinates": ["x"]}


@pytest.fixture
def failed_solution():
    """Create a failed solution"""
    return {"success": False, "message": "Simulation failed"}


@pytest.fixture
def basic_parameters():
    """Basic parameter dictionary"""
    return {"l": 1.0, "m": 1.0, "g": 9.81, "k": 1.0}


@pytest.fixture
def cleanup_plots():
    """Cleanup matplotlib figures after each test"""
    yield
    plt.close("all")


# ============================================================================
# TEST ANIMATOR INITIALIZATION
# ============================================================================


class TestAnimatorInit:
    """Test Animator initialization"""

    def test_init_default(self):
        """Test default initialization"""
        anim = Animator()
        assert anim.trail_length == config.trail_length
        assert anim.fps == config.animation_fps
        assert anim.fig is None
        assert anim.ax is None
        assert anim.animation is None

    def test_init_custom(self):
        """Test custom initialization"""
        anim = Animator(trail_length=100, fps=60)
        assert anim.trail_length == 100
        assert anim.fps == 60

    def test_init_partial_custom(self):
        """Test partial custom initialization"""
        anim = Animator(trail_length=75)
        assert anim.trail_length == 75
        assert anim.fps == config.animation_fps


# ============================================================================
# TEST SETUP_FIGURE
# ============================================================================


class TestSetupFigure:
    """Test figure setup"""

    def test_setup_figure_default(self, animator, cleanup_plots):
        """Test default figure setup"""
        fig, ax = animator.setup_figure()
        assert animator.fig is not None
        assert animator.ax is not None
        assert fig is animator.fig
        assert ax is animator.ax

    def test_setup_figure_custom(self, animator, cleanup_plots):
        """Test custom figure setup"""
        fig, ax = animator.setup_figure(xlim=(-5, 5), ylim=(-3, 3), title="Custom Title")
        assert animator.fig is not None
        assert ax.get_title() == "Custom Title"


# ============================================================================
# TEST ANIMATE_PENDULUM
# ============================================================================


class TestAnimatePendulum:
    """Test pendulum animation"""

    def test_animate_pendulum_success(self, animator, basic_solution, cleanup_plots):
        """Test successful pendulum animation"""
        anim = animator.animate_pendulum(basic_solution, length=1.0)
        assert anim is not None
        assert animator.fig is not None
        assert animator.ax is not None
        assert isinstance(anim, animation.FuncAnimation)

    def test_animate_pendulum_custom_length(self, animator, basic_solution, cleanup_plots):
        """Test pendulum animation with custom length"""
        anim = animator.animate_pendulum(basic_solution, length=2.0)
        assert anim is not None

    def test_animate_pendulum_custom_title(self, animator, basic_solution, cleanup_plots):
        """Test pendulum animation with custom title"""
        anim = animator.animate_pendulum(basic_solution, length=1.0, title="My Pendulum")
        assert anim is not None
        assert animator.ax.get_title() == "My Pendulum"


# ============================================================================
# TEST ANIMATE (GENERIC DISPATCHER)
# ============================================================================


class TestAnimate:
    """Test generic animation dispatcher"""

    def test_animate_none_solution(self, animator, basic_parameters, cleanup_plots):
        """Test with None solution"""
        result = animator.animate(None, basic_parameters)
        assert result is None

    def test_animate_failed_solution(
        self, animator, failed_solution, basic_parameters, cleanup_plots
    ):
        """Test with failed solution"""
        result = animator.animate(failed_solution, basic_parameters)
        assert result is None

    def test_animate_pendulum_by_name(
        self, animator, basic_solution, basic_parameters, cleanup_plots
    ):
        """Test pendulum animation by system name"""
        anim = animator.animate(basic_solution, basic_parameters, "pendulum")
        assert anim is not None

    def test_animate_pendulum_by_coordinate(
        self, animator, basic_solution, basic_parameters, cleanup_plots
    ):
        """Test pendulum animation by coordinate name (theta)"""
        anim = animator.animate(basic_solution, basic_parameters, "system")
        assert anim is not None

    def test_animate_with_custom_parameters(self, animator, basic_solution, cleanup_plots):
        """Test animation with custom parameters"""
        params = {"l": 2.5}
        anim = animator.animate(basic_solution, params, "test_system")
        assert anim is not None


# ============================================================================
# TEST ANIMATE_PARTICLES
# ============================================================================


class TestAnimateParticles:
    """Test particle animation"""

    def test_animate_particles_success(self, animator, cleanup_plots):
        """Test successful particle animation"""
        positions = [(np.random.rand(10), np.random.rand(10)) for _ in range(20)]
        anim = animator.animate_particles(positions)
        assert anim is not None
        assert isinstance(anim, animation.FuncAnimation)

    def test_animate_particles_custom_title(self, animator, cleanup_plots):
        """Test particle animation with custom title"""
        positions = [(np.random.rand(5), np.random.rand(5)) for _ in range(10)]
        anim = animator.animate_particles(positions, title="My Particles")
        assert anim is not None


# ============================================================================
# TEST SAVE
# ============================================================================


class TestSave:
    """Test animation saving"""

    def test_save_no_animation(self, animator, cleanup_plots):
        """Test save with no animation"""
        result = animator.save("test.gif")
        assert result is False

    def test_save_gif_success(self, animator, basic_solution, cleanup_plots):
        """Test successful GIF save"""
        animator.animate_pendulum(basic_solution, length=1.0)

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = animator.save(tmp_path)
            assert result is True
            assert os.path.exists(tmp_path)
        finally:
            plt.close("all")  # Close figures to release file handles
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except PermissionError:
                    pass  # Ignore on Windows if still locked

    def test_save_with_error(self, animator, basic_solution, cleanup_plots):
        """Test save with error"""
        animator.animate_pendulum(basic_solution, length=1.0)

        with patch.object(animator.animation, "save", side_effect=Exception("Save error")):
            result = animator.save("test.gif")
            assert result is False


# ============================================================================
# TEST MECHANICS VISUALIZER (BACKWARD COMPAT)
# ============================================================================


class TestMechanicsVisualizer:
    """Test MechanicsVisualizer from visualization.py (different API from Animator)"""

    def test_init(self):
        """Test initialization"""
        viz = MechanicsVisualizer()
        assert viz is not None
        assert viz.trail_length == config.trail_length

    def test_is_visualizer(self):
        """Test that MechanicsVisualizer has expected visualization methods"""
        viz = MechanicsVisualizer(trail_length=50, fps=30)
        assert hasattr(viz, "animate_pendulum")
        assert hasattr(viz, "animate")
        assert hasattr(
            viz, "setup_3d_plot"
        )  # Note: MechanicsVisualizer uses setup_3d_plot, not setup_figure
        assert hasattr(viz, "has_ffmpeg")

    def test_animate_pendulum(self, visualizer, basic_solution, cleanup_plots):
        """Test pendulum animation using correct API with parameters dict"""
        parameters = {"l": 1.0, "m": 1.0, "g": 9.81}
        visualizer.animate_pendulum(basic_solution, parameters)
        # Animation may be None if simulation data doesn't match expectations
        # but it shouldn't raise an error


# ============================================================================
# TEST PLOTTER
# ============================================================================


class TestPlotter:
    """Test Plotter class"""

    def test_init(self):
        """Test initialization"""
        plotter = Plotter()
        assert plotter is not None

    def test_has_methods(self, plotter):
        """Test plotter has expected methods"""
        assert hasattr(plotter, "plot_time_series")
        assert hasattr(plotter, "plot_energy")

    @patch("matplotlib.pyplot.show")
    def test_plot_time_series(self, mock_show, plotter, basic_solution, cleanup_plots):
        """Test time series plot"""
        fig = plotter.plot_time_series(basic_solution)
        assert fig is not None

    @patch("matplotlib.pyplot.show")
    def test_plot_energy(self, mock_show, plotter, basic_solution, cleanup_plots):
        """Test energy plot"""
        kinetic = 0.5 * basic_solution["y"][1] ** 2
        potential = 0.5 * basic_solution["y"][0] ** 2
        fig = plotter.plot_energy(basic_solution, kinetic, potential)
        assert fig is not None


# ============================================================================
# TEST PHASE SPACE VISUALIZER
# ============================================================================


class TestPhaseSpaceVisualizer:
    """Test PhaseSpaceVisualizer class"""

    def test_init(self):
        """Test initialization"""
        viz = PhaseSpaceVisualizer()
        assert viz is not None

    def test_has_methods(self, phase_viz):
        """Test phase viz has expected methods"""
        assert hasattr(phase_viz, "plot_phase_portrait")

    @patch("matplotlib.pyplot.show")
    def test_plot_phase_portrait(self, mock_show, phase_viz, basic_solution, cleanup_plots):
        """Test phase portrait plot"""
        fig = phase_viz.plot_phase_portrait(basic_solution)
        assert fig is not None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_pendulum_workflow(self, cleanup_plots):
        """Test complete pendulum workflow"""
        viz = Animator()

        # Create solution
        t = np.linspace(0, 10, 100)
        theta = np.sin(t)
        theta_dot = np.cos(t)
        solution = {
            "success": True,
            "t": t,
            "y": np.array([theta, theta_dot]),
            "coordinates": ["theta"],
        }
        params = {"l": 1.0, "m": 1.0, "g": 9.81}

        # Test animation via generic dispatcher
        anim = viz.animate(solution, params, "pendulum")
        assert anim is not None

        # Test direct animation
        viz2 = Animator()
        anim2 = viz2.animate_pendulum(solution, length=1.0)
        assert anim2 is not None

    def test_animation_frame_execution(self, cleanup_plots):
        """Test that animation frames can be executed"""
        viz = Animator(trail_length=10)
        t = np.linspace(0, 5, 50)
        theta = np.sin(t)
        theta_dot = np.cos(t)
        solution = {
            "success": True,
            "t": t,
            "y": np.array([theta, theta_dot]),
            "coordinates": ["theta"],
        }

        anim = viz.animate_pendulum(solution, length=1.0)

        # Execute a few frames to ensure no errors
        assert anim is not None
        # Frame 0
        anim._func(0)
        # Frame in middle
        anim._func(25)
        # Frame at end
        anim._func(49)

    def test_particle_animation_workflow(self, cleanup_plots):
        """Test particle animation workflow"""
        viz = Animator()

        positions = [(np.random.rand(20) * 2 - 1, np.random.rand(20) * 2 - 1) for _ in range(30)]

        anim = viz.animate_particles(positions, title="Particles")
        assert anim is not None

        # Execute frames
        anim._func(0)
        anim._func(15)
        anim._func(29)


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestEdgeCases:
    """Test edge cases"""

    def test_small_solution(self, animator, cleanup_plots):
        """Test with very small solution"""
        t = np.linspace(0, 1, 10)
        theta = np.sin(t)
        theta_dot = np.cos(t)
        solution = {
            "success": True,
            "t": t,
            "y": np.array([theta, theta_dot]),
            "coordinates": ["theta"],
        }
        anim = animator.animate_pendulum(solution, length=0.5)
        assert anim is not None

    def test_large_solution(self, animator, cleanup_plots):
        """Test with large solution"""
        t = np.linspace(0, 100, 1000)
        theta = np.sin(t)
        theta_dot = np.cos(t)
        solution = {
            "success": True,
            "t": t,
            "y": np.array([theta, theta_dot]),
            "coordinates": ["theta"],
        }
        anim = animator.animate_pendulum(solution, length=1.0)
        assert anim is not None

    def test_zero_initial_conditions(self, animator, cleanup_plots):
        """Test with zero initial conditions"""
        t = np.linspace(0, 10, 100)
        theta = np.zeros_like(t)
        theta_dot = np.zeros_like(t)
        solution = {
            "success": True,
            "t": t,
            "y": np.array([theta, theta_dot]),
            "coordinates": ["theta"],
        }
        anim = animator.animate_pendulum(solution, length=1.0)
        assert anim is not None

    def test_constant_solution(self, animator, cleanup_plots):
        """Test with constant solution"""
        t = np.linspace(0, 10, 100)
        theta = np.ones_like(t) * 0.5
        theta_dot = np.zeros_like(t)
        solution = {
            "success": True,
            "t": t,
            "y": np.array([theta, theta_dot]),
            "coordinates": ["theta"],
        }
        anim = animator.animate_pendulum(solution, length=1.0)
        assert anim is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=mechanics_dsl.visualization", "--cov-report=term-missing"])
