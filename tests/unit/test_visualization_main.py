"""
Unit tests for MechanicsDSL visualization module.

Tests the visualization classes: Animator, Plotter, and PhaseSpaceVisualizer.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from mechanics_dsl.visualization.animator import Animator
from mechanics_dsl.visualization.phase_space import PhaseSpaceVisualizer
from mechanics_dsl.visualization.plotter import Plotter


@pytest.fixture
def simple_solution():
    """Create a simple solution dictionary for testing."""
    t = np.linspace(0, 10, 100)
    theta = 0.5 * np.cos(t)
    theta_dot = -0.5 * np.sin(t)
    return {"success": True, "t": t, "y": np.vstack([theta, theta_dot]), "coordinates": ["theta"]}


@pytest.fixture
def oscillator_solution():
    """Create oscillator solution for testing."""
    t = np.linspace(0, 10, 100)
    x = np.cos(t)
    x_dot = -np.sin(t)
    return {"success": True, "t": t, "y": np.vstack([x, x_dot]), "coordinates": ["x"]}


class TestAnimatorInit:
    """Tests for Animator initialization."""

    def test_init_creates_instance(self):
        """Test that Animator can be instantiated."""
        animator = Animator()
        assert animator is not None

    def test_init_fig_is_none(self):
        """Test that fig is None on init."""
        animator = Animator()
        assert animator.fig is None

    def test_init_ax_is_none(self):
        """Test that ax is None on init."""
        animator = Animator()
        assert animator.ax is None

    def test_init_animation_is_none(self):
        """Test that animation is None on init."""
        animator = Animator()
        assert animator.animation is None

    def test_init_with_trail_length(self):
        """Test initialization with custom trail length."""
        animator = Animator(trail_length=200)
        assert animator.trail_length == 200

    def test_init_with_fps(self):
        """Test initialization with custom FPS."""
        animator = Animator(fps=60)
        assert animator.fps == 60

    def test_init_with_both_params(self):
        """Test initialization with both parameters."""
        animator = Animator(trail_length=100, fps=30)
        assert animator.trail_length == 100
        assert animator.fps == 30


class TestAnimatorSetupFigure:
    """Tests for Animator.setup_figure method."""

    def test_setup_figure_returns_tuple(self):
        """Test that setup_figure returns figure and axes."""
        animator = Animator()
        result = animator.setup_figure()

        assert isinstance(result, tuple)
        assert len(result) == 2
        plt.close("all")

    def test_setup_figure_sets_fig(self):
        """Test setup sets self.fig."""
        animator = Animator()
        animator.setup_figure()

        assert animator.fig is not None
        plt.close("all")

    def test_setup_figure_sets_ax(self):
        """Test setup sets self.ax."""
        animator = Animator()
        animator.setup_figure()

        assert animator.ax is not None
        plt.close("all")

    def test_setup_figure_custom_limits(self):
        """Test with custom limits."""
        animator = Animator()
        animator.setup_figure(xlim=(-5, 5), ylim=(-10, 10))

        assert animator.ax is not None
        plt.close("all")

    def test_setup_figure_custom_title(self):
        """Test with custom title."""
        animator = Animator()
        animator.setup_figure(title="Test Animation")

        assert animator.ax is not None
        plt.close("all")


class TestAnimatorAnimatePendulum:
    """Tests for Animator.animate_pendulum method."""

    def test_animate_pendulum_returns_animation(self, simple_solution):
        """Test that animate_pendulum returns animation object."""
        animator = Animator()
        anim = animator.animate_pendulum(simple_solution, length=1.0)

        assert anim is not None
        assert isinstance(anim, animation.FuncAnimation)
        plt.close("all")

    def test_animate_pendulum_with_title(self, simple_solution):
        """Test with custom title."""
        animator = Animator()
        anim = animator.animate_pendulum(simple_solution, length=1.0, title="My Pendulum")

        assert anim is not None
        plt.close("all")

    def test_animate_pendulum_different_length(self, simple_solution):
        """Test with different pendulum lengths."""
        animator = Animator()
        anim = animator.animate_pendulum(simple_solution, length=2.0)

        assert anim is not None
        plt.close("all")


class TestAnimatorAnimate:
    """Tests for Animator.animate dispatcher method."""

    def test_animate_with_solution(self, simple_solution):
        """Test animate dispatches correctly."""
        animator = Animator()
        anim = animator.animate(simple_solution, parameters={"l": 1.0}, system_name="pendulum")

        assert anim is not None
        plt.close("all")

    def test_animate_failed_solution(self):
        """Test animate with failed solution."""
        animator = Animator()
        anim = animator.animate({"success": False}, {}, "test")

        assert anim is None

    def test_animate_none_solution(self):
        """Test animate with None solution."""
        animator = Animator()
        anim = animator.animate(None, {}, "test")

        assert anim is None


class TestAnimatorSave:
    """Tests for Animator.save method."""

    def test_save_without_animation(self):
        """Test save returns False without animation."""
        animator = Animator()
        result = animator.save("test.gif")

        assert result == False


class TestPlotterInit:
    """Tests for Plotter initialization."""

    def test_init_creates_instance(self):
        """Test Plotter can be instantiated."""
        plotter = Plotter()
        assert plotter is not None


class TestPlotterPlotTimeSeries:
    """Tests for Plotter.plot_time_series method."""

    def test_plot_time_series_creates_figure(self, simple_solution):
        """Test that plot_time_series works."""
        plotter = Plotter()
        fig = plotter.plot_time_series(simple_solution)

        assert fig is not None
        plt.close("all")

    def test_plot_time_series_with_title(self, simple_solution):
        """Test with title."""
        plotter = Plotter()
        fig = plotter.plot_time_series(simple_solution, title="Test Plot")

        assert fig is not None
        plt.close("all")


class TestPlotterPlotEnergy:
    """Tests for Plotter.plot_energy method."""

    def test_plot_energy_creates_figure(self, simple_solution):
        """Test that plot_energy creates a figure."""
        plotter = Plotter()
        kinetic = 0.5 * np.sin(simple_solution["t"]) ** 2
        potential = 0.5 * np.cos(simple_solution["t"]) ** 2

        fig = plotter.plot_energy(simple_solution, kinetic, potential)

        assert fig is not None
        plt.close("all")


class TestPhaseSpaceVisualizerInit:
    """Tests for PhaseSpaceVisualizer initialization."""

    def test_init_creates_instance(self):
        """Test PhaseSpaceVisualizer can be instantiated."""
        viz = PhaseSpaceVisualizer()
        assert viz is not None


class TestPhaseSpaceVisualizerPlot:
    """Tests for PhaseSpaceVisualizer.plot_phase_portrait method."""

    def test_plot_phase_portrait(self, simple_solution):
        """Test plotting phase portrait."""
        viz = PhaseSpaceVisualizer()
        fig = viz.plot_phase_portrait(simple_solution)

        assert fig is not None
        plt.close("all")

    def test_plot_phase_portrait_with_title(self, simple_solution):
        """Test with custom title."""
        viz = PhaseSpaceVisualizer()
        fig = viz.plot_phase_portrait(simple_solution, title="Phase Space")

        assert fig is not None
        plt.close("all")

    def test_plot_phase_portrait_coordinate_index(self, simple_solution):
        """Test with coordinate index."""
        viz = PhaseSpaceVisualizer()
        fig = viz.plot_phase_portrait(simple_solution, coordinate_index=0)

        assert fig is not None
        plt.close("all")
