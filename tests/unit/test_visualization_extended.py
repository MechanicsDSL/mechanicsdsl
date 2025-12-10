"""
Comprehensive unit tests for the visualization module.

Tests the visualization classes: Animator, Plotter, PhaseSpaceVisualizer.
"""

import pytest
import numpy as np

# Set matplotlib to non-interactive backend BEFORE any imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mechanics_dsl.visualization import Animator, Plotter, PhaseSpaceVisualizer


@pytest.fixture
def pendulum_solution():
    """Create a pendulum solution."""
    t = np.linspace(0, 10, 100)
    theta = 0.5 * np.cos(t)
    theta_dot = -0.5 * np.sin(t)
    return {
        'success': True,
        't': t,
        'y': np.array([theta, theta_dot]),
        'coordinates': ['theta']
    }


@pytest.fixture
def oscillator_solution():
    """Create an oscillator solution."""
    t = np.linspace(0, 10, 100)
    x = np.cos(t)
    x_dot = -np.sin(t)
    return {
        'success': True,
        't': t,
        'y': np.array([x, x_dot]),
        'coordinates': ['x']
    }


class TestAnimatorInit:
    """Tests for Animator initialization."""
    
    def test_init(self):
        animator = Animator()
        assert animator is not None
    
    def test_init_creates_instance(self):
        animator = Animator()
        assert animator.__class__.__name__ == 'Animator'


class TestAnimatorSetupFigure:
    """Tests for Animator setup_figure method."""
    
    def test_setup_figure(self):
        animator = Animator()
        animator.setup_figure()
        
        assert animator.fig is not None
        assert animator.ax is not None
        plt.close('all')
    
    def test_setup_figure_custom_limits(self):
        animator = Animator()
        animator.setup_figure(xlim=(-5, 5), ylim=(-3, 3))
        
        assert animator.fig is not None
        plt.close('all')
    
    def test_setup_figure_custom_title(self):
        animator = Animator()
        animator.setup_figure(title="Test Animation")
        
        assert animator.fig is not None
        plt.close('all')


class TestAnimatorAnimatePendulum:
    """Tests for Animator animate_pendulum method."""
    
    def test_animate_pendulum(self, pendulum_solution):
        animator = Animator()
        result = animator.animate_pendulum(
            pendulum_solution, 
            length=1.0, 
            title="Pendulum"
        )
        
        assert result is not None
        plt.close('all')
    
    def test_animate_pendulum_custom_length(self, pendulum_solution):
        animator = Animator()
        result = animator.animate_pendulum(
            pendulum_solution, 
            length=2.0, 
            title="Pendulum"
        )
        
        assert result is not None
        plt.close('all')


class TestAnimatorAnimateOscillator:
    """Tests for Animator animate_oscillator method if exists."""
    
    def test_has_animate_method(self, oscillator_solution):
        animator = Animator()
        if hasattr(animator, 'animate_oscillator'):
            assert callable(animator.animate_oscillator)


class TestAnimatorAnimate:
    """Tests for Animator animate dispatcher method."""
    
    def test_has_animate(self, pendulum_solution):
        animator = Animator()
        assert hasattr(animator, 'animate')


class TestPlotterInit:
    """Tests for Plotter initialization."""
    
    def test_init(self):
        plotter = Plotter()
        assert plotter is not None
    
    def test_init_creates_instance(self):
        plotter = Plotter()
        assert plotter.__class__.__name__ == 'Plotter'


class TestPlotterPlotTimeSeries:
    """Tests for Plotter plot_time_series method."""
    
    def test_plot_time_series(self, oscillator_solution):
        plotter = Plotter()
        result = plotter.plot_time_series(
            oscillator_solution,
            title="Time Series"
        )
        
        assert result is not None
        plt.close('all')


class TestPlotterPlotEnergy:
    """Tests for Plotter plot_energy method."""
    
    def test_plot_energy(self, oscillator_solution):
        plotter = Plotter()
        
        t = oscillator_solution['t']
        kinetic = 0.5 * (oscillator_solution['y'][1])**2
        potential = 0.5 * (oscillator_solution['y'][0])**2
        
        result = plotter.plot_energy(
            oscillator_solution,
            kinetic=kinetic,
            potential=potential,
            title="Energy"
        )
        
        assert result is not None
        plt.close('all')


class TestPhaseSpaceVisualizerInit:
    """Tests for PhaseSpaceVisualizer initialization."""
    
    def test_init(self):
        viz = PhaseSpaceVisualizer()
        assert viz is not None
    
    def test_init_creates_instance(self):
        viz = PhaseSpaceVisualizer()
        assert viz.__class__.__name__ == 'PhaseSpaceVisualizer'


class TestPhaseSpaceVisualizerPlotPhasePortrait:
    """Tests for PhaseSpaceVisualizer plot_phase_portrait method."""
    
    def test_plot_phase_portrait(self, oscillator_solution):
        viz = PhaseSpaceVisualizer()
        result = viz.plot_phase_portrait(
            oscillator_solution,
            title="Phase Portrait"
        )
        
        assert result is not None
        plt.close('all')


class TestPhaseSpaceVisualizerPlotPoincare:
    """Tests for PhaseSpaceVisualizer plot_poincare method if exists."""
    
    def test_has_plot_poincare(self):
        viz = PhaseSpaceVisualizer()
        if hasattr(viz, 'plot_poincare'):
            assert callable(viz.plot_poincare)


class TestVisualizationWithDifferentSolutions:
    """Tests for visualization with different types of solutions."""
    
    def test_small_solution(self):
        animator = Animator()
        t = np.linspace(0, 1, 10)
        theta = 0.1 * np.cos(t)
        theta_dot = -0.1 * np.sin(t)
        solution = {
            'success': True,
            't': t,
            'y': np.array([theta, theta_dot]),
            'coordinates': ['theta']
        }
        result = animator.animate_pendulum(solution, length=1.0)
        assert result is not None
        plt.close('all')
    
    def test_large_solution(self):
        animator = Animator()
        t = np.linspace(0, 50, 500)
        theta = 0.5 * np.cos(t)
        theta_dot = -0.5 * np.sin(t)
        solution = {
            'success': True,
            't': t,
            'y': np.array([theta, theta_dot]),
            'coordinates': ['theta']
        }
        result = animator.animate_pendulum(solution, length=1.0)
        assert result is not None
        plt.close('all')


class TestVisualizationEdgeCases:
    """Tests for edge cases in visualization."""
    
    def test_zero_initial_conditions(self):
        plotter = Plotter()
        t = np.linspace(0, 10, 100)
        x = np.zeros_like(t)
        x_dot = np.zeros_like(t)
        solution = {
            'success': True,
            't': t,
            'y': np.array([x, x_dot]),
            'coordinates': ['x']
        }
        result = plotter.plot_time_series(solution)
        assert result is not None
        plt.close('all')
    
    def test_constant_solution(self):
        plotter = Plotter()
        t = np.linspace(0, 10, 100)
        x = np.ones_like(t)
        x_dot = np.zeros_like(t)
        solution = {
            'success': True,
            't': t,
            'y': np.array([x, x_dot]),
            'coordinates': ['x']
        }
        result = plotter.plot_time_series(solution)
        assert result is not None
        plt.close('all')


class TestAnimatorWithVariousLengths:
    """Tests for Animator with various pendulum lengths."""
    
    def test_short_pendulum(self, pendulum_solution):
        animator = Animator()
        result = animator.animate_pendulum(pendulum_solution, length=0.5)
        assert result is not None
        plt.close('all')
    
    def test_long_pendulum(self, pendulum_solution):
        animator = Animator()
        result = animator.animate_pendulum(pendulum_solution, length=3.0)
        assert result is not None
        plt.close('all')
    
    def test_unit_pendulum(self, pendulum_solution):
        animator = Animator()
        result = animator.animate_pendulum(pendulum_solution, length=1.0)
        assert result is not None
        plt.close('all')
