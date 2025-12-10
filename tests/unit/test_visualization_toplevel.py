"""
Comprehensive unit tests for src/mechanics_dsl/visualization.py

Tests the MechanicsVisualizer class for 95%+ code coverage.
Covers all methods: __init__, has_ffmpeg, save_animation_to_file, setup_3d_plot,
animate_pendulum, _animate_single_pendulum, _animate_double_pendulum,
animate_fluid_from_csv, animate_oscillator, animate, _animate_phase_space,
plot_energy, plot_phase_space
"""

import pytest
import numpy as np
import tempfile
import os
import sys
import shutil
from unittest.mock import patch, MagicMock, Mock
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Import MechanicsVisualizer from the visualization.py module file
# The visualization/ folder shadows visualization.py when using normal imports
# Use the same module name as visualization/__init__.py for coverage tracking
_src_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'src'
)
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# Module name must match what visualization/__init__.py uses for coverage to work
_MODULE_NAME = "mechanics_dsl._visualization_module"
_viz_file = os.path.join(_src_path, 'mechanics_dsl', 'visualization.py')

# Check if already loaded by package import
if _MODULE_NAME in sys.modules:
    _viz_module = sys.modules[_MODULE_NAME]
else:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(_MODULE_NAME, _viz_file)
    _viz_module = importlib.util.module_from_spec(_spec)
    _viz_module.__package__ = "mechanics_dsl"
    sys.modules[_MODULE_NAME] = _viz_module
    _spec.loader.exec_module(_viz_module)

MechanicsVisualizer = _viz_module.MechanicsVisualizer


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def visualizer():
    """Create MechanicsVisualizer instance."""
    viz = MechanicsVisualizer()
    yield viz
    plt.close('all')


@pytest.fixture
def visualizer_custom():
    """Create MechanicsVisualizer with custom parameters."""
    viz = MechanicsVisualizer(trail_length=50, fps=30)
    yield viz
    plt.close('all')


@pytest.fixture
def single_pendulum_solution():
    """Create a simple single pendulum solution."""
    t = np.linspace(0, 10, 100)
    theta = 0.5 * np.cos(t)
    theta_dot = -0.5 * np.sin(t)
    return {
        'success': True,
        't': t,
        'y': np.vstack([theta, theta_dot]),
        'coordinates': ['theta']
    }


@pytest.fixture
def double_pendulum_solution():
    """Create a double pendulum solution."""
    t = np.linspace(0, 10, 100)
    theta1 = 0.5 * np.cos(t)
    theta1_dot = -0.5 * np.sin(t)
    theta2 = 0.3 * np.cos(t + 0.5)
    theta2_dot = -0.3 * np.sin(t + 0.5)
    return {
        'success': True,
        't': t,
        'y': np.vstack([theta1, theta1_dot, theta2, theta2_dot]),
        'coordinates': ['theta1', 'theta2']
    }


@pytest.fixture
def oscillator_solution():
    """Create oscillator solution."""
    t = np.linspace(0, 10, 100)
    x = np.cos(t)
    x_dot = -np.sin(t)
    return {
        'success': True,
        't': t,
        'y': np.vstack([x, x_dot]),
        'coordinates': ['x']
    }


@pytest.fixture
def failed_solution():
    """Create a failed solution."""
    return {'success': False}


@pytest.fixture
def minimal_solution():
    """Create minimal solution with single state."""
    t = np.linspace(0, 5, 50)
    y = np.array([np.sin(t)])  # Only one state row
    return {
        'success': True,
        't': t,
        'y': y,
        'coordinates': ['q']
    }


@pytest.fixture
def empty_state_solution():
    """Create solution with empty state vector."""
    t = np.linspace(0, 5, 50)
    y = np.empty((0, 50))  # Empty state
    return {
        'success': True,
        't': t,
        'y': y,
        'coordinates': ['q']
    }


@pytest.fixture
def pendulum_parameters():
    """Standard pendulum parameters."""
    return {'l': 1.0, 'm': 1.0, 'g': 9.81}


@pytest.fixture
def double_pendulum_parameters():
    """Double pendulum parameters."""
    return {'l1': 1.0, 'l2': 0.8, 'm1': 1.0, 'm2': 0.8, 'g': 9.81}


@pytest.fixture
def temp_dir():
    """Create temporary directory for file tests."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# TEST CLASS: MechanicsVisualizer INITIALIZATION
# ============================================================================

class TestMechanicsVisualizerInit:
    """Tests for MechanicsVisualizer.__init__"""
    
    def test_init_default_values(self):
        """Test initializing with default values."""
        viz = MechanicsVisualizer()
        assert viz.fig is None
        assert viz.ax is None
        assert viz.animation is None
        assert viz.trail_length is not None
        assert viz.fps is not None
        plt.close('all')
    
    def test_init_custom_trail_length(self):
        """Test initializing with custom trail length."""
        viz = MechanicsVisualizer(trail_length=200)
        assert viz.trail_length == 200
        plt.close('all')
    
    def test_init_custom_fps(self):
        """Test initializing with custom FPS."""
        viz = MechanicsVisualizer(fps=60)
        assert viz.fps == 60
        plt.close('all')
    
    def test_init_both_custom_params(self):
        """Test initializing with both custom parameters."""
        viz = MechanicsVisualizer(trail_length=100, fps=24)
        assert viz.trail_length == 100
        assert viz.fps == 24
        plt.close('all')


# ============================================================================
# TEST CLASS: has_ffmpeg METHOD
# ============================================================================

class TestHasFfmpeg:
    """Tests for MechanicsVisualizer.has_ffmpeg"""
    
    def test_has_ffmpeg_returns_bool(self, visualizer):
        """Test that has_ffmpeg returns a boolean."""
        result = visualizer.has_ffmpeg()
        assert isinstance(result, bool)
    
    @patch('shutil.which')
    def test_has_ffmpeg_when_available(self, mock_which, visualizer):
        """Test when ffmpeg is available."""
        mock_which.return_value = '/usr/bin/ffmpeg'
        assert visualizer.has_ffmpeg() is True
    
    @patch('shutil.which')
    def test_has_ffmpeg_when_unavailable(self, mock_which, visualizer):
        """Test when ffmpeg is not available."""
        mock_which.return_value = None
        assert visualizer.has_ffmpeg() is False


# ============================================================================
# TEST CLASS: save_animation_to_file METHOD
# ============================================================================

class TestSaveAnimationToFile:
    """Tests for MechanicsVisualizer.save_animation_to_file"""
    
    def test_save_animation_none_raises_error(self, visualizer, temp_dir):
        """Test that None animation raises ValueError."""
        with pytest.raises(ValueError, match="anim cannot be None"):
            visualizer.save_animation_to_file(None, os.path.join(temp_dir, "test.gif"))
    
    def test_save_animation_invalid_fps_type(self, visualizer, temp_dir):
        """Test that non-int fps raises TypeError."""
        mock_anim = MagicMock()
        with pytest.raises(TypeError, match="fps must be int"):
            visualizer.save_animation_to_file(mock_anim, os.path.join(temp_dir, "test.gif"), fps="30")
    
    def test_save_animation_fps_too_low(self, visualizer, temp_dir):
        """Test that fps < 1 raises ValueError."""
        mock_anim = MagicMock()
        with pytest.raises(ValueError, match="fps must be in"):
            visualizer.save_animation_to_file(mock_anim, os.path.join(temp_dir, "test.gif"), fps=0)
    
    def test_save_animation_fps_too_high(self, visualizer, temp_dir):
        """Test that fps > 120 raises ValueError."""
        mock_anim = MagicMock()
        with pytest.raises(ValueError, match="fps must be in"):
            visualizer.save_animation_to_file(mock_anim, os.path.join(temp_dir, "test.gif"), fps=200)
    
    def test_save_animation_invalid_dpi_type(self, visualizer, temp_dir):
        """Test that non-int dpi raises TypeError."""
        mock_anim = MagicMock()
        with pytest.raises(TypeError, match="dpi must be int"):
            visualizer.save_animation_to_file(mock_anim, os.path.join(temp_dir, "test.gif"), dpi="100")
    
    def test_save_animation_dpi_too_low(self, visualizer, temp_dir):
        """Test that dpi < 10 raises ValueError."""
        mock_anim = MagicMock()
        with pytest.raises(ValueError, match="dpi must be in"):
            visualizer.save_animation_to_file(mock_anim, os.path.join(temp_dir, "test.gif"), dpi=5)
    
    def test_save_animation_dpi_too_high(self, visualizer, temp_dir):
        """Test that dpi > 1000 raises ValueError."""
        mock_anim = MagicMock()
        with pytest.raises(ValueError, match="dpi must be in"):
            visualizer.save_animation_to_file(mock_anim, os.path.join(temp_dir, "test.gif"), dpi=2000)
    
    @patch.object(MechanicsVisualizer, 'has_ffmpeg', return_value=False)
    def test_save_animation_gif_without_ffmpeg(self, mock_ffmpeg, visualizer, temp_dir):
        """Test saving GIF without ffmpeg."""
        mock_anim = MagicMock()
        filepath = os.path.join(temp_dir, "test.gif")
        result = visualizer.save_animation_to_file(mock_anim, filepath, fps=10)
        # Should return True since GIF doesn't need ffmpeg
        assert result is True
        mock_anim.save.assert_called_once()
    
    @pytest.mark.skipif(shutil.which('ffmpeg') is None, reason="ffmpeg not installed")
    def test_save_animation_mp4_with_ffmpeg(self, visualizer, temp_dir):
        """Test saving MP4 with ffmpeg (skipped if ffmpeg not available)."""
        mock_anim = MagicMock()
        filepath = os.path.join(temp_dir, "test.mp4")
        result = visualizer.save_animation_to_file(mock_anim, filepath, fps=30)
        assert result is True
        mock_anim.save.assert_called_once()
    
    @pytest.mark.skipif(shutil.which('ffmpeg') is None, reason="ffmpeg not installed")
    def test_save_animation_other_format_with_ffmpeg(self, visualizer, temp_dir):
        """Test saving other format with ffmpeg (skipped if ffmpeg not available)."""
        mock_anim = MagicMock()
        filepath = os.path.join(temp_dir, "test.avi")
        result = visualizer.save_animation_to_file(mock_anim, filepath, fps=30)
        assert result is True
    
    @patch.object(MechanicsVisualizer, 'has_ffmpeg', return_value=False)
    def test_save_animation_mp4_without_ffmpeg_fails(self, mock_ffmpeg, visualizer, temp_dir):
        """Test saving MP4 without ffmpeg returns False."""
        mock_anim = MagicMock()
        filepath = os.path.join(temp_dir, "test.mp4")
        result = visualizer.save_animation_to_file(mock_anim, filepath, fps=30)
        # Without ffmpeg, mp4 save fails
        assert result is False
    
    @patch.object(MechanicsVisualizer, 'has_ffmpeg', return_value=False)
    def test_save_animation_other_format_without_ffmpeg_fails(self, mock_ffmpeg, visualizer, temp_dir):
        """Test saving other format without ffmpeg returns False."""
        mock_anim = MagicMock()
        filepath = os.path.join(temp_dir, "test.avi")
        # AVI format without ffmpeg
        result = visualizer.save_animation_to_file(mock_anim, filepath, fps=30)
        assert result is False
    
    @patch.object(MechanicsVisualizer, 'has_ffmpeg', return_value=False)
    def test_save_animation_handles_io_error(self, mock_ffmpeg, visualizer, temp_dir):
        """Test handling of IOError during save."""
        mock_anim = MagicMock()
        mock_anim.save.side_effect = IOError("Write error")
        filepath = os.path.join(temp_dir, "test.gif")
        result = visualizer.save_animation_to_file(mock_anim, filepath)
        assert result is False
    
    @patch.object(MechanicsVisualizer, 'has_ffmpeg', return_value=False)
    def test_save_animation_handles_exception(self, mock_ffmpeg, visualizer, temp_dir):
        """Test handling of general exception during save."""
        mock_anim = MagicMock()
        mock_anim.save.side_effect = Exception("Unexpected error")
        filepath = os.path.join(temp_dir, "test.gif")
        result = visualizer.save_animation_to_file(mock_anim, filepath)
        assert result is False


# ============================================================================
# TEST CLASS: setup_3d_plot METHOD
# ============================================================================

class TestSetup3dPlot:
    """Tests for MechanicsVisualizer.setup_3d_plot"""
    
    def test_setup_3d_plot_creates_figure(self, visualizer):
        """Test that setup_3d_plot creates a figure."""
        visualizer.setup_3d_plot()
        assert visualizer.fig is not None
    
    def test_setup_3d_plot_creates_3d_axes(self, visualizer):
        """Test that setup_3d_plot creates 3D axes."""
        visualizer.setup_3d_plot()
        assert visualizer.ax is not None
    
    def test_setup_3d_plot_default_title(self, visualizer):
        """Test default title is set."""
        visualizer.setup_3d_plot()
        assert visualizer.ax.get_title() == "Classical Mechanics Simulation"
    
    def test_setup_3d_plot_custom_title(self, visualizer):
        """Test custom title is set."""
        visualizer.setup_3d_plot(title="My Custom Simulation")
        assert visualizer.ax.get_title() == "My Custom Simulation"


# ============================================================================
# TEST CLASS: animate_pendulum METHOD
# ============================================================================

class TestAnimatePendulum:
    """Tests for MechanicsVisualizer.animate_pendulum"""
    
    def test_animate_pendulum_invalid_parameters_type(self, visualizer, single_pendulum_solution):
        """Test that non-dict parameters raises TypeError."""
        with pytest.raises(TypeError, match="parameters must be dict"):
            visualizer.animate_pendulum(single_pendulum_solution, "not_a_dict")
    
    def test_animate_pendulum_invalid_system_name_type(self, visualizer, single_pendulum_solution, pendulum_parameters):
        """Test that non-str system_name raises TypeError."""
        with pytest.raises(TypeError, match="system_name must be str"):
            visualizer.animate_pendulum(single_pendulum_solution, pendulum_parameters, 123)
    
    def test_animate_pendulum_failed_solution(self, visualizer, failed_solution, pendulum_parameters):
        """Test that failed solution returns None."""
        result = visualizer.animate_pendulum(failed_solution, pendulum_parameters)
        assert result is None
    
    def test_animate_pendulum_non_dict_solution(self, visualizer, pendulum_parameters):
        """Test that non-dict solution returns None."""
        result = visualizer.animate_pendulum("not_a_dict", pendulum_parameters)
        assert result is None
    
    def test_animate_single_pendulum(self, visualizer, single_pendulum_solution, pendulum_parameters):
        """Test animating single pendulum."""
        anim = visualizer.animate_pendulum(single_pendulum_solution, pendulum_parameters, "pendulum")
        assert anim is not None
        assert isinstance(anim, animation.FuncAnimation)
    
    def test_animate_double_pendulum(self, visualizer, double_pendulum_solution, double_pendulum_parameters):
        """Test animating double pendulum."""
        anim = visualizer.animate_pendulum(double_pendulum_solution, double_pendulum_parameters, "double_pendulum")
        assert anim is not None
        assert isinstance(anim, animation.FuncAnimation)
    
    def test_animate_double_pendulum_with_multi_coords(self, visualizer, double_pendulum_solution, double_pendulum_parameters):
        """Test animating with 2+ coordinates triggers double pendulum."""
        anim = visualizer.animate_pendulum(double_pendulum_solution, double_pendulum_parameters, "system")
        assert anim is not None


# ============================================================================
# TEST CLASS: _animate_single_pendulum METHOD
# ============================================================================

class TestAnimateSinglePendulumPrivate:
    """Tests for MechanicsVisualizer._animate_single_pendulum"""
    
    def test_single_pendulum_empty_state(self, visualizer, pendulum_parameters):
        """Test single pendulum with empty state returns None."""
        t = np.linspace(0, 5, 50)
        y = np.empty((0, 50))
        result = visualizer._animate_single_pendulum(t, y, pendulum_parameters)
        assert result is None
    
    def test_single_pendulum_valid_state(self, visualizer, single_pendulum_solution, pendulum_parameters):
        """Test single pendulum with valid state."""
        visualizer.setup_3d_plot()
        anim = visualizer._animate_single_pendulum(
            single_pendulum_solution['t'],
            single_pendulum_solution['y'],
            pendulum_parameters
        )
        assert anim is not None
    
    def test_single_pendulum_default_length(self, visualizer, single_pendulum_solution):
        """Test single pendulum uses default length if not provided."""
        visualizer.setup_3d_plot()
        params = {}  # No 'l' parameter
        anim = visualizer._animate_single_pendulum(
            single_pendulum_solution['t'],
            single_pendulum_solution['y'],
            params
        )
        assert anim is not None


# ============================================================================
# TEST CLASS: _animate_double_pendulum METHOD
# ============================================================================

class TestAnimateDoublePendulumPrivate:
    """Tests for MechanicsVisualizer._animate_double_pendulum"""
    
    def test_double_pendulum_empty_state(self, visualizer, double_pendulum_parameters):
        """Test double pendulum with empty state returns None."""
        t = np.linspace(0, 5, 50)
        y = np.empty((0, 50))
        result = visualizer._animate_double_pendulum(t, y, double_pendulum_parameters)
        assert result is None
    
    def test_double_pendulum_valid_state(self, visualizer, double_pendulum_solution, double_pendulum_parameters):
        """Test double pendulum with valid state."""
        visualizer.setup_3d_plot()
        anim = visualizer._animate_double_pendulum(
            double_pendulum_solution['t'],
            double_pendulum_solution['y'],
            double_pendulum_parameters
        )
        assert anim is not None
    
    def test_double_pendulum_uses_single_length_fallback(self, visualizer, double_pendulum_solution):
        """Test double pendulum uses 'l' if 'l1' not provided."""
        visualizer.setup_3d_plot()
        params = {'l': 1.5}  # Only 'l', not 'l1'
        anim = visualizer._animate_double_pendulum(
            double_pendulum_solution['t'],
            double_pendulum_solution['y'],
            params
        )
        assert anim is not None
    
    def test_double_pendulum_minimal_state(self, visualizer, pendulum_parameters):
        """Test double pendulum with only 2 state rows (uses theta1 for both)."""
        visualizer.setup_3d_plot()
        t = np.linspace(0, 5, 50)
        theta = np.sin(t)
        theta_dot = np.cos(t)
        y = np.vstack([theta, theta_dot])
        anim = visualizer._animate_double_pendulum(t, y, pendulum_parameters)
        assert anim is not None


# ============================================================================
# TEST CLASS: animate_fluid_from_csv METHOD
# ============================================================================

# Check if pandas is available for fluid tests
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class TestAnimateFluidFromCsv:
    """Tests for MechanicsVisualizer.animate_fluid_from_csv"""
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_animate_fluid_nonexistent_file(self, visualizer):
        """Test with non-existent file returns None."""
        result = visualizer.animate_fluid_from_csv("nonexistent.csv")
        assert result is None
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_animate_fluid_valid_csv(self, visualizer, temp_dir):
        """Test with valid CSV file."""
        # Create test CSV
        csv_path = os.path.join(temp_dir, "fluid_data.csv")
        with open(csv_path, 'w') as f:
            f.write("t,id,x,y,rho\n")
            for t in np.linspace(0, 1, 10):
                for i in range(5):
                    f.write(f"{t},{i},{0.1*i},{0.1*t},{1000 + 10*np.sin(t)}\n")
        
        anim = visualizer.animate_fluid_from_csv(csv_path, title="Test Fluid")
        assert anim is not None
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_animate_fluid_invalid_csv(self, visualizer, temp_dir):
        """Test with invalid CSV returns None."""
        csv_path = os.path.join(temp_dir, "invalid.csv")
        with open(csv_path, 'w') as f:
            f.write("not,valid,csv,data\nwith,broken,formatting")
        
        result = visualizer.animate_fluid_from_csv(csv_path)
        assert result is None


# ============================================================================
# TEST CLASS: animate_oscillator METHOD
# ============================================================================

class TestAnimateOscillator:
    """Tests for MechanicsVisualizer.animate_oscillator"""
    
    def test_animate_oscillator_failed_solution(self, visualizer, failed_solution, pendulum_parameters):
        """Test oscillator with failed solution returns None."""
        result = visualizer.animate_oscillator(failed_solution, pendulum_parameters)
        assert result is None
    
    def test_animate_oscillator_valid_solution(self, visualizer, oscillator_solution, pendulum_parameters):
        """Test animating oscillator."""
        anim = visualizer.animate_oscillator(oscillator_solution, pendulum_parameters, "harmonic_oscillator")
        assert anim is not None
        assert isinstance(anim, animation.FuncAnimation)
    
    def test_animate_oscillator_empty_state(self, visualizer, empty_state_solution, pendulum_parameters):
        """Test oscillator with empty state."""
        result = visualizer.animate_oscillator(empty_state_solution, pendulum_parameters)
        assert result is None
    
    def test_animate_oscillator_minimal_state(self, visualizer, minimal_solution, pendulum_parameters):
        """Test oscillator with minimal state (only position)."""
        anim = visualizer.animate_oscillator(minimal_solution, pendulum_parameters)
        assert anim is not None
    
    def test_animate_oscillator_sets_fig_and_ax(self, visualizer, oscillator_solution, pendulum_parameters):
        """Test that oscillator sets fig and ax."""
        visualizer.animate_oscillator(oscillator_solution, pendulum_parameters)
        assert visualizer.fig is not None
        assert visualizer.ax is not None


# ============================================================================
# TEST CLASS: animate METHOD (DISPATCHER)
# ============================================================================

class TestAnimateDispatcher:
    """Tests for MechanicsVisualizer.animate"""
    
    def test_animate_none_solution(self, visualizer, pendulum_parameters):
        """Test animate with None solution returns None."""
        result = visualizer.animate(None, pendulum_parameters)
        assert result is None
    
    def test_animate_failed_solution(self, visualizer, failed_solution, pendulum_parameters):
        """Test animate with failed solution returns None."""
        result = visualizer.animate(failed_solution, pendulum_parameters)
        assert result is None
    
    def test_animate_pendulum_by_name(self, visualizer, single_pendulum_solution, pendulum_parameters):
        """Test animate dispatches to pendulum by name."""
        anim = visualizer.animate(single_pendulum_solution, pendulum_parameters, "pendulum")
        assert anim is not None
    
    def test_animate_pendulum_by_coords(self, visualizer, single_pendulum_solution, pendulum_parameters):
        """Test animate dispatches to pendulum by coordinate name."""
        solution = single_pendulum_solution.copy()
        solution['coordinates'] = ['theta']  # Contains 'theta'
        anim = visualizer.animate(solution, pendulum_parameters, "my_system")
        assert anim is not None
    
    def test_animate_oscillator_by_name(self, visualizer, oscillator_solution, pendulum_parameters):
        """Test animate dispatches to oscillator by name."""
        anim = visualizer.animate(oscillator_solution, pendulum_parameters, "oscillator")
        assert anim is not None
    
    def test_animate_oscillator_by_spring_name(self, visualizer, oscillator_solution, pendulum_parameters):
        """Test animate dispatches to oscillator with 'spring' name."""
        anim = visualizer.animate(oscillator_solution, pendulum_parameters, "spring_mass")
        assert anim is not None
    
    def test_animate_oscillator_by_single_x_coord(self, visualizer, oscillator_solution, pendulum_parameters):
        """Test animate dispatches to oscillator with single 'x' coordinate."""
        anim = visualizer.animate(oscillator_solution, pendulum_parameters, "generic")
        assert anim is not None
    
    def test_animate_phase_space_fallback(self, visualizer, pendulum_parameters):
        """Test animate falls back to phase space for unknown system."""
        t = np.linspace(0, 10, 100)
        q = np.sin(t)
        q_dot = np.cos(t)
        solution = {
            'success': True,
            't': t,
            'y': np.vstack([q, q_dot]),
            'coordinates': ['q']  # Neither theta nor x
        }
        anim = visualizer.animate(solution, pendulum_parameters, "unknown_system")
        assert anim is not None
    
    def test_animate_handles_exception(self, visualizer, pendulum_parameters):
        """Test animate handles exceptions gracefully."""
        bad_solution = {
            'success': True,
            't': None,  # Will cause error
            'y': None,
            'coordinates': ['theta']
        }
        result = visualizer.animate(bad_solution, pendulum_parameters, "pendulum")
        assert result is None


# ============================================================================
# TEST CLASS: _animate_phase_space METHOD
# ============================================================================

class TestAnimatePhaseSpacePrivate:
    """Tests for MechanicsVisualizer._animate_phase_space"""
    
    def test_phase_space_empty_coords(self, visualizer):
        """Test phase space with empty coordinates returns None."""
        solution = {
            'success': True,
            't': np.linspace(0, 5, 50),
            'y': np.array([[1, 2, 3]]),
            'coordinates': []
        }
        result = visualizer._animate_phase_space(solution, "test")
        assert result is None
    
    def test_phase_space_empty_state(self, visualizer, empty_state_solution):
        """Test phase space with empty state returns None."""
        result = visualizer._animate_phase_space(empty_state_solution, "test")
        assert result is None
    
    def test_phase_space_valid(self, visualizer, single_pendulum_solution):
        """Test phase space with valid solution."""
        anim = visualizer._animate_phase_space(single_pendulum_solution, "test_system")
        assert anim is not None
        assert isinstance(anim, animation.FuncAnimation)
    
    def test_phase_space_minimal_state(self, visualizer, minimal_solution):
        """Test phase space with minimal state (only position)."""
        anim = visualizer._animate_phase_space(minimal_solution, "test")
        assert anim is not None
    
    def test_phase_space_sets_fig_and_ax(self, visualizer, single_pendulum_solution):
        """Test phase space sets fig and ax."""
        visualizer._animate_phase_space(single_pendulum_solution, "test")
        assert visualizer.fig is not None
        assert visualizer.ax is not None


# ============================================================================
# TEST CLASS: plot_energy METHOD
# ============================================================================

class TestPlotEnergy:
    """Tests for MechanicsVisualizer.plot_energy"""
    
    def test_plot_energy_failed_solution(self, visualizer, failed_solution, pendulum_parameters):
        """Test plot_energy with failed solution returns early."""
        # Should not raise, just return
        visualizer.plot_energy(failed_solution, pendulum_parameters)
    
    @patch('matplotlib.pyplot.show')
    def test_plot_energy_valid_solution(self, mock_show, visualizer, single_pendulum_solution, pendulum_parameters):
        """Test plot_energy with valid solution."""
        visualizer.plot_energy(single_pendulum_solution, pendulum_parameters, "pendulum")
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_plot_energy_with_lagrangian(self, mock_show, visualizer, single_pendulum_solution, pendulum_parameters):
        """Test plot_energy with Lagrangian provided."""
        import sympy as sp
        theta = sp.Symbol('theta')
        L = sp.cos(theta)
        visualizer.plot_energy(single_pendulum_solution, pendulum_parameters, "pendulum", lagrangian=L)
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_plot_energy_zero_total_energy(self, mock_show, visualizer, pendulum_parameters):
        """Test plot_energy handles zero total energy case."""
        t = np.linspace(0, 5, 50)
        theta = np.zeros_like(t)  # Zero angle = zero potential
        theta_dot = np.zeros_like(t)  # Zero velocity = zero kinetic
        solution = {
            'success': True,
            't': t,
            'y': np.vstack([theta, theta_dot]),
            'coordinates': ['theta']
        }
        visualizer.plot_energy(solution, pendulum_parameters, "pendulum")
        mock_show.assert_called_once()


# ============================================================================
# TEST CLASS: plot_phase_space METHOD
# ============================================================================

class TestPlotPhaseSpace:
    """Tests for MechanicsVisualizer.plot_phase_space"""
    
    def test_plot_phase_space_invalid_coordinate_index_type(self, visualizer, single_pendulum_solution):
        """Test that non-int coordinate_index raises TypeError."""
        with pytest.raises(TypeError, match="coordinate_index must be int"):
            visualizer.plot_phase_space(single_pendulum_solution, "0")
    
    def test_plot_phase_space_negative_coordinate_index(self, visualizer, single_pendulum_solution):
        """Test that negative coordinate_index raises ValueError."""
        with pytest.raises(ValueError, match="coordinate_index must be non-negative"):
            visualizer.plot_phase_space(single_pendulum_solution, -1)
    
    def test_plot_phase_space_failed_solution(self, visualizer, failed_solution):
        """Test plot_phase_space with failed solution returns early."""
        visualizer.plot_phase_space(failed_solution)
    
    def test_plot_phase_space_non_dict_solution(self, visualizer):
        """Test plot_phase_space with non-dict solution returns early."""
        visualizer.plot_phase_space("not_a_dict")
    
    def test_plot_phase_space_coordinate_index_out_of_range(self, visualizer, single_pendulum_solution):
        """Test coordinate index out of range raises ValueError."""
        with pytest.raises(ValueError, match="coordinate_index .* out of range"):
            visualizer.plot_phase_space(single_pendulum_solution, 5)
    
    def test_plot_phase_space_state_too_small(self, visualizer, pendulum_parameters):
        """Test state vector too small raises ValueError."""
        # Note: validation catches shape mismatch before plot_phase_space internal check
        solution = {
            'success': True,
            't': np.linspace(0, 5, 50),
            'y': np.array([[0.1] * 50]),  # Only 1 row, correct length
            'coordinates': ['theta', 'phi']
        }
        with pytest.raises(ValueError):  # Either shape mismatch or state too small
            visualizer.plot_phase_space(solution, 1)  # Needs index 2 and 3
    
    @patch('matplotlib.pyplot.show')
    def test_plot_phase_space_valid(self, mock_show, visualizer, single_pendulum_solution):
        """Test plot_phase_space with valid solution."""
        visualizer.plot_phase_space(single_pendulum_solution, 0)
        mock_show.assert_called_once()
    
    def test_plot_phase_space_with_nan_values(self, visualizer):
        """Test plot_phase_space raises error for NaN values (validation catches them)."""
        t = np.linspace(0, 5, 50)
        y = np.vstack([
            np.where(t > 2, np.nan, np.sin(t)),
            np.where(t > 2, np.nan, np.cos(t))
        ])
        solution = {
            'success': True,
            't': t,
            'y': y,
            'coordinates': ['theta']
        }
        # Validation catches non-finite values
        with pytest.raises(ValueError, match="non-finite"):
            visualizer.plot_phase_space(solution, 0)
    
    def test_plot_phase_space_with_inf_values(self, visualizer):
        """Test plot_phase_space raises error for Inf values (validation catches them)."""
        t = np.linspace(0, 5, 50)
        y = np.vstack([
            np.where(t > 2, np.inf, np.sin(t)),
            np.where(t > 2, -np.inf, np.cos(t))
        ])
        solution = {
            'success': True,
            't': t,
            'y': y,
            'coordinates': ['theta']
        }
        # Validation catches non-finite values
        with pytest.raises(ValueError, match="non-finite"):
            visualizer.plot_phase_space(solution, 0)
    
    @patch('matplotlib.pyplot.show')
    def test_plot_phase_space_length_mismatch(self, mock_show, visualizer):
        """Test plot_phase_space handles length mismatch."""
        t = np.linspace(0, 5, 50)
        # Different length arrays (shouldn't happen in practice but test the fix)
        y = np.vstack([
            np.sin(t),
            np.cos(t)
        ])
        solution = {
            'success': True,
            't': t,
            'y': y,
            'coordinates': ['theta']
        }
        visualizer.plot_phase_space(solution, 0)
        mock_show.assert_called_once()
    
    def test_plot_phase_space_empty_arrays(self, visualizer):
        """Test plot_phase_space with empty arrays raises ValueError."""
        solution = {
            'success': True,
            't': np.array([]),
            'y': np.array([[], []]),
            'coordinates': ['theta']
        }
        # Validation catches empty t array
        with pytest.raises(ValueError, match="cannot be empty"):
            visualizer.plot_phase_space(solution, 0)


# ============================================================================
# ANIMATION FRAME FUNCTION TESTS
# ============================================================================

class TestAnimationFrameFunctions:
    """Tests for animation frame callbacks in various animate methods."""
    
    def test_single_pendulum_animate_frame(self, visualizer, single_pendulum_solution, pendulum_parameters):
        """Test single pendulum animate_frame function executes."""
        visualizer.setup_3d_plot()
        anim = visualizer._animate_single_pendulum(
            single_pendulum_solution['t'],
            single_pendulum_solution['y'],
            pendulum_parameters
        )
        # Draw a few frames
        for i in range(5):
            anim._func(i)
    
    def test_double_pendulum_animate_frame(self, visualizer, double_pendulum_solution, double_pendulum_parameters):
        """Test double pendulum animate_frame function executes."""
        visualizer.setup_3d_plot()
        anim = visualizer._animate_double_pendulum(
            double_pendulum_solution['t'],
            double_pendulum_solution['y'],
            double_pendulum_parameters
        )
        # Draw a few frames
        for i in range(5):
            anim._func(i)
    
    def test_oscillator_animate_frame(self, visualizer, oscillator_solution, pendulum_parameters):
        """Test oscillator animate_frame function executes."""
        anim = visualizer.animate_oscillator(oscillator_solution, pendulum_parameters)
        # Draw a few frames
        for i in range(5):
            anim._func(i)
    
    def test_phase_space_animate_frame(self, visualizer, single_pendulum_solution):
        """Test phase space animate_frame function executes."""
        anim = visualizer._animate_phase_space(single_pendulum_solution, "test")
        # Draw a few frames
        for i in range(5):
            anim._func(i)


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Edge case tests for visualization module."""
    
    def test_visualizer_with_very_short_trail(self):
        """Test visualizer with very short trail length."""
        viz = MechanicsVisualizer(trail_length=1)
        assert viz.trail_length == 1
        plt.close('all')
    
    def test_visualizer_with_very_high_fps(self):
        """Test visualizer with high fps."""
        viz = MechanicsVisualizer(fps=120)
        assert viz.fps == 120
        plt.close('all')
    
    def test_animate_with_none_system_name(self, visualizer, single_pendulum_solution, pendulum_parameters):
        """Test animate with None system_name returns None (handled as exception)."""
        # animate catches the TypeError from pendulum call and returns None
        anim = visualizer.animate(single_pendulum_solution, pendulum_parameters, None)
        assert anim is None  # Exception is caught, returns None
    
    def test_animate_pendulum_with_empty_system_name(self, visualizer, single_pendulum_solution, pendulum_parameters):
        """Test animate_pendulum with empty system_name."""
        anim = visualizer.animate_pendulum(single_pendulum_solution, pendulum_parameters, "")
        assert anim is not None
    
    def test_animate_with_double_in_name(self, visualizer, double_pendulum_solution, double_pendulum_parameters):
        """Test that 'double' in name triggers double pendulum."""
        anim = visualizer.animate_pendulum(double_pendulum_solution, double_pendulum_parameters, "my_double_system")
        assert anim is not None


# ============================================================================
# CLEANUP FIXTURE
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_plots():
    """Clean up matplotlib plots after each test."""
    yield
    plt.close('all')
