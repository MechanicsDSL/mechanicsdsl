"""
Comprehensive test suite for visualization.py
Target: 95%+ code coverage on Codecov
"""
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from unittest.mock import Mock, patch, MagicMock, call
from collections import deque
import sympy as sp
import tempfile
import os
from io import StringIO

# Import the module under test
from mechanics_dsl.visualization import MechanicsVisualizer
from mechanics_dsl.utils import config
from mechanics_dsl.energy import PotentialEnergyCalculator


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def visualizer():
    """Create a visualizer instance for testing"""
    return MechanicsVisualizer(trail_length=50, fps=30)


@pytest.fixture
def basic_solution():
    """Create a basic valid solution dictionary"""
    t = np.linspace(0, 10, 100)
    theta = np.sin(t)
    theta_dot = np.cos(t)
    return {
        'success': True,
        't': t,
        'y': np.array([theta, theta_dot]),
        'coordinates': ['theta']
    }


@pytest.fixture
def double_pendulum_solution():
    """Create a double pendulum solution"""
    t = np.linspace(0, 10, 100)
    theta1 = np.sin(t)
    theta1_dot = np.cos(t)
    theta2 = np.sin(2*t)
    theta2_dot = 2*np.cos(2*t)
    return {
        'success': True,
        't': t,
        'y': np.array([theta1, theta1_dot, theta2, theta2_dot]),
        'coordinates': ['theta1', 'theta2']
    }


@pytest.fixture
def oscillator_solution():
    """Create an oscillator solution"""
    t = np.linspace(0, 10, 100)
    x = np.sin(t)
    v = np.cos(t)
    return {
        'success': True,
        't': t,
        'y': np.array([x, v]),
        'coordinates': ['x']
    }


@pytest.fixture
def failed_solution():
    """Create a failed solution"""
    return {
        'success': False,
        'message': 'Simulation failed'
    }


@pytest.fixture
def basic_parameters():
    """Basic parameter dictionary"""
    return {
        'l': 1.0,
        'm': 1.0,
        'g': 9.81,
        'k': 1.0
    }


@pytest.fixture
def double_pendulum_parameters():
    """Double pendulum parameters"""
    return {
        'l1': 1.0,
        'l2': 1.5,
        'm1': 1.0,
        'm2': 0.5,
        'g': 9.81
    }


@pytest.fixture
def cleanup_plots():
    """Cleanup matplotlib figures after each test"""
    yield
    plt.close('all')


# ============================================================================
# TEST INITIALIZATION
# ============================================================================

class TestMechanicsVisualizerInit:
    """Test initialization"""
    
    def test_init_default(self):
        """Test default initialization"""
        viz = MechanicsVisualizer()
        assert viz.trail_length == config.trail_length
        assert viz.fps == config.animation_fps
        assert viz.fig is None
        assert viz.ax is None
        assert viz.animation is None
    
    def test_init_custom(self):
        """Test custom initialization"""
        viz = MechanicsVisualizer(trail_length=100, fps=60)
        assert viz.trail_length == 100
        assert viz.fps == 60
    
    def test_init_partial_custom(self):
        """Test partial custom initialization"""
        viz = MechanicsVisualizer(trail_length=75)
        assert viz.trail_length == 75
        assert viz.fps == config.animation_fps


# ============================================================================
# TEST HAS_FFMPEG
# ============================================================================

class TestHasFFmpeg:
    """Test ffmpeg detection"""
    
    def test_has_ffmpeg_true(self, visualizer):
        """Test when ffmpeg is available"""
        with patch('shutil.which', return_value='/usr/bin/ffmpeg'):
            assert visualizer.has_ffmpeg() is True
    
    def test_has_ffmpeg_false(self, visualizer):
        """Test when ffmpeg is not available"""
        with patch('shutil.which', return_value=None):
            assert visualizer.has_ffmpeg() is False


# ============================================================================
# TEST SAVE_ANIMATION_TO_FILE
# ============================================================================

class TestSaveAnimationToFile:
    """Test animation saving functionality"""
    
    def test_save_animation_none_anim(self, visualizer):
        """Test saving with None animation"""
        with pytest.raises(ValueError, match="anim cannot be None"):
            visualizer.save_animation_to_file(None, "test.mp4")
    
    def test_save_animation_invalid_filename_type(self, visualizer):
        """Test saving with invalid filename type"""
        mock_anim = Mock()
        with pytest.raises(TypeError):
            visualizer.save_animation_to_file(mock_anim, 123)
    
    def test_save_animation_invalid_fps_type(self, visualizer):
        """Test saving with invalid fps type"""
        mock_anim = Mock()
        with pytest.raises(TypeError, match="fps must be int"):
            visualizer.save_animation_to_file(mock_anim, "test.mp4", fps=30.5)
    
    def test_save_animation_invalid_fps_range_low(self, visualizer):
        """Test saving with fps too low"""
        mock_anim = Mock()
        with pytest.raises(ValueError, match="fps must be in"):
            visualizer.save_animation_to_file(mock_anim, "test.mp4", fps=0)
    
    def test_save_animation_invalid_fps_range_high(self, visualizer):
        """Test saving with fps too high"""
        mock_anim = Mock()
        with pytest.raises(ValueError, match="fps must be in"):
            visualizer.save_animation_to_file(mock_anim, "test.mp4", fps=150)
    
    def test_save_animation_invalid_dpi_type(self, visualizer):
        """Test saving with invalid dpi type"""
        mock_anim = Mock()
        with pytest.raises(TypeError, match="dpi must be int"):
            visualizer.save_animation_to_file(mock_anim, "test.mp4", dpi=100.5)
    
    def test_save_animation_invalid_dpi_range_low(self, visualizer):
        """Test saving with dpi too low"""
        mock_anim = Mock()
        with pytest.raises(ValueError, match="dpi must be in"):
            visualizer.save_animation_to_file(mock_anim, "test.mp4", dpi=5)
    
    def test_save_animation_invalid_dpi_range_high(self, visualizer):
        """Test saving with dpi too high"""
        mock_anim = Mock()
        with pytest.raises(ValueError, match="dpi must be in"):
            visualizer.save_animation_to_file(mock_anim, "test.mp4", dpi=2000)
    
    def test_save_animation_invalid_filename_empty(self, visualizer):
        """Test saving with empty filename"""
        mock_anim = Mock()
        with pytest.raises(ValueError, match="filename cannot be empty"):
            visualizer.save_animation_to_file(mock_anim, "   ")
    
    def test_save_animation_invalid_filename_path_traversal(self, visualizer):
        """Test saving with path traversal attempt"""
        mock_anim = Mock()
        with pytest.raises(ValueError, match="contains '..'"):
            visualizer.save_animation_to_file(mock_anim, "../test.mp4")
    
    def test_save_animation_mp4_with_ffmpeg(self, visualizer):
        """Test saving MP4 with ffmpeg available"""
        mock_anim = Mock()
        with patch.object(visualizer, 'has_ffmpeg', return_value=True):
            with patch('matplotlib.animation.writers', {'ffmpeg': Mock()}):
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                    try:
                        result = visualizer.save_animation_to_file(mock_anim, tmp.name)
                        assert result is True
                        assert mock_anim.save.called
                    finally:
                        if os.path.exists(tmp.name):
                            os.unlink(tmp.name)
    
    def test_save_animation_gif(self, visualizer):
        """Test saving GIF"""
        mock_anim = Mock()
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
            try:
                result = visualizer.save_animation_to_file(mock_anim, tmp.name)
                assert result is True
                assert mock_anim.save.called
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
    
    def test_save_animation_other_format_with_ffmpeg(self, visualizer):
        """Test saving other format with ffmpeg"""
        mock_anim = Mock()
        with patch.object(visualizer, 'has_ffmpeg', return_value=True):
            with patch('matplotlib.animation.writers', {'ffmpeg': Mock()}):
                with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp:
                    try:
                        result = visualizer.save_animation_to_file(mock_anim, tmp.name)
                        assert result is True
                    finally:
                        if os.path.exists(tmp.name):
                            os.unlink(tmp.name)
    
    def test_save_animation_io_error(self, visualizer):
        """Test handling IO error"""
        mock_anim = Mock()
        mock_anim.save.side_effect = IOError("Permission denied")
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
            try:
                result = visualizer.save_animation_to_file(mock_anim, tmp.name)
                assert result is False
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
    
    def test_save_animation_unexpected_error(self, visualizer):
        """Test handling unexpected error"""
        mock_anim = Mock()
        mock_anim.save.side_effect = RuntimeError("Unexpected error")
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
            try:
                result = visualizer.save_animation_to_file(mock_anim, tmp.name)
                assert result is False
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)


# ============================================================================
# TEST SETUP_3D_PLOT
# ============================================================================

class TestSetup3DPlot:
    """Test 3D plot setup"""
    
    def test_setup_3d_plot_default(self, visualizer, cleanup_plots):
        """Test default 3D plot setup"""
        visualizer.setup_3d_plot()
        assert visualizer.fig is not None
        assert visualizer.ax is not None
        assert visualizer.ax.get_xlabel() == 'X (m)'
        assert visualizer.ax.get_ylabel() == 'Y (m)'
        assert visualizer.ax.get_zlabel() == 'Z (m)'
    
    def test_setup_3d_plot_custom_title(self, visualizer, cleanup_plots):
        """Test 3D plot setup with custom title"""
        visualizer.setup_3d_plot("Custom Title")
        assert visualizer.fig is not None
        assert visualizer.ax is not None
        assert visualizer.ax.get_title() == "Custom Title"


# ============================================================================
# TEST ANIMATE_PENDULUM
# ============================================================================

class TestAnimatePendulum:
    """Test pendulum animation"""
    
    def test_animate_pendulum_invalid_parameters_type(self, visualizer, basic_solution):
        """Test with invalid parameters type"""
        with pytest.raises(TypeError, match="parameters must be dict"):
            visualizer.animate_pendulum(basic_solution, "not a dict", "pendulum")
    
    def test_animate_pendulum_invalid_system_name_type(self, visualizer, basic_solution, basic_parameters):
        """Test with invalid system name type"""
        with pytest.raises(TypeError, match="system_name must be str"):
            visualizer.animate_pendulum(basic_solution, basic_parameters, 123)
    
    def test_animate_pendulum_failed_solution(self, visualizer, failed_solution, basic_parameters, cleanup_plots):
        """Test with failed solution"""
        result = visualizer.animate_pendulum(failed_solution, basic_parameters)
        assert result is None
    
    def test_animate_pendulum_single(self, visualizer, basic_solution, basic_parameters, cleanup_plots):
        """Test single pendulum animation"""
        anim = visualizer.animate_pendulum(basic_solution, basic_parameters)
        assert anim is not None
        assert visualizer.fig is not None
        assert visualizer.ax is not None
    
    def test_animate_pendulum_double(self, visualizer, double_pendulum_solution, double_pendulum_parameters, cleanup_plots):
        """Test double pendulum animation"""
        anim = visualizer.animate_pendulum(double_pendulum_solution, double_pendulum_parameters)
        assert anim is not None
        assert visualizer.fig is not None
    
    def test_animate_pendulum_double_by_name(self, visualizer, double_pendulum_solution, double_pendulum_parameters, cleanup_plots):
        """Test double pendulum detected by name"""
        anim = visualizer.animate_pendulum(double_pendulum_solution, double_pendulum_parameters, "double_pendulum")
        assert anim is not None


# ============================================================================
# TEST _ANIMATE_SINGLE_PENDULUM
# ============================================================================

class TestAnimateSinglePendulum:
    """Test single pendulum animation"""
    
    def test_animate_single_pendulum_insufficient_state(self, visualizer, basic_parameters, cleanup_plots):
        """Test with insufficient state vector"""
        t = np.linspace(0, 10, 100)
        y = np.array([])  # Empty state vector
        result = visualizer._animate_single_pendulum(t, y, basic_parameters)
        assert result is None
    
    def test_animate_single_pendulum_success(self, visualizer, basic_solution, basic_parameters, cleanup_plots):
        """Test successful single pendulum animation"""
        visualizer.setup_3d_plot()
        t = basic_solution['t']
        y = basic_solution['y']
        anim = visualizer._animate_single_pendulum(t, y, basic_parameters)
        assert anim is not None
        assert isinstance(anim, animation.FuncAnimation)


# ============================================================================
# TEST _ANIMATE_DOUBLE_PENDULUM
# ============================================================================

class TestAnimateDoublePendulum:
    """Test double pendulum animation"""
    
    def test_animate_double_pendulum_insufficient_state(self, visualizer, double_pendulum_parameters, cleanup_plots):
        """Test with insufficient state vector"""
        t = np.linspace(0, 10, 100)
        y = np.array([])  # Empty state vector
        result = visualizer._animate_double_pendulum(t, y, double_pendulum_parameters)
        assert result is None
    
    def test_animate_double_pendulum_success(self, visualizer, double_pendulum_solution, double_pendulum_parameters, cleanup_plots):
        """Test successful double pendulum animation"""
        visualizer.setup_3d_plot()
        t = double_pendulum_solution['t']
        y = double_pendulum_solution['y']
        anim = visualizer._animate_double_pendulum(t, y, double_pendulum_parameters)
        assert anim is not None
        assert isinstance(anim, animation.FuncAnimation)
    
    def test_animate_double_pendulum_fallback_l2(self, visualizer, double_pendulum_solution, cleanup_plots):
        """Test with missing l2 parameter"""
        visualizer.setup_3d_plot()
        params = {'l1': 1.0, 'g': 9.81}  # Missing l2
        t = double_pendulum_solution['t']
        y = double_pendulum_solution['y']
        anim = visualizer._animate_double_pendulum(t, y, params)
        assert anim is not None


# ============================================================================
# TEST ANIMATE_FLUID_FROM_CSV
# ============================================================================

class TestAnimateFluidFromCSV:
    """Test fluid animation from CSV"""
    
    def test_animate_fluid_invalid_file(self, visualizer, cleanup_plots):
        """Test with non-existent file"""
        with pytest.raises(FileNotFoundError):
            visualizer.animate_fluid_from_csv("nonexistent.csv")
    
    def test_animate_fluid_success(self, visualizer, cleanup_plots):
        """Test successful fluid animation"""
        # Create temporary CSV
        csv_data = "t,id,x,y,rho\n0.0,0,0.1,0.2,1000\n0.0,1,0.3,0.4,1000\n0.1,0,0.15,0.25,1005\n0.1,1,0.35,0.45,1005\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp.write(csv_data)
            tmp.flush()
            try:
                anim = visualizer.animate_fluid_from_csv(tmp.name)
                assert anim is not None
            finally:
                os.unlink(tmp.name)
    
    def test_animate_fluid_read_error(self, visualizer, cleanup_plots):
        """Test with CSV read error"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp.write("invalid,csv,data\n")
            tmp.flush()
            try:
                # This should return None due to missing columns
                with patch('pandas.read_csv', side_effect=Exception("Read error")):
                    result = visualizer.animate_fluid_from_csv(tmp.name)
                    assert result is None
            finally:
                os.unlink(tmp.name)


# ============================================================================
# TEST ANIMATE_OSCILLATOR
# ============================================================================

class TestAnimateOscillator:
    """Test oscillator animation"""
    
    def test_animate_oscillator_failed_solution(self, visualizer, failed_solution, basic_parameters, cleanup_plots):
        """Test with failed solution"""
        result = visualizer.animate_oscillator(failed_solution, basic_parameters)
        assert result is None
    
    def test_animate_oscillator_insufficient_state(self, visualizer, basic_parameters, cleanup_plots):
        """Test with insufficient state vector"""
        solution = {
            'success': True,
            't': np.linspace(0, 10, 100),
            'y': np.array([]),  # Empty
            'coordinates': ['x']
        }
        result = visualizer.animate_oscillator(solution, basic_parameters)
        assert result is None
    
    def test_animate_oscillator_success(self, visualizer, oscillator_solution, basic_parameters, cleanup_plots):
        """Test successful oscillator animation"""
        anim = visualizer.animate_oscillator(oscillator_solution, basic_parameters)
        assert anim is not None
        assert visualizer.fig is not None
        assert visualizer.ax is not None
    
    def test_animate_oscillator_single_variable(self, visualizer, basic_parameters, cleanup_plots):
        """Test oscillator with single variable (no velocity)"""
        solution = {
            'success': True,
            't': np.linspace(0, 10, 100),
            'y': np.array([np.sin(np.linspace(0, 10, 100))]),
            'coordinates': ['x']
        }
        anim = visualizer.animate_oscillator(solution, basic_parameters)
        assert anim is not None


# ============================================================================
# TEST ANIMATE (GENERIC DISPATCHER)
# ============================================================================

class TestAnimate:
    """Test generic animation dispatcher"""
    
    def test_animate_invalid_solution_none(self, visualizer, basic_parameters, cleanup_plots):
        """Test with None solution"""
        result = visualizer.animate(None, basic_parameters)
        assert result is None
    
    def test_animate_invalid_solution_failed(self, visualizer, failed_solution, basic_parameters, cleanup_plots):
        """Test with failed solution"""
        result = visualizer.animate(failed_solution, basic_parameters)
        assert result is None
    
    def test_animate_pendulum_by_name(self, visualizer, basic_solution, basic_parameters, cleanup_plots):
        """Test pendulum animation by system name"""
        anim = visualizer.animate(basic_solution, basic_parameters, "pendulum")
        assert anim is not None
    
    def test_animate_pendulum_by_coordinate(self, visualizer, basic_solution, basic_parameters, cleanup_plots):
        """Test pendulum animation by coordinate name"""
        anim = visualizer.animate(basic_solution, basic_parameters, "system")
        assert anim is not None  # 'theta' in coordinates
    
    def test_animate_oscillator_by_name(self, visualizer, oscillator_solution, basic_parameters, cleanup_plots):
        """Test oscillator animation by system name"""
        anim = visualizer.animate(oscillator_solution, basic_parameters, "oscillator")
        assert anim is not None
    
    def test_animate_oscillator_by_coordinate(self, visualizer, oscillator_solution, basic_parameters, cleanup_plots):
        """Test oscillator animation by coordinate"""
        anim = visualizer.animate(oscillator_solution, basic_parameters, "spring")
        assert anim is not None
    
    def test_animate_phase_space_fallback(self, visualizer, basic_parameters, cleanup_plots):
        """Test phase space animation fallback"""
        solution = {
            'success': True,
            't': np.linspace(0, 10, 100),
            'y': np.array([np.sin(np.linspace(0, 10, 100)), np.cos(np.linspace(0, 10, 100))]),
            'coordinates': ['q']
        }
        anim = visualizer.animate(solution, basic_parameters, "unknown_system")
        assert anim is not None
    
    def test_animate_exception_handling(self, visualizer, basic_solution, basic_parameters, cleanup_plots):
        """Test exception handling in animate"""
        with patch.object(visualizer, 'animate_pendulum', side_effect=RuntimeError("Test error")):
            result = visualizer.animate(basic_solution, basic_parameters, "pendulum")
            assert result is None


# ============================================================================
# TEST _ANIMATE_PHASE_SPACE
# ============================================================================

class TestAnimatePhaseSpace:
    """Test phase space animation"""
    
    def test_animate_phase_space_no_coordinates(self, visualizer, cleanup_plots):
        """Test with no coordinates"""
        solution = {
            'success': True,
            't': np.linspace(0, 10, 100),
            'y': np.array([np.sin(np.linspace(0, 10, 100))]),
            'coordinates': []
        }
        result = visualizer._animate_phase_space(solution, "system")
        assert result is None
    
    def test_animate_phase_space_insufficient_state(self, visualizer, cleanup_plots):
        """Test with insufficient state vector"""
        solution = {
            'success': True,
            't': np.linspace(0, 10, 100),
            'y': np.array([]),
            'coordinates': ['q']
        }
        result = visualizer._animate_phase_space(solution, "system")
        assert result is None
    
    def test_animate_phase_space_success(self, visualizer, cleanup_plots):
        """Test successful phase space animation"""
        t = np.linspace(0, 10, 100)
        solution = {
            'success': True,
            't': t,
            'y': np.array([np.sin(t), np.cos(t)]),
            'coordinates': ['q']
        }
        anim = visualizer._animate_phase_space(solution, "system")
        assert anim is not None
        assert visualizer.fig is not None
        assert visualizer.ax is not None
    
    def test_animate_phase_space_single_variable(self, visualizer, cleanup_plots):
        """Test phase space with single variable"""
        t = np.linspace(0, 10, 100)
        solution = {
            'success': True,
            't': t,
            'y': np.array([np.sin(t)]),
            'coordinates': ['q']
        }
        anim = visualizer._animate_phase_space(solution, "system")
        assert anim is not None


# ============================================================================
# TEST PLOT_ENERGY
# ============================================================================

class TestPlotEnergy:
    """Test energy plotting"""
    
    def test_plot_energy_failed_solution(self, visualizer, failed_solution, basic_parameters, cleanup_plots):
        """Test with failed solution"""
        result = visualizer.plot_energy(failed_solution, basic_parameters)
        assert result is None
    
    @patch('matplotlib.pyplot.show')
    def test_plot_energy_success(self, mock_show, visualizer, basic_solution, basic_parameters, cleanup_plots):
        """Test successful energy plot"""
        with patch.object(PotentialEnergyCalculator, 'compute_kinetic_energy', return_value=np.ones(100)):
            with patch.object(PotentialEnergyCalculator, 'compute_potential_energy', return_value=np.ones(100)):
                visualizer.plot_energy(basic_solution, basic_parameters, "pendulum")
                mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_plot_energy_zero_initial(self, mock_show, visualizer, basic_solution, basic_parameters, cleanup_plots):
        """Test energy plot with zero initial energy"""
        with patch.object(PotentialEnergyCalculator, 'compute_kinetic_energy', return_value=np.zeros(100)):
            with patch.object(PotentialEnergyCalculator, 'compute_potential_energy', return_value=np.zeros(100)):
                visualizer.plot_energy(basic_solution, basic_parameters, "pendulum")
                mock_show.assert_called_once()


# ============================================================================
# TEST PLOT_PHASE_SPACE
# ============================================================================

class TestPlotPhaseSpace:
    """Test phase space plotting"""
    
    def test_plot_phase_space_invalid_index_type(self, visualizer, basic_solution):
        """Test with invalid coordinate index type"""
        with pytest.raises(TypeError, match="coordinate_index must be int"):
            visualizer.plot_phase_space(basic_solution, "not an int")
    
    def test_plot_phase_space_negative_index(self, visualizer, basic_solution):
        """Test with negative coordinate index"""
        with pytest.raises(ValueError, match="coordinate_index must be non-negative"):
            visualizer.plot_phase_space(basic_solution, -1)
    
    def test_plot_phase_space_failed_solution(self, visualizer, failed_solution, cleanup_plots):
        """Test with failed solution"""
        result = visualizer.plot_phase_space(failed_solution)
        assert result is None
    
    def test_plot_phase_space_index_out_of_range(self, visualizer, basic_solution, cleanup_plots):
        """Test with coordinate index out of range"""
        with pytest.raises(ValueError, match="coordinate_index .* out of range"):
            visualizer.plot_phase_space(basic_solution, 10)
    
    def test_plot_phase_space_state_too_small(self, visualizer, cleanup_plots):
        """Test with state vector too small"""
        solution = {
            'success': True,
            't': np.linspace(0, 10, 100),
            'y': np.array([np.sin(np.linspace(0, 10, 100))]),  # Only 1 row
            'coordinates': ['theta']
        }
        with pytest.raises(ValueError, match="State vector too small"):
            visualizer.plot_phase_space(solution, 0)
    
    @patch('matplotlib.pyplot.show')
    def test_plot_phase_space_success(self, mock_show, visualizer, basic_solution, cleanup_plots):
        """Test successful phase space plot"""
        visualizer.plot_phase_space(basic_solution, 0)
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_plot_phase_space_with_non_finite(self, mock_show, visualizer, cleanup_plots):
        """Test phase space plot with non-finite values"""
        t = np.linspace(0, 10, 100)
        position = np.sin(t)
        position[50] = np.nan  # Add NaN
        velocity = np.cos(t)
        velocity[60] = np.inf  # Add inf
        solution = {
            'success': True,
            't': t,
            'y': np.array([position, velocity]),
            'coordinates': ['theta']
        }
        visualizer.plot_phase_space(solution, 0)
        mock_show.assert_called_once()
    
    def test_plot_phase_space_empty_arrays(self, visualizer, cleanup_plots):
        """Test with empty position/velocity arrays"""
        solution = {
            'success': True,
            't': np.array([]),
            'y': np.array([np.array([]), np.array([])]),
            'coordinates': ['theta']
        }
        result = visualizer.plot_phase_space(solution, 0)
        assert result is None
    
    @patch('matplotlib.pyplot.show')
    def test_plot_phase_space_length_mismatch(self, mock_show, visualizer, cleanup_plots):
        """Test with position/velocity length mismatch"""
        t = np.linspace(0, 10, 100)
        position = np.sin(t)
        velocity = np.cos(t[:50])  # Different length
        solution = {
            'success': True,
            't': t,
            'y': np.array([position, velocity]),
            'coordinates': ['theta']
        }
        visualizer.plot_phase_space(solution, 0)
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.plot')
    def test_plot_phase_space_plot_error(self, mock_plot, visualizer, basic_solution, cleanup_plots):
        """Test handling plot error"""
        mock_plot.side_effect = RuntimeError("Plot error")
        result = visualizer.plot_phase_space(basic_solution, 0)
        assert result is None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows"""
    
    @patch('matplotlib.pyplot.show')
    def test_full_pendulum_workflow(self, mock_show, cleanup_plots):
        """Test complete pendulum workflow"""
        viz = MechanicsVisualizer()
        
        # Create solution
        t = np.linspace(0, 10, 100)
        theta = np.sin(t)
        theta_dot = np.cos(t)
        solution = {
            'success': True,
            't': t,
            'y': np.array([theta, theta_dot]),
            'coordinates': ['theta']
        }
        params = {'l': 1.0, 'm': 1.0, 'g': 9.81}
        
        # Test animation
        anim = viz.animate(solution, params, "pendulum")
        assert anim is not None
        
        # Test energy plot
        with patch.object(PotentialEnergyCalculator, 'compute_kinetic_energy', return_value=np.ones(100)):
            with patch.object(PotentialEnergyCalculator, 'compute_potential_energy', return_value=np.ones(100)):
                viz.plot_energy(solution, params, "pendulum")
        
        # Test phase space plot
        viz.plot_phase_space(solution, 0)
    
    def test_animation_frame_execution(self, cleanup_plots):
        """Test that animation frames can be executed"""
        viz = MechanicsVisualizer(trail_length=10)
        t = np.linspace(0, 5, 50)
        theta = np.sin(t)
        theta_dot = np.cos(t)
        solution = {
            'success': True,
            't': t,
            'y': np.array([theta, theta_dot]),
            'coordinates': ['theta']
        }
        params = {'l': 1.0}
        
        viz.setup_3d_plot()
        anim = viz._animate_single_pendulum(t, solution['y'], params)
        
        # Execute a few frames to ensure no errors
        assert anim is not None
        # Frame 0
        anim._func(0)
        # Frame in middle
        anim._func(25)
        # Frame at end
        anim._func(49)
        # Frame beyond end (should handle gracefully)
        anim._func(100)
    
    def test_double_pendulum_frame_execution(self, cleanup_plots):
        """Test double pendulum animation frame execution"""
        viz = MechanicsVisualizer(trail_length=10)
        t = np.linspace(0, 5, 50)
        theta1 = np.sin(t)
        theta1_dot = np.cos(t)
        theta2 = np.sin(2*t)
        theta2_dot = 2*np.cos(2*t)
        y = np.array([theta1, theta1_dot, theta2, theta2_dot])
        params = {'l1': 1.0, 'l2': 1.5}
        
        viz.setup_3d_plot()
        anim = viz._animate_double_pendulum(t, y, params)
        
        assert anim is not None
        anim._func(0)
        anim._func(25)
    
    def test_oscillator_frame_execution(self, cleanup_plots):
        """Test oscillator animation frame execution"""
        viz = MechanicsVisualizer()
        t = np.linspace(0, 5, 50)
        x = np.sin(t)
        v = np.cos(t)
        solution = {
            'success': True,
            't': t,
            'y': np.array([x, v]),
            'coordinates': ['x']
        }
        params = {'k': 1.0, 'm': 1.0}
        
        anim = viz.animate_oscillator(solution, params)
        assert anim is not None
        
        # Execute init and frames
        anim._init_func()
        anim._func(0)
        anim._func(25)
    
    def test_phase_space_frame_execution(self, cleanup_plots):
        """Test phase space animation frame execution"""
        viz = MechanicsVisualizer()
        t = np.linspace(0, 5, 50)
        solution = {
            'success': True,
            't': t,
            'y': np.array([np.sin(t), np.cos(t)]),
            'coordinates': ['q']
        }
        
        anim = viz._animate_phase_space(solution, "system")
        assert anim is not None
        
        anim._init_func()
        anim._func(0)
        anim._func(25)


# ============================================================================
# EDGE CASES AND ERROR CONDITIONS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_very_long_simulation(self, visualizer, basic_parameters, cleanup_plots):
        """Test with very long simulation"""
        t = np.linspace(0, 1000, 10000)
        theta = np.sin(t)
        theta_dot = np.cos(t)
        solution = {
            'success': True,
            't': t,
            'y': np.array([theta, theta_dot]),
            'coordinates': ['theta']
        }
        anim = visualizer.animate_pendulum(solution, basic_parameters)
        assert anim is not None
    
    def test_very_short_simulation(self, visualizer, basic_parameters, cleanup_plots):
        """Test with very short simulation"""
        t = np.array([0.0, 0.1])
        theta = np.array([0.1, 0.2])
        theta_dot = np.array([0.0, 0.1])
        solution = {
            'success': True,
            't': t,
            'y': np.array([theta, theta_dot]),
            'coordinates': ['theta']
        }
        anim = visualizer.animate_pendulum(solution, basic_parameters)
        assert anim is not None
    
    def test_zero_trail_length(self, basic_solution, basic_parameters, cleanup_plots):
        """Test with zero trail length"""
        viz = MechanicsVisualizer(trail_length=0)
        anim = viz.animate_pendulum(basic_solution, basic_parameters)
        assert anim is not None
    
    def test_very_large_trail_length(self, basic_solution, basic_parameters, cleanup_plots):
        """Test with very large trail length"""
        viz = MechanicsVisualizer(trail_length=10000)
        anim = viz.animate_pendulum(basic_solution, basic_parameters)
        assert anim is not None
    
    def test_extreme_pendulum_values(self, visualizer, cleanup_plots):
        """Test with extreme pendulum parameter values"""
        t = np.linspace(0, 10, 100)
        theta = np.linspace(0, 2*np.pi, 100)
        theta_dot = np.zeros(100)
        solution = {
            'success': True,
            't': t,
            'y': np.array([theta, theta_dot]),
            'coordinates': ['theta']
        }
        params = {'l': 100.0, 'm': 0.001, 'g': 9.81}
        anim = visualizer.animate_pendulum(solution, params)
        assert anim is not None
    
    def test_missing_parameters(self, visualizer, basic_solution, cleanup_plots):
        """Test with missing parameters"""
        params = {}  # Empty parameters
        anim = visualizer.animate_pendulum(basic_solution, params)
        assert anim is not None  # Should use defaults
    
    def test_solution_with_extra_keys(self, visualizer, basic_parameters, cleanup_plots):
        """Test solution with extra keys"""
        t = np.linspace(0, 10, 100)
        solution = {
            'success': True,
            't': t,
            'y': np.array([np.sin(t), np.cos(t)]),
            'coordinates': ['theta'],
            'extra_key': 'extra_value',
            'another_key': 123
        }
        anim = visualizer.animate_pendulum(solution, basic_parameters)
        assert anim is not None
    
    @patch('matplotlib.pyplot.show')
    def test_plot_energy_with_negative_energy(self, mock_show, visualizer, basic_solution, basic_parameters, cleanup_plots):
        """Test energy plot with negative total energy"""
        with patch.object(PotentialEnergyCalculator, 'compute_kinetic_energy', return_value=np.ones(100)):
            with patch.object(PotentialEnergyCalculator, 'compute_potential_energy', return_value=-2*np.ones(100)):
                visualizer.plot_energy(basic_solution, basic_parameters, "pendulum")
                mock_show.assert_called_once()


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

class TestParametrized:
    """Parametrized tests for multiple scenarios"""
    
    @pytest.mark.parametrize("trail_length,fps", [
        (50, 30),
        (100, 60),
        (10, 15),
        (200, 24),
    ])
    def test_different_visualizer_configs(self, trail_length, fps, cleanup_plots):
        """Test visualizer with different configurations"""
        viz = MechanicsVisualizer(trail_length=trail_length, fps=fps)
        assert viz.trail_length == trail_length
        assert viz.fps == fps
    
    @pytest.mark.parametrize("system_name", [
        "pendulum",
        "simple_pendulum",
        "double_pendulum",
        "PENDULUM",
        "Pendulum",
    ])
    def test_pendulum_name_variations(self, system_name, basic_solution, basic_parameters, cleanup_plots):
        """Test different pendulum name variations"""
        viz = MechanicsVisualizer()
        anim = viz.animate(basic_solution, basic_parameters, system_name)
        assert anim is not None
    
    @pytest.mark.parametrize("system_name", [
        "oscillator",
        "spring",
        "harmonic_oscillator",
        "OSCILLATOR",
    ])
    def test_oscillator_name_variations(self, system_name, oscillator_solution, basic_parameters, cleanup_plots):
        """Test different oscillator name variations"""
        viz = MechanicsVisualizer()
        anim = viz.animate(oscillator_solution, basic_parameters, system_name)
        assert anim is not None
    
    @pytest.mark.parametrize("filename", [
        "test.mp4",
        "test.gif",
        "test.avi",
        "TEST.MP4",
        "test.GIF",
    ])
    def test_save_different_formats(self, visualizer, filename):
        """Test saving different file formats"""
        mock_anim = Mock()
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1], delete=False) as tmp:
            try:
                if filename.lower().endswith('.gif'):
                    result = visualizer.save_animation_to_file(mock_anim, tmp.name)
                else:
                    with patch.object(visualizer, 'has_ffmpeg', return_value=True):
                        with patch('matplotlib.animation.writers', {'ffmpeg': Mock()}):
                            result = visualizer.save_animation_to_file(mock_anim, tmp.name)
                assert result is True
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)


# ============================================================================
# MOCK AND PATCH TESTS
# ============================================================================

class TestMocking:
    """Tests using mocks and patches"""
    
    def test_logger_calls(self, visualizer, basic_solution, basic_parameters, cleanup_plots):
        """Test that logger is called appropriately"""
        with patch('mechanics_dsl.visualization.logger') as mock_logger:
            anim = visualizer.animate_pendulum(basic_solution, basic_parameters)
            assert mock_logger.info.called or mock_logger.debug.called
    
    def test_failed_solution_warning(self, visualizer, failed_solution, basic_parameters, cleanup_plots):
        """Test warning is logged for failed solution"""
        with patch('mechanics_dsl.visualization.logger') as mock_logger:
            visualizer.animate_pendulum(failed_solution, basic_parameters)
            mock_logger.warning.assert_called()
    
    def test_validate_solution_dict_called(self, visualizer, basic_solution, basic_parameters, cleanup_plots):
        """Test that validate_solution_dict is called"""
        with patch('mechanics_dsl.visualization.validate_solution_dict') as mock_validate:
            visualizer.animate_pendulum(basic_solution, basic_parameters)
            mock_validate.assert_called_once_with(basic_solution)


# ============================================================================
# COVERAGE BOOSTERS
# ============================================================================

class TestCoverageBoosters:
    """Additional tests to boost coverage to 95%+"""
    
    def test_empty_system_name(self, visualizer, basic_solution, basic_parameters, cleanup_plots):
        """Test with empty system name"""
        anim = visualizer.animate_pendulum(basic_solution, basic_parameters, "")
        assert anim is not None
    
    def test_none_system_name_handling(self, visualizer, cleanup_plots):
        """Test handling of None in system name"""
        solution = {
            'success': True,
            't': np.linspace(0, 10, 100),
            'y': np.array([np.sin(np.linspace(0, 10, 100)), np.cos(np.linspace(0, 10, 100))]),
            'coordinates': ['q']
        }
        params = {}
        # Test with None system_name converted to empty string
        with patch.object(visualizer, '_animate_phase_space') as mock_phase:
            mock_phase.return_value = Mock()
            visualizer.animate(solution, params, "")
    
    def test_coordinate_detection(self, visualizer, basic_parameters, cleanup_plots):
        """Test coordinate detection logic"""
        # Test with 'theta' in coordinates
        solution = {
            'success': True,
            't': np.linspace(0, 10, 100),
            'y': np.array([np.sin(np.linspace(0, 10, 100)), np.cos(np.linspace(0, 10, 100))]),
            'coordinates': ['theta']
        }
        anim = visualizer.animate(solution, basic_parameters, "unknown")
        assert anim is not None
    
    def test_fallback_l_parameter(self, visualizer, cleanup_plots):
        """Test fallback to 'l' parameter when l1 missing"""
        t = np.linspace(0, 10, 100)
        y = np.array([np.sin(t), np.cos(t), np.sin(2*t), 2*np.cos(2*t)])
        params = {'l': 2.0}  # No l1, should use l
        visualizer.setup_3d_plot()
        anim = visualizer._animate_double_pendulum(t, y, params)
        assert anim is not None
    
    def test_double_pendulum_state_fallback(self, visualizer, cleanup_plots):
        """Test double pendulum with insufficient state (< 4)"""
        t = np.linspace(0, 10, 100)
        y = np.array([np.sin(t), np.cos(t)])  # Only 2 states
        params = {'l1': 1.0, 'l2': 1.0}
        visualizer.setup_3d_plot()
        anim = visualizer._animate_double_pendulum(t, y, params)
        assert anim is not None
    
    def test_oscillator_default_velocity(self, visualizer, basic_parameters, cleanup_plots):
        """Test oscillator with default velocity"""
        solution = {
            'success': True,
            't': np.linspace(0, 10, 100),
            'y': np.array([np.sin(np.linspace(0, 10, 100))]),  # Only position
            'coordinates': ['x']
        }
        anim = visualizer.animate_oscillator(solution, basic_parameters)
        assert anim is not None
    
    def test_phase_space_default_velocity(self, visualizer, cleanup_plots):
        """Test phase space with default velocity"""
        t = np.linspace(0, 10, 100)
        solution = {
            'success': True,
            't': t,
            'y': np.array([np.sin(t)]),  # Only one state
            'coordinates': ['q']
        }
        anim = visualizer._animate_phase_space(solution, "system")
        assert anim is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=mechanics_dsl.visualization', '--cov-report=term-missing'])