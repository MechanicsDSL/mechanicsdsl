"""
Tests for visualization/phase_space.py (PhaseSpaceVisualizer).

Covers: __init__, plot_phase_portrait, plot_phase_portrait_3d, plot_poincare_section
(enough and not enough crossings).
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from mechanics_dsl.visualization.phase_space import PhaseSpaceVisualizer


@pytest.fixture
def phase_visualizer():
    return PhaseSpaceVisualizer()


@pytest.fixture
def solution_1d():
    """Solution with one coordinate (x, x_dot)."""
    return {
        "t": np.linspace(0, 10, 100),
        "y": np.vstack([
            np.cos(np.linspace(0, 10, 100)),
            -np.sin(np.linspace(0, 10, 100)),
        ]),
        "coordinates": ["x"],
    }


@pytest.fixture
def solution_2d():
    """Solution with two coordinates for 3D and Poincaré."""
    t = np.linspace(0, 20, 500)
    return {
        "t": t,
        "y": np.vstack([
            np.cos(t),
            -np.sin(t),
            np.cos(0.7 * t),
            -0.7 * np.sin(0.7 * t),
        ]),
        "coordinates": ["x", "y"],
    }


class TestPhaseSpaceVisualizerInit:
    """Test __init__."""

    def test_init_succeeds_with_matplotlib(self):
        v = PhaseSpaceVisualizer()
        assert v is not None


class TestPlotPhasePortrait:
    """Test plot_phase_portrait."""

    def test_plot_phase_portrait(self, phase_visualizer, solution_1d):
        fig = phase_visualizer.plot_phase_portrait(
            solution_1d, coordinate_index=0, title="Test"
        )
        assert fig is not None

    def test_plot_phase_portrait_coord_index_out_of_range(self, phase_visualizer, solution_1d):
        with pytest.raises(ValueError, match="out of range"):
            phase_visualizer.plot_phase_portrait(solution_1d, coordinate_index=5)


class TestPlotPhasePortrait3d:
    """Test plot_phase_portrait_3d."""

    def test_plot_phase_portrait_3d(self, phase_visualizer, solution_2d):
        fig = phase_visualizer.plot_phase_portrait_3d(
            solution_2d,
            coords=(0, 0, 1),  # (coord1_idx, coord1_type, coord2_idx)
            title="3D Test",
        )
        assert fig is not None


class TestPlotPoincareSection:
    """Test plot_poincare_section."""

    def test_plot_poincare_section_enough_crossings(self, phase_visualizer, solution_2d):
        fig = phase_visualizer.plot_poincare_section(
            solution_2d,
            section_var=0,
            section_value=0.0,
            plot_vars=(1, 2),
            title="Poincaré",
        )
        assert fig is not None

    def test_plot_poincare_section_not_enough_crossings(self, phase_visualizer):
        """When len(crossings_idx) < 2, returns empty figure and logs warning."""
        solution = {
            "t": np.linspace(0, 1, 10),
            "y": np.ones((2, 10)),  # No crossings
            "coordinates": ["x"],
        }
        fig = phase_visualizer.plot_poincare_section(
            solution, section_var=0, section_value=0.5
        )
        assert fig is not None
