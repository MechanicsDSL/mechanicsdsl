"""
MechanicsDSL Visualization Package

Modular visualization tools for animations, plots, and phase space analysis.
"""

import os

# Re-export original MechanicsVisualizer for backward compatibility
# Note: This imports from the parent package's visualization.py file
import sys

# New modular components
from .animator import Animator
from .phase_space import PhaseSpaceVisualizer
from .plotter import Plotter

# Import the original MechanicsVisualizer from the module file
# We need a workaround since the module file has the same name as this package
_parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_viz_module_path = os.path.join(_parent_path, "visualization.py")

# Use a module name that coverage tools can track
_MODULE_NAME = "mechanics_dsl._visualization_module"


def _load_legacy_visualizer():
    """Load MechanicsVisualizer from the standalone visualization.py module.

    Uses importlib because the module file shares a name with this package.
    Falls back to Animator if loading fails.
    """
    if not os.path.exists(_viz_module_path):
        return type(
            "MechanicsVisualizer",
            (Animator,),
            {"__doc__": "Backward-compatible wrapper for MechanicsVisualizer."},
        )

    import importlib.util
    import logging

    _spec = importlib.util.spec_from_file_location(_MODULE_NAME, _viz_module_path)
    if _spec is None or _spec.loader is None:
        return type(
            "MechanicsVisualizer",
            (Animator,),
            {"__doc__": "Backward-compatible wrapper for MechanicsVisualizer."},
        )

    _viz_module = importlib.util.module_from_spec(_spec)
    _viz_module.__package__ = "mechanics_dsl"
    sys.modules[_MODULE_NAME] = _viz_module
    try:
        _spec.loader.exec_module(_viz_module)
        return _viz_module.MechanicsVisualizer
    except Exception as e:
        logging.getLogger(__name__).debug(f"Failed to load legacy MechanicsVisualizer: {e}")
        return type(
            "MechanicsVisualizer",
            (Animator,),
            {"__doc__": "Backward-compatible wrapper for MechanicsVisualizer."},
        )


MechanicsVisualizer = _load_legacy_visualizer()


__all__ = ["Animator", "Plotter", "PhaseSpaceVisualizer", "MechanicsVisualizer"]
