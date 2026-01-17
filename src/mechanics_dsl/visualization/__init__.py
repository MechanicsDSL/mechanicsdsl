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

if os.path.exists(_viz_module_path):
    import importlib.util

    _spec = importlib.util.spec_from_file_location(_MODULE_NAME, _viz_module_path)
    if _spec is not None:
        _viz_module = importlib.util.module_from_spec(_spec)
        # Register in sys.modules BEFORE loading so relative imports work
        _viz_module.__package__ = "mechanics_dsl"
        sys.modules[_MODULE_NAME] = _viz_module
        try:
            if _spec.loader is not None:
                _spec.loader.exec_module(_viz_module)
            MechanicsVisualizer = _viz_module.MechanicsVisualizer
        except Exception:
            # Fallback: create a stub that redirects to Animator
            class MechanicsVisualizer(Animator):  # type: ignore[no-redef]
                """Backward-compatible wrapper for MechanicsVisualizer."""

    else:
        # Fallback: create a stub
        class MechanicsVisualizer(Animator):  # type: ignore[no-redef]
            """Backward-compatible wrapper for MechanicsVisualizer."""

else:
    # Fallback: create a stub
    class MechanicsVisualizer(Animator):
        """Backward-compatible wrapper for MechanicsVisualizer."""


__all__ = ["Animator", "Plotter", "PhaseSpaceVisualizer", "MechanicsVisualizer"]
