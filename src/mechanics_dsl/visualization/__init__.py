"""
MechanicsDSL Visualization Package

Modular visualization tools for animations, plots, and phase space analysis.

The :class:`MechanicsVisualizer` class is the legacy all-in-one visualizer
preserved for backward compatibility. New code should prefer the focused
:class:`Animator`, :class:`Plotter`, and :class:`PhaseSpaceVisualizer`
classes.
"""

from ._legacy import MechanicsVisualizer
from .animator import Animator
from .phase_space import PhaseSpaceVisualizer
from .plotter import Plotter

__all__ = ["Animator", "MechanicsVisualizer", "PhaseSpaceVisualizer", "Plotter"]
