"""
MechanicsDSL Jupyter Integration

IPython magic commands and widgets for interactive physics simulation.

Usage in Jupyter:
    %load_ext mechanics_dsl.jupyter

    %%mechanicsdsl
    \\system{pendulum}
    \\lagrangian{...}
"""

from .display import (
    display_energy_plot,
    display_phase_portrait,
    display_simulation,
)
from .magic import (
    MechanicsDSLMagics,
    load_ipython_extension,
    unload_ipython_extension,
)

__all__ = [
    "MechanicsDSLMagics",
    "load_ipython_extension",
    "unload_ipython_extension",
    "display_simulation",
    "display_phase_portrait",
    "display_energy_plot",
]
