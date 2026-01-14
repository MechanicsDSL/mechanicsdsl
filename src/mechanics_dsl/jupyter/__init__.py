"""
MechanicsDSL Jupyter Integration

IPython magic commands and widgets for interactive physics simulation.

Usage in Jupyter:
    %load_ext mechanics_dsl.jupyter
    
    %%mechanicsdsl
    \\system{pendulum}
    \\lagrangian{...}
"""

from .magic import (
    MechanicsDSLMagics,
    load_ipython_extension,
    unload_ipython_extension,
)
from .display import (
    display_simulation,
    display_phase_portrait,
    display_energy_plot,
)

__all__ = [
    'MechanicsDSLMagics',
    'load_ipython_extension',
    'unload_ipython_extension',
    'display_simulation',
    'display_phase_portrait',
    'display_energy_plot',
]
