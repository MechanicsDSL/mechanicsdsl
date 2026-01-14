"""
MechanicsDSL Plugin System

Extensible plugin architecture for custom physics domains, code generators,
visualizations, and solvers.

Example Usage:
    # Register a custom domain plugin
    from mechanics_dsl.plugins import register_domain, PhysicsDomainPlugin
    
    @register_domain("acoustics")
    class AcousticsPlugin(PhysicsDomainPlugin):
        def get_domain_class(self):
            return AcousticsDomain

    # Register via entry points (pyproject.toml)
    # [project.entry-points."mechanics_dsl.domains"]
    # acoustics = "my_package.acoustics:AcousticsPlugin"
"""

from .base import (
    Plugin,
    PhysicsDomainPlugin,
    CodeGeneratorPlugin,
    VisualizationPlugin,
    SolverPlugin,
)
from .registry import (
    PluginRegistry,
    registry,
    register_domain,
    register_generator,
    register_visualization,
    register_solver,
    get_plugin,
    list_plugins,
)
from .loader import (
    PluginLoader,
    load_entry_point_plugins,
    load_plugin_from_path,
)

__all__ = [
    # Base classes
    'Plugin',
    'PhysicsDomainPlugin',
    'CodeGeneratorPlugin',
    'VisualizationPlugin',
    'SolverPlugin',
    # Registry
    'PluginRegistry',
    'registry',
    'register_domain',
    'register_generator',
    'register_visualization',
    'register_solver',
    'get_plugin',
    'list_plugins',
    # Loader
    'PluginLoader',
    'load_entry_point_plugins',
    'load_plugin_from_path',
]
