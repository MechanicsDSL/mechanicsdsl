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
    CodeGeneratorPlugin,
    PhysicsDomainPlugin,
    Plugin,
    SolverPlugin,
    VisualizationPlugin,
)
from .loader import (
    PluginLoader,
    load_entry_point_plugins,
    load_plugin_from_path,
)
from .registry import (
    PluginRegistry,
    get_plugin,
    list_plugins,
    register_domain,
    register_generator,
    register_solver,
    register_visualization,
    registry,
)

__all__ = [
    # Base classes
    "Plugin",
    "PhysicsDomainPlugin",
    "CodeGeneratorPlugin",
    "VisualizationPlugin",
    "SolverPlugin",
    # Registry
    "PluginRegistry",
    "registry",
    "register_domain",
    "register_generator",
    "register_visualization",
    "register_solver",
    "get_plugin",
    "list_plugins",
    # Loader
    "PluginLoader",
    "load_entry_point_plugins",
    "load_plugin_from_path",
]
