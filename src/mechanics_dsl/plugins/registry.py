"""
Plugin registry for MechanicsDSL.

Central registry for all loaded plugins with decorator-based registration.
"""

import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Type

from .base import (
    CodeGeneratorPlugin,
    ImporterPlugin,
    PhysicsDomainPlugin,
    Plugin,
    SolverPlugin,
    TransformPlugin,
    VisualizationPlugin,
)


class PluginType(Enum):
    """Types of plugins."""

    DOMAIN = auto()
    GENERATOR = auto()
    VISUALIZATION = auto()
    SOLVER = auto()
    TRANSFORM = auto()
    IMPORTER = auto()


@dataclass
class RegisteredPlugin:
    """Container for a registered plugin."""

    name: str
    plugin_type: PluginType
    plugin_class: Type[Plugin]
    instance: Optional[Plugin] = None
    enabled: bool = True

    def get_instance(self) -> Plugin:
        """Get or create plugin instance (lazy loading)."""
        if self.instance is None:
            self.instance = self.plugin_class()
        return self.instance


class PluginRegistry:
    """
    Central registry for all MechanicsDSL plugins.

    Thread-safe singleton that manages plugin registration, discovery,
    and lifecycle.

    Example:
        # Register directly
        registry.register(PluginType.DOMAIN, "acoustics", AcousticsPlugin)

        # Get plugin
        acoustics = registry.get(PluginType.DOMAIN, "acoustics")

        # List plugins
        for name in registry.list(PluginType.GENERATOR):
            print(f"Generator: {name}")
    """

    _instance: Optional["PluginRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "PluginRegistry":
        """Singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._plugins: Dict[PluginType, Dict[str, RegisteredPlugin]] = {
            plugin_type: {} for plugin_type in PluginType
        }
        self._hooks: Dict[str, List[Callable]] = {}
        self._initialized = True

    def register(
        self, plugin_type: PluginType, name: str, plugin_class: Type[Plugin], replace: bool = False
    ) -> None:
        """
        Register a plugin.

        Args:
            plugin_type: Type of plugin (DOMAIN, GENERATOR, etc.)
            name: Unique name for the plugin
            plugin_class: Plugin class (not instance)
            replace: If True, replace existing plugin with same name

        Raises:
            ValueError: If plugin with same name exists and replace=False
        """
        with self._lock:
            if name in self._plugins[plugin_type] and not replace:
                raise ValueError(
                    f"Plugin '{name}' already registered for {plugin_type.name}. "
                    f"Use replace=True to override."
                )

            registered = RegisteredPlugin(
                name=name, plugin_type=plugin_type, plugin_class=plugin_class
            )

            # Validate plugin before registering
            instance = registered.get_instance()
            if not instance.validate():
                raise ValueError(f"Plugin '{name}' failed validation")

            self._plugins[plugin_type][name] = registered
            self._fire_hook("on_register", plugin_type, name, instance)

    def unregister(self, plugin_type: PluginType, name: str) -> bool:
        """
        Unregister a plugin.

        Args:
            plugin_type: Type of plugin
            name: Plugin name

        Returns:
            True if plugin was removed, False if not found
        """
        with self._lock:
            if name in self._plugins[plugin_type]:
                plugin = self._plugins[plugin_type].pop(name)
                if plugin.instance:
                    plugin.instance.deactivate()
                self._fire_hook("on_unregister", plugin_type, name)
                return True
            return False

    def get(self, plugin_type: PluginType, name: str) -> Optional[Plugin]:
        """
        Get a plugin instance by type and name.

        Args:
            plugin_type: Type of plugin
            name: Plugin name

        Returns:
            Plugin instance or None if not found
        """
        with self._lock:
            registered = self._plugins[plugin_type].get(name)
            if registered and registered.enabled:
                return registered.get_instance()
            return None

    def get_all(self, plugin_type: PluginType) -> Dict[str, Plugin]:
        """
        Get all plugins of a type.

        Args:
            plugin_type: Type of plugin

        Returns:
            Dictionary of name -> plugin instance
        """
        with self._lock:
            return {
                name: reg.get_instance()
                for name, reg in self._plugins[plugin_type].items()
                if reg.enabled
            }

    def list(self, plugin_type: PluginType) -> List[str]:
        """
        List all registered plugin names of a type.

        Args:
            plugin_type: Type of plugin

        Returns:
            List of plugin names
        """
        with self._lock:
            return list(self._plugins[plugin_type].keys())

    def list_all(self) -> Dict[PluginType, List[str]]:
        """
        List all registered plugins by type.

        Returns:
            Dictionary of plugin type -> list of names
        """
        with self._lock:
            return {ptype: list(plugins.keys()) for ptype, plugins in self._plugins.items()}

    def enable(self, plugin_type: PluginType, name: str) -> bool:
        """Enable a disabled plugin."""
        with self._lock:
            if name in self._plugins[plugin_type]:
                self._plugins[plugin_type][name].enabled = True
                instance = self._plugins[plugin_type][name].get_instance()
                instance.activate()
                return True
            return False

    def disable(self, plugin_type: PluginType, name: str) -> bool:
        """Disable a plugin without unregistering."""
        with self._lock:
            if name in self._plugins[plugin_type]:
                reg = self._plugins[plugin_type][name]
                reg.enabled = False
                if reg.instance:
                    reg.instance.deactivate()
                return True
            return False

    def add_hook(self, event: str, callback: Callable) -> None:
        """Add a callback for registry events."""
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)

    def _fire_hook(self, event: str, *args, **kwargs) -> None:
        """Fire all callbacks for an event."""
        for callback in self._hooks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception:
                pass  # Don't let hook errors break registration

    def clear(self) -> None:
        """Clear all registered plugins (for testing)."""
        with self._lock:
            for plugins in self._plugins.values():
                for reg in plugins.values():
                    if reg.instance:
                        reg.instance.deactivate()
                plugins.clear()


# Global registry instance
registry = PluginRegistry()


# Decorator factories for convenient registration
def register_domain(name: str):
    """
    Decorator to register a physics domain plugin.

    Example:
        @register_domain("acoustics")
        class AcousticsPlugin(PhysicsDomainPlugin):
            ...
    """

    def decorator(cls: Type[PhysicsDomainPlugin]) -> Type[PhysicsDomainPlugin]:
        registry.register(PluginType.DOMAIN, name, cls)
        return cls

    return decorator


def register_generator(name: str):
    """
    Decorator to register a code generator plugin.

    Example:
        @register_generator("swift")
        class SwiftGeneratorPlugin(CodeGeneratorPlugin):
            ...
    """

    def decorator(cls: Type[CodeGeneratorPlugin]) -> Type[CodeGeneratorPlugin]:
        registry.register(PluginType.GENERATOR, name, cls)
        return cls

    return decorator


def register_visualization(name: str):
    """
    Decorator to register a visualization plugin.

    Example:
        @register_visualization("plotly")
        class PlotlyPlugin(VisualizationPlugin):
            ...
    """

    def decorator(cls: Type[VisualizationPlugin]) -> Type[VisualizationPlugin]:
        registry.register(PluginType.VISUALIZATION, name, cls)
        return cls

    return decorator


def register_solver(name: str):
    """
    Decorator to register a solver plugin.

    Example:
        @register_solver("symplectic")
        class SymplecticSolverPlugin(SolverPlugin):
            ...
    """

    def decorator(cls: Type[SolverPlugin]) -> Type[SolverPlugin]:
        registry.register(PluginType.SOLVER, name, cls)
        return cls

    return decorator


def register_transform(name: str):
    """Decorator to register a transform plugin."""

    def decorator(cls: Type[TransformPlugin]) -> Type[TransformPlugin]:
        registry.register(PluginType.TRANSFORM, name, cls)
        return cls

    return decorator


def register_importer(name: str):
    """Decorator to register an importer plugin."""

    def decorator(cls: Type[ImporterPlugin]) -> Type[ImporterPlugin]:
        registry.register(PluginType.IMPORTER, name, cls)
        return cls

    return decorator


# Convenience functions
def get_plugin(plugin_type: PluginType, name: str) -> Optional[Plugin]:
    """Get a plugin from the global registry."""
    return registry.get(plugin_type, name)


def list_plugins(plugin_type: Optional[PluginType] = None) -> Any:
    """List all plugins, optionally filtered by type."""
    if plugin_type is None:
        return registry.list_all()
    return registry.list(plugin_type)


__all__ = [
    "PluginType",
    "RegisteredPlugin",
    "PluginRegistry",
    "registry",
    "register_domain",
    "register_generator",
    "register_visualization",
    "register_solver",
    "register_transform",
    "register_importer",
    "get_plugin",
    "list_plugins",
]
