"""
Dynamic plugin loader for MechanicsDSL.

Discovers and loads plugins from:
- Python entry points (installed packages)
- Local plugin directories
- Individual plugin files
"""

import importlib
import importlib.metadata
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .base import (
    CodeGeneratorPlugin,
    PhysicsDomainPlugin,
    Plugin,
    SolverPlugin,
    VisualizationPlugin,
)
from .registry import PluginRegistry, PluginType, registry

logger = logging.getLogger("MechanicsDSL")


# Entry point group names
ENTRY_POINT_GROUPS = {
    PluginType.DOMAIN: "mechanics_dsl.domains",
    PluginType.GENERATOR: "mechanics_dsl.codegen",
    PluginType.VISUALIZATION: "mechanics_dsl.visualization",
    PluginType.SOLVER: "mechanics_dsl.solvers",
    PluginType.TRANSFORM: "mechanics_dsl.transforms",
    PluginType.IMPORTER: "mechanics_dsl.importers",
}


class PluginLoader:
    """
    Discovers and loads MechanicsDSL plugins.

    Plugins can be loaded from:
    1. Python entry points (pip-installed packages)
    2. Plugin directories
    3. Individual Python files

    Example:
        loader = PluginLoader()

        # Load all entry point plugins
        loader.load_entry_points()

        # Load from a directory
        loader.load_directory("./my_plugins")

        # Load a single file
        loader.load_file("custom_domain.py")
    """

    def __init__(self, registry: Optional[PluginRegistry] = None):
        """
        Initialize loader.

        Args:
            registry: Plugin registry to load into. Uses global registry if None.
        """
        self._registry = registry or globals()["registry"]
        self._loaded_sources: List[str] = []

    def load_entry_points(self) -> Dict[PluginType, List[str]]:
        """
        Load all plugins from Python entry points.

        Scans for packages that declare entry points in the
        mechanics_dsl.* groups.

        Returns:
            Dictionary of plugin type -> list of loaded plugin names
        """
        loaded: Dict[PluginType, List[str]] = {pt: [] for pt in PluginType}

        for plugin_type, group in ENTRY_POINT_GROUPS.items():
            try:
                # Python 3.10+ API
                eps = importlib.metadata.entry_points(group=group)
            except TypeError:
                # Python 3.9 fallback
                all_eps = importlib.metadata.entry_points()
                eps = all_eps.get(group, [])

            for ep in eps:
                try:
                    plugin_class = ep.load()
                    self._registry.register(plugin_type, ep.name, plugin_class)
                    loaded[plugin_type].append(ep.name)
                    logger.info(f"Loaded plugin '{ep.name}' from entry point")
                except Exception as e:
                    logger.warning(f"Failed to load plugin '{ep.name}': {e}")

        self._loaded_sources.append("entry_points")
        return loaded

    def load_directory(
        self, directory: str, recursive: bool = False
    ) -> Dict[PluginType, List[str]]:
        """
        Load all plugins from a directory.

        Scans for Python files and loads any Plugin subclasses found.

        Args:
            directory: Path to plugin directory
            recursive: If True, scan subdirectories

        Returns:
            Dictionary of plugin type -> list of loaded plugin names
        """
        loaded: Dict[PluginType, List[str]] = {pt: [] for pt in PluginType}
        dir_path = Path(directory)

        if not dir_path.exists():
            logger.warning(f"Plugin directory not found: {directory}")
            return loaded

        pattern = "**/*.py" if recursive else "*.py"

        for file_path in dir_path.glob(pattern):
            if file_path.name.startswith("_"):
                continue

            try:
                file_loaded = self.load_file(str(file_path))
                for pt, names in file_loaded.items():
                    loaded[pt].extend(names)
            except Exception as e:
                logger.warning(f"Failed to load plugin file '{file_path}': {e}")

        self._loaded_sources.append(str(directory))
        return loaded

    def load_file(self, file_path: str) -> Dict[PluginType, List[str]]:
        """
        Load plugins from a Python file.

        Args:
            file_path: Path to Python file

        Returns:
            Dictionary of plugin type -> list of loaded plugin names
        """
        loaded: Dict[PluginType, List[str]] = {pt: [] for pt in PluginType}
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Plugin file not found: {file_path}")

        # Create module spec and load
        module_name = f"mechanics_dsl_plugin_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)

        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Find and register plugin classes
        for name in dir(module):
            obj = getattr(module, name)

            if not isinstance(obj, type):
                continue
            if obj in (
                Plugin,
                PhysicsDomainPlugin,
                CodeGeneratorPlugin,
                VisualizationPlugin,
                SolverPlugin,
            ):
                continue

            plugin_type = self._classify_plugin(obj)
            if plugin_type is None:
                continue

            try:
                instance = obj()
                plugin_name = instance.metadata.name
                self._registry.register(plugin_type, plugin_name, obj)
                loaded[plugin_type].append(plugin_name)
                logger.info(f"Loaded plugin '{plugin_name}' from {file_path}")
            except Exception as e:
                logger.warning(f"Failed to register {name} from {file_path}: {e}")

        return loaded

    def load_module(self, module_name: str) -> Dict[PluginType, List[str]]:
        """
        Load plugins from an installed Python module.

        Args:
            module_name: Fully qualified module name

        Returns:
            Dictionary of plugin type -> list of loaded plugin names
        """
        loaded: Dict[PluginType, List[str]] = {pt: [] for pt in PluginType}

        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(f"Cannot import module {module_name}: {e}")

        for name in dir(module):
            obj = getattr(module, name)

            if not isinstance(obj, type):
                continue

            plugin_type = self._classify_plugin(obj)
            if plugin_type is None:
                continue

            try:
                instance = obj()
                plugin_name = instance.metadata.name
                self._registry.register(plugin_type, plugin_name, obj)
                loaded[plugin_type].append(plugin_name)
            except Exception:
                pass

        return loaded

    def _classify_plugin(self, cls: Type) -> Optional[PluginType]:
        """Determine plugin type from class inheritance."""
        if not isinstance(cls, type):
            return None

        if issubclass(cls, PhysicsDomainPlugin) and cls != PhysicsDomainPlugin:
            return PluginType.DOMAIN
        elif issubclass(cls, CodeGeneratorPlugin) and cls != CodeGeneratorPlugin:
            return PluginType.GENERATOR
        elif issubclass(cls, VisualizationPlugin) and cls != VisualizationPlugin:
            return PluginType.VISUALIZATION
        elif issubclass(cls, SolverPlugin) and cls != SolverPlugin:
            return PluginType.SOLVER

        return None

    def get_loaded_sources(self) -> List[str]:
        """Return list of sources plugins were loaded from."""
        return self._loaded_sources.copy()


# Module-level convenience functions
_loader: Optional[PluginLoader] = None


def _get_loader() -> PluginLoader:
    """Get or create the global loader."""
    global _loader
    if _loader is None:
        _loader = PluginLoader()
    return _loader


def load_entry_point_plugins() -> Dict[PluginType, List[str]]:
    """
    Load all plugins from Python entry points.

    Call this at application startup to discover installed plugins.

    Returns:
        Dictionary of plugin type -> list of loaded plugin names

    Example:
        from mechanics_dsl.plugins import load_entry_point_plugins

        loaded = load_entry_point_plugins()
        print(f"Loaded {len(loaded[PluginType.DOMAIN])} domain plugins")
    """
    return _get_loader().load_entry_points()


def load_plugin_from_path(path: str) -> Dict[PluginType, List[str]]:
    """
    Load plugins from a file or directory.

    Args:
        path: Path to plugin file or directory

    Returns:
        Dictionary of plugin type -> list of loaded plugin names
    """
    p = Path(path)
    if p.is_file():
        return _get_loader().load_file(path)
    elif p.is_dir():
        return _get_loader().load_directory(path)
    else:
        raise ValueError(f"Path does not exist: {path}")


__all__ = [
    "PluginLoader",
    "load_entry_point_plugins",
    "load_plugin_from_path",
    "ENTRY_POINT_GROUPS",
]
