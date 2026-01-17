"""
Extended plugins/loader.py coverage tests.

Covers: load_entry_points (including Python 3.9 fallback), load_directory
(exists, not exists, recursive), load_file, load_module, _classify_plugin,
load_plugin_from_path (file, dir, ValueError), _get_loader, get_loaded_sources.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mechanics_dsl.plugins import (
    PluginLoader,
    load_entry_point_plugins,
    load_plugin_from_path,
)
from mechanics_dsl.plugins.base import PhysicsDomainPlugin, Plugin
from mechanics_dsl.plugins.registry import PluginRegistry, PluginType


@pytest.fixture
def custom_registry():
    """Fresh registry to avoid polluting the global one."""
    return PluginRegistry()


class TestLoadEntryPoints:
    """Test load_entry_points."""

    def test_load_entry_points_returns_dict(self, custom_registry):
        loader = PluginLoader(registry=custom_registry)
        loaded = loader.load_entry_points()
        assert isinstance(loaded, dict)
        for pt in PluginType:
            assert pt in loaded
            assert isinstance(loaded[pt], list)

    def test_load_entry_points_appends_source(self, custom_registry):
        loader = PluginLoader(registry=custom_registry)
        loader.load_entry_points()
        assert "entry_points" in loader.get_loaded_sources()


class TestLoadEntryPointsPython39Fallback:
    """Test Python 3.9 fallback when entry_points(group=...) raises TypeError."""

    def test_entry_points_typeerror_fallback(self, custom_registry):
        def mock_entry_points(*args, **kwargs):
            if "group" in kwargs:
                raise TypeError("entry_points() got an unexpected keyword argument 'group'")
            return {}

        with patch(
            "mechanics_dsl.plugins.loader.importlib.metadata.entry_points",
            side_effect=mock_entry_points,
        ):
            loader = PluginLoader(registry=custom_registry)
            loaded = loader.load_entry_points()
            assert isinstance(loaded, dict)
            for pt in PluginType:
                assert pt in loaded


class TestLoadDirectory:
    """Test load_directory."""

    def test_load_directory_nonexistent(self, custom_registry):
        loader = PluginLoader(registry=custom_registry)
        loaded = loader.load_directory("/nonexistent/path/xyz")
        assert isinstance(loaded, dict)
        for pt in PluginType:
            assert loaded[pt] == []

    def test_load_directory_empty_dir(self, custom_registry):
        with tempfile.TemporaryDirectory() as d:
            loader = PluginLoader(registry=custom_registry)
            loaded = loader.load_directory(d)
            assert isinstance(loaded, dict)
            assert d in loader.get_loaded_sources()

    def test_load_directory_recursive(self, custom_registry):
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "sub"
            sub.mkdir()
            (sub / "empty.py").write_text("# empty\n")
            loader = PluginLoader(registry=custom_registry)
            loaded = loader.load_directory(d, recursive=True)
            assert isinstance(loaded, dict)


class TestLoadFile:
    """Test load_file."""

    def test_load_file_nonexistent_raises(self, custom_registry):
        loader = PluginLoader(registry=custom_registry)
        with pytest.raises(FileNotFoundError):
            loader.load_file("/nonexistent/plugin.py")

    def test_load_file_no_plugin_classes(self, custom_registry):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("x = 1\n")
            f.flush()
            path = f.name
        try:
            loader = PluginLoader(registry=custom_registry)
            loaded = loader.load_file(path)
            assert isinstance(loaded, dict)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_file_with_plugin_class(self, custom_registry):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
from mechanics_dsl.plugins.base import PhysicsDomainPlugin

class TestDomainPlugin(PhysicsDomainPlugin):
    @property
    def metadata(self):
        from mechanics_dsl.plugins.base import PluginMetadata
        return PluginMetadata(name="test_domain", version="0.1")

    def get_domain_class(self):
        return None
"""
            )
            f.flush()
            path = f.name
        try:
            loader = PluginLoader(registry=custom_registry)
            loaded = loader.load_file(path)
            assert isinstance(loaded, dict)
            # May have registered DOMAIN
            assert PluginType.DOMAIN in loaded
        finally:
            Path(path).unlink(missing_ok=True)


class TestLoadModule:
    """Test load_module."""

    def test_load_module_nonexistent_raises(self, custom_registry):
        loader = PluginLoader(registry=custom_registry)
        with pytest.raises(ImportError):
            loader.load_module("nonexistent_module_xyz_123")

    def test_load_module_mechanics_dsl(self, custom_registry):
        loader = PluginLoader(registry=custom_registry)
        loaded = loader.load_module("mechanics_dsl.parser")
        assert isinstance(loaded, dict)


class TestClassifyPlugin:
    """Test _classify_plugin (via load_file with known plugin types)."""

    def test_classify_physics_domain(self, custom_registry):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
from mechanics_dsl.plugins.base import PhysicsDomainPlugin

class MyDomain(PhysicsDomainPlugin):
    @property
    def metadata(self):
        from mechanics_dsl.plugins.base import PluginMetadata
        return PluginMetadata(name="mydomain", version="0.1")
    def get_domain_class(self): return None
"""
            )
            f.flush()
            path = f.name
        try:
            loader = PluginLoader(registry=custom_registry)
            loaded = loader.load_file(path)
            assert PluginType.DOMAIN in loaded
        finally:
            Path(path).unlink(missing_ok=True)


class TestLoadPluginFromPath:
    """Test load_plugin_from_path."""

    def test_load_plugin_from_path_file(self, custom_registry):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("x=1\n")
            f.flush()
            path = f.name
        try:
            with patch("mechanics_dsl.plugins.loader._get_loader") as mock_get:
                mock_loader = PluginLoader(registry=custom_registry)
                mock_get.return_value = mock_loader
                loaded = load_plugin_from_path(path)
                assert isinstance(loaded, dict)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_plugin_from_path_dir(self, custom_registry):
        with tempfile.TemporaryDirectory() as d:
            with patch("mechanics_dsl.plugins.loader._get_loader") as mock_get:
                mock_loader = PluginLoader(registry=custom_registry)
                mock_get.return_value = mock_loader
                loaded = load_plugin_from_path(d)
                assert isinstance(loaded, dict)

    def test_load_plugin_from_path_nonexistent_raises(self):
        with pytest.raises(ValueError, match="Path does not exist"):
            load_plugin_from_path("/nonexistent/path/xyz/123")


class TestGetLoadedSources:
    """Test get_loaded_sources."""

    def test_get_loaded_sources_copy(self, custom_registry):
        loader = PluginLoader(registry=custom_registry)
        loader.load_entry_points()
        sources = loader.get_loaded_sources()
        assert isinstance(sources, list)
        # Modifying the return should not affect internal
        sources.append("x")
        assert "x" not in loader.get_loaded_sources()
