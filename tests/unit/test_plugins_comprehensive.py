"""
Comprehensive tests for plugins system with mocking.
"""

import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestPluginMetadata:
    """Tests for PluginMetadata dataclass"""

    def test_metadata_creation(self):
        """Test creating plugin metadata"""
        from mechanics_dsl.plugins.base import PluginMetadata

        meta = PluginMetadata(name="test_plugin")
        assert meta.name == "test_plugin"
        assert meta.version == "1.0.0"
        assert meta.author == ""

    def test_metadata_with_all_fields(self):
        """Test metadata with all fields"""
        from mechanics_dsl.plugins.base import PluginMetadata

        meta = PluginMetadata(
            name="my_plugin",
            version="2.0.0",
            author="Test Author",
            description="A test plugin",
            dependencies=["dep1", "dep2"],
            homepage="https://example.com",
        )
        assert meta.name == "my_plugin"
        assert meta.version == "2.0.0"
        assert meta.author == "Test Author"
        assert len(meta.dependencies) == 2


class TestPluginBase:
    """Tests for Plugin base class"""

    def test_plugin_is_abstract(self):
        """Test Plugin base is abstract"""
        from mechanics_dsl.plugins.base import Plugin

        # Plugin has abstract methods, can't instantiate directly
        with pytest.raises(TypeError):
            Plugin()

    def test_concrete_plugin(self):
        """Test creating a concrete plugin"""
        from mechanics_dsl.plugins.base import Plugin, PluginMetadata

        class MyPlugin(Plugin):
            @property
            def metadata(self):
                return PluginMetadata(name="my_plugin")

            def activate(self):
                pass

            def deactivate(self):
                pass

            def validate(self):
                return True

        plugin = MyPlugin()
        assert plugin.metadata.name == "my_plugin"
        assert plugin.validate() is True

    def test_plugin_lifecycle(self):
        """Test plugin activate/deactivate"""
        from mechanics_dsl.plugins.base import Plugin, PluginMetadata

        class LifecyclePlugin(Plugin):
            def __init__(self):
                self.activated = False
                self.deactivated = False

            @property
            def metadata(self):
                return PluginMetadata(name="lifecycle")

            def activate(self):
                self.activated = True

            def deactivate(self):
                self.deactivated = True

            def validate(self):
                return True

        plugin = LifecyclePlugin()
        plugin.activate()
        assert plugin.activated is True
        plugin.deactivate()
        assert plugin.deactivated is True


class TestPhysicsDomainPlugin:
    """Tests for PhysicsDomainPlugin"""

    def test_domain_plugin_abstract(self):
        """Test PhysicsDomainPlugin is abstract"""
        from mechanics_dsl.plugins.base import PhysicsDomainPlugin

        with pytest.raises(TypeError):
            PhysicsDomainPlugin()

    def test_concrete_domain_plugin(self):
        """Test creating concrete domain plugin"""
        from mechanics_dsl.plugins.base import PhysicsDomainPlugin, PluginMetadata

        class AcousticsPlugin(PhysicsDomainPlugin):
            @property
            def metadata(self):
                return PluginMetadata(name="acoustics")

            def activate(self):
                pass

            def deactivate(self):
                pass

            def validate(self):
                return True

            def get_domain_class(self):
                return Mock()

            def get_domain_name(self):
                return "acoustics"

            def get_default_parameters(self):
                return {"frequency": 440.0}

        plugin = AcousticsPlugin()
        assert plugin.get_domain_name() == "acoustics"
        assert "frequency" in plugin.get_default_parameters()


class TestCodeGeneratorPlugin:
    """Tests for CodeGeneratorPlugin"""

    def test_generator_plugin_abstract(self):
        """Test CodeGeneratorPlugin is abstract"""
        from mechanics_dsl.plugins.base import CodeGeneratorPlugin

        with pytest.raises(TypeError):
            CodeGeneratorPlugin()

    def test_concrete_generator_plugin(self):
        """Test creating concrete generator plugin"""
        from mechanics_dsl.plugins.base import CodeGeneratorPlugin, PluginMetadata

        class SwiftPlugin(CodeGeneratorPlugin):
            @property
            def metadata(self):
                return PluginMetadata(name="swift_gen")

            def activate(self):
                pass

            def deactivate(self):
                pass

            def validate(self):
                return True

            def get_generator_class(self):
                return Mock()

            def get_target_name(self):
                return "swift"

            def get_file_extension(self):
                return ".swift"

            def get_template_path(self):
                return None

        plugin = SwiftPlugin()
        assert plugin.get_target_name() == "swift"
        assert plugin.get_file_extension() == ".swift"


class TestVisualizationPlugin:
    """Tests for VisualizationPlugin"""

    def test_viz_plugin_abstract(self):
        """Test VisualizationPlugin is abstract"""
        from mechanics_dsl.plugins.base import VisualizationPlugin

        with pytest.raises(TypeError):
            VisualizationPlugin()

    def test_concrete_viz_plugin(self):
        """Test creating concrete visualization plugin"""
        from mechanics_dsl.plugins.base import PluginMetadata, VisualizationPlugin

        class PlotlyPlugin(VisualizationPlugin):
            @property
            def metadata(self):
                return PluginMetadata(name="plotly_viz")

            def activate(self):
                pass

            def deactivate(self):
                pass

            def validate(self):
                return True

            def render(self, solution, **kwargs):
                return {"type": "plotly", "data": solution}

            def get_visualization_name(self):
                return "plotly"

            def get_supported_systems(self):
                return []  # All systems

        plugin = PlotlyPlugin()
        assert plugin.get_visualization_name() == "plotly"
        result = plugin.render({"t": [0, 1], "y": [[1, 2]]})
        assert result["type"] == "plotly"


class TestSolverPlugin:
    """Tests for SolverPlugin"""

    def test_solver_plugin_abstract(self):
        """Test SolverPlugin is abstract"""
        from mechanics_dsl.plugins.base import SolverPlugin

        with pytest.raises(TypeError):
            SolverPlugin()

    def test_concrete_solver_plugin(self):
        """Test creating concrete solver plugin"""
        from mechanics_dsl.plugins.base import PluginMetadata, SolverPlugin

        class SymplecticPlugin(SolverPlugin):
            @property
            def metadata(self):
                return PluginMetadata(name="symplectic")

            def activate(self):
                pass

            def deactivate(self):
                pass

            def validate(self):
                return True

            def solve(self, f, t_span, y0, **kwargs):
                return {"t": [0, 1], "y": [y0, y0]}

            def get_solver_name(self):
                return "symplectic4"

            def supports_stiff(self):
                return False

            def supports_events(self):
                return False

        plugin = SymplecticPlugin()
        assert plugin.get_solver_name() == "symplectic4"
        assert plugin.supports_stiff() is False


class TestPluginRegistry:
    """Tests for PluginRegistry"""

    def test_registry_singleton(self):
        """Test registry is singleton"""
        from mechanics_dsl.plugins.registry import PluginRegistry

        r1 = PluginRegistry()
        r2 = PluginRegistry()
        assert r1 is r2

    def test_registry_register_plugin(self):
        """Test registering a plugin"""
        from mechanics_dsl.plugins.base import Plugin, PluginMetadata
        from mechanics_dsl.plugins.registry import PluginRegistry, PluginType

        class TestPlugin(Plugin):
            @property
            def metadata(self):
                return PluginMetadata(name="test_reg")

            def activate(self):
                pass

            def deactivate(self):
                pass

            def validate(self):
                return True

        registry = PluginRegistry()
        registry.clear()  # Clean state
        registry.register(PluginType.DOMAIN, "test_reg", TestPlugin)

        plugins = registry.list(PluginType.DOMAIN)
        assert "test_reg" in plugins

        registry.clear()

    def test_registry_unregister(self):
        """Test unregistering a plugin"""
        from mechanics_dsl.plugins.base import Plugin, PluginMetadata
        from mechanics_dsl.plugins.registry import PluginRegistry, PluginType

        class TestPlugin(Plugin):
            @property
            def metadata(self):
                return PluginMetadata(name="unreg")

            def activate(self):
                pass

            def deactivate(self):
                pass

            def validate(self):
                return True

        registry = PluginRegistry()
        registry.clear()
        registry.register(PluginType.GENERATOR, "unreg", TestPlugin)
        assert "unreg" in registry.list(PluginType.GENERATOR)

        result = registry.unregister(PluginType.GENERATOR, "unreg")
        assert result is True
        assert "unreg" not in registry.list(PluginType.GENERATOR)

        registry.clear()

    def test_registry_get_plugin(self):
        """Test getting a plugin instance"""
        from mechanics_dsl.plugins.base import Plugin, PluginMetadata
        from mechanics_dsl.plugins.registry import PluginRegistry, PluginType

        class GetPlugin(Plugin):
            @property
            def metadata(self):
                return PluginMetadata(name="get_test")

            def activate(self):
                pass

            def deactivate(self):
                pass

            def validate(self):
                return True

        registry = PluginRegistry()
        registry.clear()
        registry.register(PluginType.SOLVER, "get_test", GetPlugin)

        instance = registry.get(PluginType.SOLVER, "get_test")
        assert instance is not None
        assert instance.metadata.name == "get_test"

        registry.clear()

    def test_registry_enable_disable(self):
        """Test enabling/disabling plugins"""
        from mechanics_dsl.plugins.base import Plugin, PluginMetadata
        from mechanics_dsl.plugins.registry import PluginRegistry, PluginType

        class TogglePlugin(Plugin):
            @property
            def metadata(self):
                return PluginMetadata(name="toggle")

            def activate(self):
                pass

            def deactivate(self):
                pass

            def validate(self):
                return True

        registry = PluginRegistry()
        registry.clear()
        registry.register(PluginType.VISUALIZATION, "toggle", TogglePlugin)

        registry.disable(PluginType.VISUALIZATION, "toggle")
        registry.enable(PluginType.VISUALIZATION, "toggle")

        registry.clear()

    def test_registry_list_all(self):
        """Test listing all plugins"""
        from mechanics_dsl.plugins.registry import PluginRegistry, PluginType

        registry = PluginRegistry()
        registry.clear()

        all_plugins = registry.list_all()
        assert isinstance(all_plugins, dict)

        registry.clear()

    def test_registry_hooks(self):
        """Test registry hooks"""
        from mechanics_dsl.plugins.registry import PluginRegistry

        registry = PluginRegistry()

        hook_called = []

        def my_hook(*args, **kwargs):
            hook_called.append(True)

        registry.add_hook("custom_event", my_hook)
        registry._fire_hook("custom_event")

        assert len(hook_called) == 1


class TestPluginLoader:
    """Tests for PluginLoader"""

    def test_loader_creation(self):
        """Test loader creation"""
        from mechanics_dsl.plugins.loader import PluginLoader

        loader = PluginLoader()
        assert loader is not None

    def test_loader_with_custom_registry(self):
        """Test loader with custom registry"""
        from mechanics_dsl.plugins.loader import PluginLoader
        from mechanics_dsl.plugins.registry import PluginRegistry

        registry = PluginRegistry()
        loader = PluginLoader(registry=registry)
        assert loader is not None

    def test_loader_get_loaded_sources(self):
        """Test getting loaded sources"""
        from mechanics_dsl.plugins.loader import PluginLoader

        loader = PluginLoader()
        sources = loader.get_loaded_sources()
        assert isinstance(sources, list)

    def test_load_nonexistent_directory(self):
        """Test loading from nonexistent directory"""
        from mechanics_dsl.plugins.loader import PluginLoader

        loader = PluginLoader()
        result = loader.load_directory("/nonexistent/path")
        assert isinstance(result, dict)

    def test_load_entry_points_mocked(self):
        """Test loading entry points with mocking"""
        from mechanics_dsl.plugins.loader import PluginLoader

        loader = PluginLoader()
        with patch("importlib.metadata.entry_points") as mock_ep:
            mock_ep.return_value = {}
            result = loader.load_entry_points()
            assert isinstance(result, dict)

    def test_classify_plugin(self):
        """Test plugin classification"""
        from mechanics_dsl.plugins.base import CodeGeneratorPlugin, PhysicsDomainPlugin
        from mechanics_dsl.plugins.loader import PluginLoader
        from mechanics_dsl.plugins.registry import PluginType

        loader = PluginLoader()

        # Mock a domain plugin class
        mock_domain = type(
            "MockDomain",
            (PhysicsDomainPlugin,),
            {
                "metadata": property(lambda s: None),
                "activate": lambda s: None,
                "deactivate": lambda s: None,
                "validate": lambda s: True,
                "get_domain_class": lambda s: None,
                "get_domain_name": lambda s: "mock",
                "get_default_parameters": lambda s: {},
            },
        )

        result = loader._classify_plugin(mock_domain)
        assert result == PluginType.DOMAIN


class TestPluginLoaderFile:
    """Tests for loading plugins from files"""

    def test_load_plugin_file(self):
        """Test loading plugin from a Python file"""
        from mechanics_dsl.plugins.loader import PluginLoader

        loader = PluginLoader()

        # Create a temp plugin file
        plugin_code = """
from mechanics_dsl.plugins.base import Plugin, PluginMetadata

class TempPlugin(Plugin):
    @property
    def metadata(self):
        return PluginMetadata(name="temp")
    
    def activate(self):
        pass
    
    def deactivate(self):
        pass
    
    def validate(self):
        return True
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(plugin_code)
            temp_path = f.name

        try:
            result = loader.load_file(temp_path)
            assert isinstance(result, dict)
        finally:
            os.unlink(temp_path)


class TestPluginDecorators:
    """Tests for plugin registration decorators"""

    def test_register_domain_decorator(self):
        """Test @register_domain decorator"""
        from mechanics_dsl.plugins.base import PhysicsDomainPlugin, PluginMetadata
        from mechanics_dsl.plugins.registry import PluginRegistry, PluginType, register_domain

        registry = PluginRegistry()
        registry.clear()

        @register_domain("test_acoustics")
        class TestAcoustics(PhysicsDomainPlugin):
            @property
            def metadata(self):
                return PluginMetadata(name="test_acoustics")

            def activate(self):
                pass

            def deactivate(self):
                pass

            def validate(self):
                return True

            def get_domain_class(self):
                return None

            def get_domain_name(self):
                return "test_acoustics"

            def get_default_parameters(self):
                return {}

        plugins = registry.list(PluginType.DOMAIN)
        assert "test_acoustics" in plugins

        registry.clear()

    def test_register_generator_decorator(self):
        """Test @register_generator decorator"""
        from mechanics_dsl.plugins.base import CodeGeneratorPlugin, PluginMetadata
        from mechanics_dsl.plugins.registry import PluginRegistry, PluginType, register_generator

        registry = PluginRegistry()
        registry.clear()

        @register_generator("test_kotlin")
        class TestKotlin(CodeGeneratorPlugin):
            @property
            def metadata(self):
                return PluginMetadata(name="test_kotlin")

            def activate(self):
                pass

            def deactivate(self):
                pass

            def validate(self):
                return True

            def get_generator_class(self):
                return None

            def get_target_name(self):
                return "kotlin"

            def get_file_extension(self):
                return ".kt"

            def get_template_path(self):
                return None

        plugins = registry.list(PluginType.GENERATOR)
        assert "test_kotlin" in plugins

        registry.clear()

    def test_register_visualization_decorator(self):
        """Test @register_visualization decorator"""
        from mechanics_dsl.plugins.base import PluginMetadata, VisualizationPlugin
        from mechanics_dsl.plugins.registry import (
            PluginRegistry,
            PluginType,
            register_visualization,
        )

        registry = PluginRegistry()
        registry.clear()

        @register_visualization("test_viz")
        class TestViz(VisualizationPlugin):
            @property
            def metadata(self):
                return PluginMetadata(name="test_viz")

            def activate(self):
                pass

            def deactivate(self):
                pass

            def validate(self):
                return True

            def render(self, solution, **kwargs):
                return None

            def get_visualization_name(self):
                return "test_viz"

            def get_supported_systems(self):
                return []

        plugins = registry.list(PluginType.VISUALIZATION)
        assert "test_viz" in plugins

        registry.clear()
