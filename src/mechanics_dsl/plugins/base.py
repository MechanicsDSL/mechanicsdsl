"""
Plugin base classes for MechanicsDSL.

Defines abstract interfaces that all plugins must implement.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Type, Dict, Any, Optional, List, Callable
import sympy as sp


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str = "1.0.0"
    author: str = ""
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    homepage: str = ""


class Plugin(ABC):
    """
    Base class for all MechanicsDSL plugins.
    
    Plugins must provide metadata and implement activation/deactivation hooks.
    """
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    def activate(self) -> None:
        """Called when plugin is activated. Override for initialization."""
        pass
    
    def deactivate(self) -> None:
        """Called when plugin is deactivated. Override for cleanup."""
        pass
    
    def validate(self) -> bool:
        """Validate plugin configuration. Return False to prevent loading."""
        return True


class PhysicsDomainPlugin(Plugin):
    """
    Plugin interface for custom physics domains.
    
    Implement this to add new physics formulations (e.g., acoustics, optics).
    
    Example:
        class AcousticsPlugin(PhysicsDomainPlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="acoustics",
                    version="1.0.0",
                    description="Acoustic wave equation support"
                )
            
            def get_domain_class(self):
                return AcousticsDomain
            
            def get_domain_name(self):
                return "acoustics"
    """
    
    @abstractmethod
    def get_domain_class(self) -> Type:
        """
        Return the physics domain class.
        
        The class should inherit from PhysicsDomain (domains.base).
        """
        pass
    
    @abstractmethod
    def get_domain_name(self) -> str:
        """Return unique identifier for this domain."""
        pass
    
    def get_default_parameters(self) -> Dict[str, float]:
        """Return default parameters for this domain."""
        return {}


class CodeGeneratorPlugin(Plugin):
    """
    Plugin interface for custom code generators.
    
    Implement this to add new target languages or platforms.
    
    Example:
        class SwiftGeneratorPlugin(CodeGeneratorPlugin):
            @property
            def metadata(self):
                return PluginMetadata(name="swift", description="Swift code generator")
            
            def get_generator_class(self):
                return SwiftGenerator
            
            def get_target_name(self):
                return "swift"
            
            def get_file_extension(self):
                return ".swift"
    """
    
    @abstractmethod
    def get_generator_class(self) -> Type:
        """
        Return the code generator class.
        
        The class should inherit from CodeGenerator (codegen.base).
        """
        pass
    
    @abstractmethod
    def get_target_name(self) -> str:
        """Return target platform name (e.g., 'swift', 'kotlin')."""
        pass
    
    @abstractmethod
    def get_file_extension(self) -> str:
        """Return file extension for generated code (e.g., '.swift')."""
        pass
    
    def get_template_path(self) -> Optional[str]:
        """Return path to code template file, if any."""
        return None


class VisualizationPlugin(Plugin):
    """
    Plugin interface for custom visualization backends.
    
    Implement this to add new visualization types or rendering engines.
    
    Example:
        class PlotlyVisualizationPlugin(VisualizationPlugin):
            @property
            def metadata(self):
                return PluginMetadata(name="plotly", description="Plotly visualizations")
            
            def render(self, solution, **kwargs):
                import plotly.graph_objects as go
                fig = go.Figure()
                # ... build figure
                return fig
    """
    
    @abstractmethod
    def render(self, solution: Dict[str, Any], **kwargs) -> Any:
        """
        Render visualization from simulation solution.
        
        Args:
            solution: Simulation result dictionary with 't', 'y', etc.
            **kwargs: Visualization options
            
        Returns:
            Visualization object (figure, widget, HTML, etc.)
        """
        pass
    
    def get_visualization_name(self) -> str:
        """Return unique identifier for this visualization type."""
        return self.metadata.name
    
    def get_supported_systems(self) -> List[str]:
        """Return list of supported system types (empty = all)."""
        return []


class SolverPlugin(Plugin):
    """
    Plugin interface for custom ODE/DAE solvers.
    
    Implement this to add specialized integration methods.
    
    Example:
        class SymplecticSolverPlugin(SolverPlugin):
            @property
            def metadata(self):
                return PluginMetadata(name="symplectic", description="Symplectic integrator")
            
            def solve(self, equations, t_span, y0, **kwargs):
                # Symplectic integration implementation
                return {'t': t, 'y': y, 'success': True}
    """
    
    @abstractmethod
    def solve(
        self,
        equations: Callable,
        t_span: tuple,
        y0: List[float],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Solve the system of equations.
        
        Args:
            equations: Callable (t, y) -> dy/dt
            t_span: (t_start, t_end)
            y0: Initial conditions
            **kwargs: Solver options
            
        Returns:
            Solution dictionary with 't', 'y', 'success' keys
        """
        pass
    
    def get_solver_name(self) -> str:
        """Return unique identifier for this solver."""
        return self.metadata.name
    
    def supports_stiff(self) -> bool:
        """Return True if solver handles stiff systems."""
        return False
    
    def supports_events(self) -> bool:
        """Return True if solver supports event detection."""
        return False


class TransformPlugin(Plugin):
    """
    Plugin interface for symbolic transformation passes.
    
    Implement this to add custom symbolic optimizations or transformations.
    
    Example:
        class SimplifyTrigPlugin(TransformPlugin):
            def transform(self, expr):
                return sp.trigsimp(expr)
    """
    
    @abstractmethod
    def transform(self, expr: sp.Expr) -> sp.Expr:
        """
        Apply transformation to symbolic expression.
        
        Args:
            expr: SymPy expression
            
        Returns:
            Transformed expression
        """
        pass
    
    def get_transform_name(self) -> str:
        """Return unique identifier for this transform."""
        return self.metadata.name
    
    def get_priority(self) -> int:
        """Return execution priority (lower = earlier). Default 100."""
        return 100


class ImporterPlugin(Plugin):
    """
    Plugin interface for importing from external formats.
    
    Implement this to import models from Modelica, Simscape, etc.
    """
    
    @abstractmethod
    def import_file(self, file_path: str) -> str:
        """
        Import file and return MechanicsDSL code.
        
        Args:
            file_path: Path to file to import
            
        Returns:
            Equivalent MechanicsDSL code string
        """
        pass
    
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        return []


__all__ = [
    'PluginMetadata',
    'Plugin',
    'PhysicsDomainPlugin',
    'CodeGeneratorPlugin',
    'VisualizationPlugin',
    'SolverPlugin',
    'TransformPlugin',
    'ImporterPlugin',
]
