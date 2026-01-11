"""
Protocol definitions for MechanicsDSL.

This module defines Protocol classes for structural typing, enabling
duck-typing with full type-checker support.
"""
from typing import Protocol, Dict, List, Tuple, Optional, Any, runtime_checkable
import numpy as np


@runtime_checkable
class SymbolicExpressionProtocol(Protocol):
    """Protocol for objects that can be used as symbolic expressions."""
    
    @property
    def free_symbols(self) -> set:
        """Return the set of free symbols in the expression."""
        ...
    
    def subs(self, substitutions: Dict) -> 'SymbolicExpressionProtocol':
        """Substitute values for symbols."""
        ...


@runtime_checkable  
class SimulatableProtocol(Protocol):
    """Protocol for objects that can be numerically simulated."""
    
    def set_parameters(self, params: Dict[str, float]) -> None:
        """Set physical parameters for the simulation."""
        ...
    
    def set_initial_conditions(self, conditions: Dict[str, float]) -> None:
        """Set initial conditions for state variables."""
        ...
    
    def simulate(
        self, 
        t_span: Tuple[float, float], 
        num_points: int = 1000
    ) -> Dict[str, Any]:
        """
        Run the simulation.
        
        Args:
            t_span: Time span (start, end)
            num_points: Number of evaluation points
            
        Returns:
            Dictionary with 't', 'y', and 'success' keys at minimum
        """
        ...


@runtime_checkable
class CompilableProtocol(Protocol):
    """Protocol for objects that can compile DSL source code."""
    
    def compile_dsl(self, source: str, **kwargs) -> Dict[str, Any]:
        """
        Compile DSL source code.
        
        Args:
            source: DSL source code string
            **kwargs: Additional compilation options
            
        Returns:
            Dictionary with 'success' key and compilation results
        """
        ...


@runtime_checkable
class VisualizableProtocol(Protocol):
    """Protocol for objects that can create visualizations."""
    
    def plot(self, solution: Dict[str, Any], **kwargs) -> Any:
        """Create a static plot of the solution."""
        ...
    
    def animate(self, solution: Dict[str, Any], **kwargs) -> Any:
        """Create an animation of the solution."""
        ...


@runtime_checkable
class CodeGeneratorProtocol(Protocol):
    """Protocol for code generation backends."""
    
    @property
    def target_name(self) -> str:
        """Name of the target platform."""
        ...
    
    @property
    def file_extension(self) -> str:
        """File extension for generated code."""
        ...
    
    def generate(self, output_file: str) -> str:
        """Generate code and write to file."""
        ...
    
    def generate_equations(self) -> str:
        """Generate the equations of motion code."""
        ...


@runtime_checkable
class PhysicsDomainProtocol(Protocol):
    """Protocol for physics domain implementations."""
    
    @property
    def name(self) -> str:
        """Human-readable domain name."""
        ...
    
    @property
    def coordinates(self) -> List[str]:
        """List of generalized coordinates."""
        ...
    
    @property
    def parameters(self) -> Dict[str, float]:
        """Dictionary of physical parameters."""
        ...
    
    def define_lagrangian(self) -> Any:
        """Define the Lagrangian for this domain."""
        ...
    
    def derive_equations_of_motion(self) -> Dict[str, Any]:
        """Derive the equations of motion."""
        ...


@runtime_checkable
class CacheProtocol(Protocol):
    """Protocol for cache implementations."""
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        ...
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the cache."""
        ...
    
    def clear(self) -> None:
        """Clear all cached values."""
        ...
    
    @property
    def hit_rate(self) -> float:
        """Return the cache hit rate."""
        ...


@runtime_checkable
class ValidatorProtocol(Protocol):
    """Protocol for input validators."""
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a value.
        
        Args:
            value: Value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        ...


__all__ = [
    'SymbolicExpressionProtocol',
    'SimulatableProtocol', 
    'CompilableProtocol',
    'VisualizableProtocol',
    'CodeGeneratorProtocol',
    'PhysicsDomainProtocol',
    'CacheProtocol',
    'ValidatorProtocol',
]
