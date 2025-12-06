"""
Base class for physics domains in MechanicsDSL

This module provides the abstract interface that all physics domains must implement.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import sympy as sp


class PhysicsDomain(ABC):
    """
    Abstract base class for physics domain implementations.
    
    All domain-specific physics (classical mechanics, fluids, quantum, etc.)
    should inherit from this class and implement the required methods.
    
    Attributes:
        name: Human-readable domain name
        coordinates: List of generalized coordinates
        parameters: Dictionary of physical parameters
    """
    
    def __init__(self, name: str = "unnamed"):
        self.name = name
        self.coordinates: List[str] = []
        self.parameters: Dict[str, float] = {}
        self.equations: Dict[str, sp.Expr] = {}
        self._is_compiled: bool = False
    
    @property
    def is_compiled(self) -> bool:
        """Whether the domain has been compiled and is ready for simulation."""
        return self._is_compiled
    
    @abstractmethod
    def define_lagrangian(self) -> sp.Expr:
        """
        Define the Lagrangian for this physics domain.
        
        Returns:
            SymPy expression for L = T - V
        """
        pass
    
    @abstractmethod
    def define_hamiltonian(self) -> sp.Expr:
        """
        Define the Hamiltonian for this physics domain.
        
        Returns:
            SymPy expression for H = T + V (in generalized coordinates)
        """
        pass
    
    @abstractmethod
    def derive_equations_of_motion(self) -> Dict[str, sp.Expr]:
        """
        Derive the equations of motion for this domain.
        
        Returns:
            Dictionary mapping acceleration variables to their expressions
        """
        pass
    
    @abstractmethod
    def get_state_variables(self) -> List[str]:
        """
        Get the list of state variables for the ODE system.
        
        Returns:
            List of variable names (positions and velocities)
        """
        pass
    
    def set_parameter(self, name: str, value: float) -> None:
        """Set a physical parameter value."""
        self.parameters[name] = value
    
    def set_parameters(self, params: Dict[str, float]) -> None:
        """Set multiple physical parameters."""
        self.parameters.update(params)
    
    def add_coordinate(self, name: str) -> None:
        """Add a generalized coordinate."""
        if name not in self.coordinates:
            self.coordinates.append(name)
    
    def get_default_initial_conditions(self) -> Dict[str, float]:
        """
        Get default initial conditions for simulation.
        
        Returns:
            Dictionary mapping state variables to initial values
        """
        ic = {}
        for q in self.coordinates:
            ic[q] = 0.0
            ic[f"{q}_dot"] = 0.0
        return ic
    
    def validate_parameters(self) -> Tuple[bool, List[str]]:
        """
        Validate that all required parameters are set.
        
        Returns:
            Tuple of (is_valid, list of missing parameter names)
        """
        required = self.get_required_parameters()
        missing = [p for p in required if p not in self.parameters]
        return len(missing) == 0, missing
    
    def get_required_parameters(self) -> List[str]:
        """
        Get list of required parameters for this domain.
        
        Override in subclasses to specify required parameters.
        
        Returns:
            List of required parameter names
        """
        return []
    
    def get_conserved_quantities(self) -> Dict[str, sp.Expr]:
        """
        Get expressions for conserved quantities (energy, momentum, etc.).
        
        Override in subclasses to provide domain-specific conserved quantities.
        
        Returns:
            Dictionary mapping quantity names to SymPy expressions
        """
        return {}
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', coordinates={self.coordinates})"
