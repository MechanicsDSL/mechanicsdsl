"""
Base Code Generator for MechanicsDSL

Provides abstract interface for all code generation backends.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import sympy as sp


class CodeGenerator(ABC):
    """
    Abstract base class for code generation backends.
    
    All code generators (C++, Python, WASM, CUDA, etc.) should inherit
    from this class and implement the required methods.
    
    Attributes:
        system_name: Name of the physics system
        coordinates: List of generalized coordinates
        parameters: Physical parameters dictionary
        equations: Symbolic equations of motion
    """
    
    def __init__(self, 
                 system_name: str,
                 coordinates: List[str],
                 parameters: Dict[str, float],
                 initial_conditions: Dict[str, float],
                 equations: Dict[str, sp.Expr]):
        self.system_name = system_name
        self.coordinates = coordinates
        self.parameters = parameters
        self.initial_conditions = initial_conditions
        self.equations = equations or {}
    
    @property
    @abstractmethod
    def target_name(self) -> str:
        """Name of the target platform (e.g., 'cpp', 'python', 'wasm')."""
        pass
    
    @property
    @abstractmethod
    def file_extension(self) -> str:
        """File extension for generated code (e.g., '.cpp', '.py')."""
        pass
    
    @abstractmethod
    def generate(self, output_file: str) -> str:
        """
        Generate code and write to file.
        
        Args:
            output_file: Output file path
            
        Returns:
            Path to generated file
        """
        pass
    
    @abstractmethod
    def generate_equations(self) -> str:
        """
        Generate the equations of motion code.
        
        Returns:
            String containing equation code
        """
        pass
    
    def generate_parameters(self) -> str:
        """
        Generate parameter declarations.
        
        Override in subclasses for target-specific syntax.
        """
        lines = []
        for name, val in self.parameters.items():
            lines.append(f"// {name} = {val}")
        return "\n".join(lines)
    
    def generate_initial_conditions(self) -> str:
        """
        Generate initial condition setup code.
        
        Override in subclasses for target-specific syntax.
        """
        vals = []
        for coord in self.coordinates:
            pos = self.initial_conditions.get(coord, 0.0)
            vel = self.initial_conditions.get(f"{coord}_dot", 0.0)
            vals.extend([str(pos), str(vel)])
        return ", ".join(vals)
    
    def validate(self) -> tuple:
        """
        Validate that the generator has all required data.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not self.system_name:
            errors.append("system_name is required")
        if not self.coordinates:
            errors.append("at least one coordinate is required")
        
        # Check that all coordinates have equations
        for coord in self.coordinates:
            accel_key = f"{coord}_ddot"
            if accel_key not in self.equations:
                errors.append(f"missing equation for {accel_key}")
        
        return len(errors) == 0, errors
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(system='{self.system_name}', target='{self.target_name}')"
