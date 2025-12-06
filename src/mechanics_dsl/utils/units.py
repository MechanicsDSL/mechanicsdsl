"""
Unit system for MechanicsDSL with dimensional analysis
"""
import ast
import operator
import numpy as np
from typing import Dict, Union
from dataclasses import dataclass, field

from .logging import logger


@dataclass
class Unit:
    """Physical unit with dimensional analysis"""
    dimensions: Dict[str, int] = field(default_factory=dict)
    scale: float = 1.0

    def __mul__(self, other: Union['Unit', float, int]) -> 'Unit':
        if isinstance(other, (int, float)):
            return Unit(self.dimensions.copy(), self.scale * other)
        result = {}
        all_dims = set(self.dimensions.keys()) | set(other.dimensions.keys())
        for dim in all_dims:
            result[dim] = self.dimensions.get(dim, 0) + other.dimensions.get(dim, 0)
            if result[dim] == 0:
                del result[dim]
        return Unit(result, self.scale * other.scale)

    def __rmul__(self, other: Union[float, int]) -> 'Unit':
        return self.__mul__(other)

    def __truediv__(self, other: Union['Unit', float, int]) -> 'Unit':
        if isinstance(other, (int, float)):
            return Unit(self.dimensions.copy(), self.scale / other)
        result = {}
        all_dims = set(self.dimensions.keys()) | set(other.dimensions.keys())
        for dim in all_dims:
            result[dim] = self.dimensions.get(dim, 0) - other.dimensions.get(dim, 0)
            if result[dim] == 0:
                del result[dim]
        return Unit(result, self.scale / other.scale)

    def __pow__(self, exponent: float) -> 'Unit':
        result = {dim: power * exponent for dim, power in self.dimensions.items()}
        return Unit(result, self.scale ** exponent)

    def is_compatible(self, other: 'Unit') -> bool:
        """Check if units are dimensionally compatible"""
        return self.dimensions == other.dimensions

    def __repr__(self) -> str:
        if not self.dimensions:
            return f"Unit(dimensionless, scale={self.scale})"
        return f"Unit({self.dimensions}, scale={self.scale})"


# Comprehensive unit system
BASE_UNITS = {
    "dimensionless": Unit({}),
    "1": Unit({}),
    
    # SI Base units
    "m": Unit({"length": 1}),
    "kg": Unit({"mass": 1}),
    "s": Unit({"time": 1}),
    "A": Unit({"current": 1}),
    "K": Unit({"temperature": 1}),
    "mol": Unit({"substance": 1}),
    "cd": Unit({"luminous_intensity": 1}),
    
    # Common derived units
    "N": Unit({"mass": 1, "length": 1, "time": -2}),
    "J": Unit({"mass": 1, "length": 2, "time": -2}),
    "W": Unit({"mass": 1, "length": 2, "time": -3}),
    "Pa": Unit({"mass": 1, "length": -1, "time": -2}),
    "Hz": Unit({"time": -1}),
    "C": Unit({"current": 1, "time": 1}),
    "V": Unit({"mass": 1, "length": 2, "time": -3, "current": -1}),
    "F": Unit({"mass": -1, "length": -2, "time": 4, "current": 2}),
    "Wb": Unit({"mass": 1, "length": 2, "time": -2, "current": -1}),
    "T": Unit({"mass": 1, "time": -2, "current": -1}),
    
    # Angle units
    "rad": Unit({"angle": 1}),
    "deg": Unit({"angle": 1}, scale=np.pi/180),
}


class UnitSystem:
    """
    Manages unit operations and conversions with safe parsing.
    
    Uses AST-based parsing instead of eval() for security.
    """
    
    def __init__(self) -> None:
        """Initialize unit system with base units."""
        self.units: Dict[str, Unit] = BASE_UNITS.copy()
    
    def _parse_unit_expression(self, expr: str) -> Unit:
        """
        Safely parse unit expression using AST.
        
        Args:
            expr: Unit expression string (e.g., 'kg*m/s^2')
            
        Returns:
            Parsed Unit object
            
        Raises:
            ValueError: If expression is invalid or contains unknown units
        """
        # Replace ^ with ** for Python syntax
        expr = expr.replace('^', '**')
        
        try:
            # Parse as AST expression
            tree = ast.parse(expr, mode='eval')
        except SyntaxError as e:
            raise ValueError(f"Invalid unit expression syntax: {expr}") from e
        
        # Safe operators mapping
        ops = {
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
        }
        
        def eval_node(node: ast.AST) -> Unit:
            """Recursively evaluate AST node."""
            if isinstance(node, ast.Name):
                # Look up unit name
                unit_name = node.id
                if unit_name not in self.units:
                    raise ValueError(f"Unknown unit: {unit_name}")
                return self.units[unit_name]
            
            elif isinstance(node, ast.BinOp):
                # Binary operation
                left = eval_node(node.left)
                right = eval_node(node.right)
                op_func = ops.get(type(node.op))
                if op_func is None:
                    raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
                
                # Special handling for power operation: if right is dimensionless Unit (numeric constant),
                # extract its scale to use as float exponent
                if isinstance(node.op, ast.Pow) and isinstance(right, Unit):
                    if not right.dimensions:  # Dimensionless = numeric constant
                        return left ** right.scale
                
                return op_func(left, right)
            
            elif isinstance(node, ast.Constant):
                # Numeric constant (for scaling)
                value = node.value
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Expected numeric constant, got {type(value).__name__}")
                return Unit({}, scale=float(value))
            
            else:
                raise ValueError(f"Unsupported AST node type: {type(node).__name__}")
        
        return eval_node(tree.body)
    
    def parse_unit(self, unit_str: str) -> Unit:
        """
        Parse unit string like 'kg*m/s^2' into Unit object.
        
        Uses safe AST-based parsing instead of eval() for security.
        
        Args:
            unit_str: Unit string to parse
            
        Returns:
            Unit object (returns dimensionless unit on error)
            
        Raises:
            TypeError: If unit_str is not a string
        """
        if not isinstance(unit_str, str):
            raise TypeError(f"unit_str must be str, got {type(unit_str).__name__}")
        
        unit_str = unit_str.strip()
        if not unit_str:
            logger.warning("Empty unit string, returning dimensionless unit")
            return Unit({})
        
        # Direct lookup
        if unit_str in self.units:
            return self.units[unit_str]
        
        # Parse expression
        try:
            if '*' in unit_str or '/' in unit_str or '^' in unit_str or '**' in unit_str:
                return self._parse_unit_expression(unit_str)
            else:
                # Unknown simple unit
                logger.warning(f"Unknown unit: {unit_str}, returning dimensionless unit")
                return Unit({})
        except (ValueError, SyntaxError) as e:
            logger.warning(f"Could not parse unit '{unit_str}': {e}")
            return Unit({})
    
    def check_compatibility(self, unit1: str, unit2: str) -> bool:
        """
        Check if two units are dimensionally compatible.
        
        Args:
            unit1: First unit string
            unit2: Second unit string
            
        Returns:
            True if units are compatible, False otherwise
            
        Raises:
            TypeError: If inputs are not strings
        """
        if not isinstance(unit1, str) or not isinstance(unit2, str):
            raise TypeError("Unit strings must be str")
        
        u1 = self.parse_unit(unit1)
        u2 = self.parse_unit(unit2)
        return u1.is_compatible(u2)
