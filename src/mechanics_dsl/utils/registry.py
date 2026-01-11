"""
Variable type registry for MechanicsDSL.

This module defines which variable types are dynamic coordinates vs. constants,
replacing the hardcoded lists scattered throughout the codebase.
"""
from enum import Enum, auto
from typing import Set, FrozenSet


class VariableCategory(Enum):
    """Categories of variables in the physics system."""
    COORDINATE = auto()      # Dynamic generalized coordinates (q)
    VELOCITY = auto()        # Velocity (q_dot) - derived from coordinates
    CONSTANT = auto()        # Physical constants (g, c, etc.)
    PARAMETER = auto()       # User-defined parameters (m, k, l, etc.)
    EXTERNAL_FIELD = auto()  # External field variables (E, B, etc.)


# Variable types that represent dynamic coordinates
COORDINATE_TYPES: FrozenSet[str] = frozenset({
    'Angle',
    'Position', 
    'Coordinate',
    'Displacement',
    'Generalized Coordinate',
})

# Variable types that represent constants/parameters (not coordinates)
CONSTANT_TYPES: FrozenSet[str] = frozenset({
    'Constant',
    'Parameter',
    'Mass',
    'Length', 
    'Spring Constant',
    'Damping Coeff',
    'Damping Coefficient',
    'Damping Parameter',
    'Acceleration',
    'Gravitational Constant',
    'Electric Field',
    'Magnetic Field',
    'Charge',
    'Angular Velocity',
    'Drive Frequency',
    'Force Amplitude',
    'Coupling Constant',
    'Moment of Inertia',
    'Moment of Inertia 1',
    'Moment of Inertia 3',
    'Spin Rate',
    'Radius',
    'Incline Angle',
    'Natural Frequency',
    'Real',
    'Time Constant',
    'Amplitude',
    'Frequency',
    'Phase',
})

# Common coordinate variable names (fallback if type detection fails)
COMMON_COORDINATE_NAMES: FrozenSet[str] = frozenset({
    # Angular coordinates
    'theta', 'theta1', 'theta2', 'theta3', 'theta4',
    'phi', 'phi1', 'phi2',
    'psi', 'psi1', 'psi2',
    'alpha', 'beta', 'gamma',
    
    # Cartesian coordinates
    'x', 'x1', 'x2', 'x3',
    'y', 'y1', 'y2', 'y3',
    'z', 'z1', 'z2', 'z3',
    
    # Spherical/cylindrical
    'r', 'r1', 'r2',
    'rho',
    
    # Generalized
    'q', 'q1', 'q2', 'q3', 'q4',
    's', 's1', 's2',
})

# Common parameter/constant variable names
COMMON_CONSTANT_NAMES: FrozenSet[str] = frozenset({
    # Physical constants
    'g', 'c', 'G', 'e', 'h', 'hbar', 'epsilon0', 'mu0', 'k_B',
    
    # Mass parameters
    'm', 'm1', 'm2', 'm3', 'M',
    
    # Length parameters
    'l', 'l1', 'l2', 'l3', 'L', 'a', 'b', 'd',
    
    # Spring/damping
    'k', 'k1', 'k2', 'c', 'c1', 'c2', 'gamma', 'b',
    
    # Frequency/angular
    'omega', 'omega0', 'omega_d', 'f', 'T',
    
    # Field quantities
    'E', 'B', 'V', 'q',
    
    # Moments of inertia
    'I', 'I1', 'I2', 'I3',
})


def is_coordinate_type(var_type: str) -> bool:
    """
    Check if a variable type represents a dynamic coordinate.
    
    Args:
        var_type: The type string from VarDef
        
    Returns:
        True if this type represents a coordinate, False otherwise
    """
    return var_type in COORDINATE_TYPES


def is_constant_type(var_type: str) -> bool:
    """
    Check if a variable type represents a constant/parameter.
    
    Args:
        var_type: The type string from VarDef
        
    Returns:
        True if this type represents a constant, False otherwise
    """
    return var_type in CONSTANT_TYPES


def is_likely_coordinate(var_name: str, var_type: str) -> bool:
    """
    Determine if a variable is likely a dynamic coordinate.
    
    Uses both type information and naming conventions to determine
    if a variable represents a generalized coordinate.
    
    Args:
        var_name: The variable name
        var_type: The variable type string
        
    Returns:
        True if this is likely a coordinate, False otherwise
    """
    # Explicit type takes precedence
    if is_constant_type(var_type):
        return False
    if is_coordinate_type(var_type):
        return True
    
    # Fall back to name-based detection
    return var_name in COMMON_COORDINATE_NAMES


def classify_variable(var_name: str, var_type: str) -> VariableCategory:
    """
    Classify a variable into its category.
    
    Args:
        var_name: The variable name
        var_type: The variable type string
        
    Returns:
        VariableCategory enum value
    """
    if is_constant_type(var_type):
        return VariableCategory.PARAMETER
    elif is_coordinate_type(var_type):
        return VariableCategory.COORDINATE
    elif var_name in COMMON_CONSTANT_NAMES:
        return VariableCategory.PARAMETER
    elif var_name in COMMON_COORDINATE_NAMES:
        return VariableCategory.COORDINATE
    else:
        # Default to parameter for unknown types
        return VariableCategory.PARAMETER


__all__ = [
    'VariableCategory',
    'COORDINATE_TYPES',
    'CONSTANT_TYPES', 
    'COMMON_COORDINATE_NAMES',
    'COMMON_CONSTANT_NAMES',
    'is_coordinate_type',
    'is_constant_type',
    'is_likely_coordinate',
    'classify_variable',
]
