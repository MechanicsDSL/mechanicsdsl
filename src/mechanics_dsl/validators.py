"""
MechanicsDSL Validators Module
==============================

Centralized validation utilities for runtime type checking and input validation.
Uses Pydantic for structured validation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import math

try:
    from pydantic import BaseModel, Field, validator, root_validator
    from pydantic import ValidationError as PydanticValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


# =============================================================================
# Simulation Configuration
# =============================================================================

if PYDANTIC_AVAILABLE:
    class SimulationConfig(BaseModel):
        """Validated simulation configuration."""
        
        t_start: float = Field(default=0.0, ge=0.0, description="Start time")
        t_end: float = Field(default=10.0, gt=0.0, description="End time")
        dt: float = Field(default=0.001, gt=0.0, le=1.0, description="Time step")
        num_points: int = Field(default=1000, ge=10, le=1000000, description="Output points")
        method: str = Field(default="RK45", description="Integration method")
        rtol: float = Field(default=1e-6, gt=0, le=1, description="Relative tolerance")
        atol: float = Field(default=1e-9, gt=0, le=1, description="Absolute tolerance")
        max_step: Optional[float] = Field(default=None, gt=0, description="Maximum step size")
        
        @validator('t_end')
        def t_end_after_t_start(cls, v, values):
            if 't_start' in values and v <= values['t_start']:
                raise ValueError('t_end must be greater than t_start')
            return v
        
        @validator('method')
        def valid_method(cls, v):
            valid = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA', 'euler', 'rk4']
            if v not in valid:
                raise ValueError(f'method must be one of {valid}')
            return v
        
        class Config:
            extra = 'forbid'  # Reject unknown fields


    class CoordinateConfig(BaseModel):
        """Validated coordinate configuration."""
        
        name: str = Field(..., min_length=1, max_length=64, regex=r'^[a-zA-Z_][a-zA-Z0-9_]*$')
        description: str = Field(default="", max_length=256)
        unit: str = Field(default="", max_length=32)
        initial_value: float = Field(default=0.0)
        initial_velocity: float = Field(default=0.0)
        bounds: Optional[Tuple[float, float]] = None
        
        @validator('bounds')
        def valid_bounds(cls, v):
            if v is not None:
                if v[0] >= v[1]:
                    raise ValueError('Lower bound must be less than upper bound')
            return v


    class ParameterConfig(BaseModel):
        """Validated parameter configuration."""
        
        name: str = Field(..., min_length=1, max_length=64, regex=r'^[a-zA-Z_][a-zA-Z0-9_]*$')
        value: float = Field(...)
        description: str = Field(default="", max_length=256)
        unit: str = Field(default="", max_length=32)
        min_value: Optional[float] = None
        max_value: Optional[float] = None
        
        @root_validator
        def value_in_bounds(cls, values):
            val = values.get('value')
            min_val = values.get('min_value')
            max_val = values.get('max_value')
            
            if val is not None:
                if min_val is not None and val < min_val:
                    raise ValueError(f'value {val} below min {min_val}')
                if max_val is not None and val > max_val:
                    raise ValueError(f'value {val} above max {max_val}')
            
            return values


    class CodegenConfig(BaseModel):
        """Validated code generation configuration."""
        
        target: str = Field(..., description="Target language")
        output_dir: str = Field(default=".", description="Output directory")
        system_name: str = Field(..., min_length=1, max_length=64)
        generate_cmake: bool = Field(default=True)
        optimization_level: int = Field(default=2, ge=0, le=3)
        use_simd: bool = Field(default=True)
        embedded: bool = Field(default=False)
        
        @validator('target')
        def valid_target(cls, v):
            valid = ['cpp', 'rust', 'cuda', 'arm', 'julia', 'matlab', 
                     'fortran', 'javascript', 'wasm', 'arduino', 'python']
            if v.lower() not in valid:
                raise ValueError(f'target must be one of {valid}')
            return v.lower()


    class ServerConfig(BaseModel):
        """Validated server configuration."""
        
        host: str = Field(default="0.0.0.0")
        port: int = Field(default=8000, ge=1, le=65535)
        workers: int = Field(default=4, ge=1, le=32)
        timeout: int = Field(default=300, ge=1, le=3600)
        max_connections: int = Field(default=1000, ge=1, le=10000)
        cors_origins: List[str] = Field(default_factory=list)
        enable_docs: bool = Field(default=True)
        
        @validator('host')
        def valid_host(cls, v):
            import socket
            try:
                socket.inet_aton(v)
            except socket.error:
                if v not in ['localhost', '0.0.0.0']:
                    raise ValueError(f'Invalid host: {v}')
            return v

else:
    # Fallback dataclasses when Pydantic not available
    @dataclass
    class SimulationConfig:
        t_start: float = 0.0
        t_end: float = 10.0
        dt: float = 0.001
        num_points: int = 1000
        method: str = "RK45"
        rtol: float = 1e-6
        atol: float = 1e-9
        max_step: Optional[float] = None
    
    @dataclass
    class CoordinateConfig:
        name: str = ""
        description: str = ""
        unit: str = ""
        initial_value: float = 0.0
        initial_velocity: float = 0.0
        bounds: Optional[Tuple[float, float]] = None
    
    @dataclass
    class ParameterConfig:
        name: str = ""
        value: float = 0.0
        description: str = ""
        unit: str = ""
        min_value: Optional[float] = None
        max_value: Optional[float] = None
    
    @dataclass
    class CodegenConfig:
        target: str = "cpp"
        output_dir: str = "."
        system_name: str = "simulation"
        generate_cmake: bool = True
        optimization_level: int = 2
        use_simd: bool = True
        embedded: bool = False
    
    @dataclass
    class ServerConfig:
        host: str = "0.0.0.0"
        port: int = 8000
        workers: int = 4
        timeout: int = 300
        max_connections: int = 1000
        cors_origins: List[str] = field(default_factory=list)
        enable_docs: bool = True


# =============================================================================
# Validation Functions
# =============================================================================

def validate_simulation_config(config: Dict[str, Any]) -> SimulationConfig:
    """Validate and create SimulationConfig from dictionary."""
    if PYDANTIC_AVAILABLE:
        return SimulationConfig(**config)
    else:
        return SimulationConfig(**{k: v for k, v in config.items() 
                                   if k in SimulationConfig.__dataclass_fields__})


def validate_coordinate(name: str, value: float, velocity: float = 0.0,
                        bounds: Optional[Tuple[float, float]] = None) -> CoordinateConfig:
    """Validate coordinate configuration."""
    if PYDANTIC_AVAILABLE:
        return CoordinateConfig(
            name=name,
            initial_value=value,
            initial_velocity=velocity,
            bounds=bounds
        )
    else:
        return CoordinateConfig(
            name=name,
            initial_value=value,
            initial_velocity=velocity,
            bounds=bounds
        )


def validate_parameter(name: str, value: float,
                       min_value: Optional[float] = None,
                       max_value: Optional[float] = None) -> ParameterConfig:
    """Validate parameter configuration."""
    if PYDANTIC_AVAILABLE:
        return ParameterConfig(
            name=name,
            value=value,
            min_value=min_value,
            max_value=max_value
        )
    else:
        # Manual validation
        if min_value is not None and value < min_value:
            raise ValueError(f'Parameter {name} ({value}) below minimum ({min_value})')
        if max_value is not None and value > max_value:
            raise ValueError(f'Parameter {name} ({value}) above maximum ({max_value})')
        
        return ParameterConfig(
            name=name,
            value=value,
            min_value=min_value,
            max_value=max_value
        )


# =============================================================================
# Error Handling
# =============================================================================

class ValidationError(Exception):
    """Validation error with detailed information."""
    
    def __init__(self, message: str, field: Optional[str] = None, 
                 value: Any = None, errors: Optional[List[Dict]] = None):
        super().__init__(message)
        self.field = field
        self.value = value
        self.errors = errors or []
    
    def to_dict(self) -> Dict:
        return {
            'message': str(self),
            'field': self.field,
            'value': self.value,
            'errors': self.errors
        }


def wrap_validation_error(func):
    """Decorator to wrap Pydantic validation errors."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if PYDANTIC_AVAILABLE and isinstance(e, PydanticValidationError):
                raise ValidationError(
                    str(e),
                    errors=[err for err in e.errors()]
                )
            raise
    return wrapper
