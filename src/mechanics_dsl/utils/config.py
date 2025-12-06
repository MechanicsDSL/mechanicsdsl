"""
Configuration management for MechanicsDSL
"""
import logging
from typing import Dict, Any, Tuple

from .logging import logger

# ============================================================================
# CONSTANTS
# ============================================================================

# Numerical constants
DEFAULT_TRAIL_LENGTH = 150
DEFAULT_FPS = 30
ENERGY_TOLERANCE = 0.01
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-8
SIMPLIFICATION_TIMEOUT = 5.0  # seconds
MAX_PARSER_ERRORS = 10

# Animation constants
ANIMATION_INTERVAL_MS = 33  # ~30 FPS
TRAIL_ALPHA = 0.4
PRIMARY_COLOR = '#E63946'
SECONDARY_COLOR = '#457B9D'
TERTIARY_COLOR = '#F1FAEE'


class Config:
    """
    Global configuration for MechanicsDSL with validation.
    
    All configuration values are validated on assignment to ensure
    they are within reasonable bounds and of correct types.
    """
    
    def __init__(self) -> None:
        """Initialize configuration with default values."""
        self._enable_profiling: bool = False
        self._enable_debug_logging: bool = False
        self._simplification_timeout: float = SIMPLIFICATION_TIMEOUT
        self._max_parser_errors: int = MAX_PARSER_ERRORS
        self._default_rtol: float = DEFAULT_RTOL
        self._default_atol: float = DEFAULT_ATOL
        self._trail_length: int = DEFAULT_TRAIL_LENGTH
        self._animation_fps: int = DEFAULT_FPS
        self._save_intermediate_results: bool = False
        self._cache_symbolic_results: bool = True
        # v6.0 Advanced features
        self._enable_performance_monitoring: bool = True
        self._cache_max_size: int = 256
        self._cache_max_memory_mb: float = 200.0
        self._enable_adaptive_solver: bool = True
        self._enable_parallel_processing: bool = False
        self._max_workers: int = 4
        self._enable_memory_monitoring: bool = True
        self._gc_threshold: Tuple[int, int, int] = (700, 10, 10)
        self._enable_type_checking: bool = True
        self._error_recovery_enabled: bool = True
        self._max_retry_attempts: int = 3
    
    @property
    def enable_profiling(self) -> bool:
        """Whether to enable performance profiling."""
        return self._enable_profiling
    
    @enable_profiling.setter
    def enable_profiling(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"enable_profiling must be bool, got {type(value).__name__}")
        self._enable_profiling = value
    
    @property
    def enable_debug_logging(self) -> bool:
        """Whether to enable debug-level logging."""
        return self._enable_debug_logging
    
    @enable_debug_logging.setter
    def enable_debug_logging(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"enable_debug_logging must be bool, got {type(value).__name__}")
        self._enable_debug_logging = value
    
    @property
    def simplification_timeout(self) -> float:
        """Timeout for symbolic simplification operations in seconds."""
        return self._simplification_timeout
    
    @simplification_timeout.setter
    def simplification_timeout(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"simplification_timeout must be numeric, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"simplification_timeout must be non-negative, got {value}")
        if value > 3600:
            raise ValueError(f"simplification_timeout too large (>{3600}s), got {value}")
        self._simplification_timeout = float(value)
    
    @property
    def max_parser_errors(self) -> int:
        """Maximum parser errors before giving up."""
        return self._max_parser_errors
    
    @max_parser_errors.setter
    def max_parser_errors(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError(f"max_parser_errors must be int, got {type(value).__name__}")
        if value < 1:
            raise ValueError(f"max_parser_errors must be at least 1, got {value}")
        if value > 1000:
            raise ValueError(f"max_parser_errors too large (>{1000}), got {value}")
        self._max_parser_errors = value
    
    @property
    def default_rtol(self) -> float:
        """Default relative tolerance for numerical integration."""
        return self._default_rtol
    
    @default_rtol.setter
    def default_rtol(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"default_rtol must be numeric, got {type(value).__name__}")
        if value <= 0 or value >= 1:
            raise ValueError(f"default_rtol must be in (0, 1), got {value}")
        self._default_rtol = float(value)
    
    @property
    def default_atol(self) -> float:
        """Default absolute tolerance for numerical integration."""
        return self._default_atol
    
    @default_atol.setter
    def default_atol(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"default_atol must be numeric, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"default_atol must be positive, got {value}")
        self._default_atol = float(value)
    
    @property
    def trail_length(self) -> int:
        """Maximum length of animation trails."""
        return self._trail_length
    
    @trail_length.setter
    def trail_length(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError(f"trail_length must be int, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"trail_length must be non-negative, got {value}")
        if value > 100000:
            raise ValueError(f"trail_length too large (>{100000}), got {value}")
        self._trail_length = value
    
    @property
    def animation_fps(self) -> int:
        """Animation frames per second."""
        return self._animation_fps
    
    @animation_fps.setter
    def animation_fps(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError(f"animation_fps must be int, got {type(value).__name__}")
        if value < 1:
            raise ValueError(f"animation_fps must be at least 1, got {value}")
        if value > 120:
            raise ValueError(f"animation_fps too large (>{120}), got {value}")
        self._animation_fps = value
    
    @property
    def save_intermediate_results(self) -> bool:
        """Whether to save intermediate computation results."""
        return self._save_intermediate_results
    
    @save_intermediate_results.setter
    def save_intermediate_results(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"save_intermediate_results must be bool, got {type(value).__name__}")
        self._save_intermediate_results = value
    
    @property
    def cache_symbolic_results(self) -> bool:
        """Whether to cache symbolic computation results."""
        return self._cache_symbolic_results
    
    @cache_symbolic_results.setter
    def cache_symbolic_results(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"cache_symbolic_results must be bool, got {type(value).__name__}")
        self._cache_symbolic_results = value
    
    @property
    def enable_performance_monitoring(self) -> bool:
        """Whether to enable performance monitoring."""
        return self._enable_performance_monitoring
    
    @enable_performance_monitoring.setter
    def enable_performance_monitoring(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"enable_performance_monitoring must be bool, got {type(value).__name__}")
        self._enable_performance_monitoring = value
    
    @property
    def enable_memory_monitoring(self) -> bool:
        """Whether to enable additional memory monitoring."""
        return self._enable_memory_monitoring
    
    @enable_memory_monitoring.setter
    def enable_memory_monitoring(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"enable_memory_monitoring must be bool, got {type(value).__name__}")
        self._enable_memory_monitoring = value
    
    @property
    def cache_max_size(self) -> int:
        """Maximum cache size."""
        return self._cache_max_size
    
    @cache_max_size.setter
    def cache_max_size(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError(f"cache_max_size must be int, got {type(value).__name__}")
        if value < 1:
            raise ValueError(f"cache_max_size must be at least 1, got {value}")
        if value > 10000:
            raise ValueError(f"cache_max_size too large (>{10000}), got {value}")
        self._cache_max_size = value
    
    @property
    def cache_max_memory_mb(self) -> float:
        """Maximum cache memory in MB."""
        return self._cache_max_memory_mb
    
    @cache_max_memory_mb.setter
    def cache_max_memory_mb(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"cache_max_memory_mb must be numeric, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"cache_max_memory_mb must be positive, got {value}")
        if value > 10000:
            raise ValueError(f"cache_max_memory_mb too large (>{10000} MB), got {value}")
        self._cache_max_memory_mb = float(value)
    
    @property
    def enable_adaptive_solver(self) -> bool:
        """Whether to enable adaptive solver selection."""
        return self._enable_adaptive_solver
    
    @enable_adaptive_solver.setter
    def enable_adaptive_solver(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"enable_adaptive_solver must be bool, got {type(value).__name__}")
        self._enable_adaptive_solver = value
    
    @property
    def error_recovery_enabled(self) -> bool:
        """Whether error recovery is enabled."""
        return self._error_recovery_enabled
    
    @error_recovery_enabled.setter
    def error_recovery_enabled(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"error_recovery_enabled must be bool, got {type(value).__name__}")
        self._error_recovery_enabled = value
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary containing all configuration values
        """
        return {
            'enable_profiling': self._enable_profiling,
            'enable_debug_logging': self._enable_debug_logging,
            'simplification_timeout': self._simplification_timeout,
            'max_parser_errors': self._max_parser_errors,
            'default_rtol': self._default_rtol,
            'default_atol': self._default_atol,
            'trail_length': self._trail_length,
            'animation_fps': self._animation_fps,
            'save_intermediate_results': self._save_intermediate_results,
            'cache_symbolic_results': self._cache_symbolic_results,
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Load configuration from dictionary with validation.
        
        Args:
            data: Dictionary containing configuration values
            
        Raises:
            TypeError: If data is not a dictionary
            ValueError: If any value is invalid
        """
        if not isinstance(data, dict):
            raise TypeError(f"data must be dict, got {type(data).__name__}")
        
        for k, v in data.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                logger.warning(f"Unknown configuration key: {k}")


# Global config instance
config = Config()
