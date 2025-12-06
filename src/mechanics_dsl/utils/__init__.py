"""
MechanicsDSL Utils Package

Modular utilities for configuration, logging, caching, profiling, and validation.
"""

from .logging import setup_logging, logger, LOG_FORMAT, LOG_DATE_FORMAT
from .config import Config, config
from .caching import LRUCache
from .profiling import (
    PerformanceMonitor, profile_function, timeout, TimeoutError, _perf_monitor
)
from .validation import (
    safe_float_conversion, validate_array_safe, safe_array_access,
    runtime_type_check, validate_finite, validate_positive, validate_non_negative,
    validate_time_span, validate_solution_dict, validate_file_path,
    resource_manager, AdvancedErrorHandler
)

# Re-export constants for backward compatibility
from .config import (
    DEFAULT_TRAIL_LENGTH, DEFAULT_FPS, ENERGY_TOLERANCE, DEFAULT_RTOL, DEFAULT_ATOL,
    SIMPLIFICATION_TIMEOUT, MAX_PARSER_ERRORS, ANIMATION_INTERVAL_MS, TRAIL_ALPHA,
    PRIMARY_COLOR, SECONDARY_COLOR, TERTIARY_COLOR
)

__all__ = [
    # Logging
    'setup_logging', 'logger', 'LOG_FORMAT', 'LOG_DATE_FORMAT',
    # Config
    'Config', 'config',
    # Caching
    'LRUCache',
    # Profiling
    'PerformanceMonitor', 'profile_function', 'timeout', 'TimeoutError', '_perf_monitor',
    # Validation
    'safe_float_conversion', 'validate_array_safe', 'safe_array_access',
    'runtime_type_check', 'validate_finite', 'validate_positive', 'validate_non_negative',
    'validate_time_span', 'validate_solution_dict', 'validate_file_path',
    'resource_manager', 'AdvancedErrorHandler',
    # Constants
    'DEFAULT_TRAIL_LENGTH', 'DEFAULT_FPS', 'ENERGY_TOLERANCE', 'DEFAULT_RTOL', 'DEFAULT_ATOL',
    'SIMPLIFICATION_TIMEOUT', 'MAX_PARSER_ERRORS', 'ANIMATION_INTERVAL_MS', 'TRAIL_ALPHA',
    'PRIMARY_COLOR', 'SECONDARY_COLOR', 'TERTIARY_COLOR',
]
