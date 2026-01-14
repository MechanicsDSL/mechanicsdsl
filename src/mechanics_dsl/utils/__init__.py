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
from .registry import (
    VariableCategory, COORDINATE_TYPES, CONSTANT_TYPES,
    COMMON_COORDINATE_NAMES, COMMON_CONSTANT_NAMES,
    is_coordinate_type, is_constant_type, is_likely_coordinate, classify_variable
)
from .rate_limit import (
    RateLimiter, SimulationRateLimiter, RateLimitExceeded, TokenBucket
)
from .path_validation import (
    PathValidationError, is_safe_filename, secure_filename,
    validate_path_within_base, safe_open, safe_path_join
)

# Re-export constants for backward compatibility and convenience
from .config import (
    # Physical constants
    STANDARD_GRAVITY, DEFAULT_GRAVITY, SPEED_OF_LIGHT, PLANCK_CONSTANT, HBAR,
    # Numerical constants
    DEFAULT_RTOL, DEFAULT_ATOL, ENERGY_TOLERANCE, DEFAULT_NUM_POINTS,
    DEFAULT_MAX_STEP_FRACTION, DEFAULT_STIFFNESS_TEST_DURATION,
    SIMPLIFICATION_TIMEOUT, MAX_PARSER_ERRORS, MAX_SOLVER_ITERATIONS,
    NUMERICAL_INFINITY, SINGULARITY_THRESHOLD,
    # Visualization constants
    DEFAULT_TRAIL_LENGTH, DEFAULT_FPS, ANIMATION_INTERVAL_MS,
    DEFAULT_DPI, MIN_DPI, MAX_DPI, TRAIL_ALPHA,
    PRIMARY_COLOR, SECONDARY_COLOR, TERTIARY_COLOR,
    ACCENT_COLOR, WARNING_COLOR, SUCCESS_COLOR,
    DEFAULT_MARKER_SIZE, LARGE_MARKER_SIZE,
    # File constants
    MAX_PATH_LENGTH, MAX_FILE_SIZE,
    # Cache constants
    DEFAULT_CACHE_SIZE, DEFAULT_CACHE_MEMORY_MB,
    # Logging constants
    LOG_LEVELS,
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
    # Registry
    'VariableCategory', 'COORDINATE_TYPES', 'CONSTANT_TYPES',
    'COMMON_COORDINATE_NAMES', 'COMMON_CONSTANT_NAMES',
    'is_coordinate_type', 'is_constant_type', 'is_likely_coordinate', 'classify_variable',
    # Rate Limiting
    'RateLimiter', 'SimulationRateLimiter', 'RateLimitExceeded', 'TokenBucket',
    # Path Validation (CWE-22 prevention)
    'PathValidationError', 'is_safe_filename', 'secure_filename',
    'validate_path_within_base', 'safe_open', 'safe_path_join',
    # Physical Constants
    'STANDARD_GRAVITY', 'DEFAULT_GRAVITY', 'SPEED_OF_LIGHT', 'PLANCK_CONSTANT', 'HBAR',
    # Numerical Constants
    'DEFAULT_RTOL', 'DEFAULT_ATOL', 'ENERGY_TOLERANCE', 'DEFAULT_NUM_POINTS',
    'SIMPLIFICATION_TIMEOUT', 'MAX_PARSER_ERRORS', 'MAX_SOLVER_ITERATIONS',
    'NUMERICAL_INFINITY', 'SINGULARITY_THRESHOLD',
    # Visualization Constants
    'DEFAULT_TRAIL_LENGTH', 'DEFAULT_FPS', 'ANIMATION_INTERVAL_MS',
    'TRAIL_ALPHA', 'PRIMARY_COLOR', 'SECONDARY_COLOR', 'TERTIARY_COLOR',
    # File/Cache Constants
    'MAX_PATH_LENGTH', 'MAX_FILE_SIZE', 'DEFAULT_CACHE_SIZE', 'DEFAULT_CACHE_MEMORY_MB',
]
