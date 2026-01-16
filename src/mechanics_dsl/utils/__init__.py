"""
MechanicsDSL Utils Package

Modular utilities for configuration, logging, caching, profiling, and validation.
"""

from .caching import LRUCache

# Re-export constants for backward compatibility and convenience
from .config import (  # Physical constants; Numerical constants; Visualization constants; File constants; Cache constants; Logging constants
    ACCENT_COLOR,
    ANIMATION_INTERVAL_MS,
    DEFAULT_ATOL,
    DEFAULT_CACHE_MEMORY_MB,
    DEFAULT_CACHE_SIZE,
    DEFAULT_DPI,
    DEFAULT_FPS,
    DEFAULT_GRAVITY,
    DEFAULT_MARKER_SIZE,
    DEFAULT_MAX_STEP_FRACTION,
    DEFAULT_NUM_POINTS,
    DEFAULT_RTOL,
    DEFAULT_STIFFNESS_TEST_DURATION,
    DEFAULT_TRAIL_LENGTH,
    ENERGY_TOLERANCE,
    HBAR,
    LARGE_MARKER_SIZE,
    LOG_LEVELS,
    MAX_DPI,
    MAX_FILE_SIZE,
    MAX_PARSER_ERRORS,
    MAX_PATH_LENGTH,
    MAX_SOLVER_ITERATIONS,
    MIN_DPI,
    NUMERICAL_INFINITY,
    PLANCK_CONSTANT,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    SIMPLIFICATION_TIMEOUT,
    SINGULARITY_THRESHOLD,
    SPEED_OF_LIGHT,
    STANDARD_GRAVITY,
    SUCCESS_COLOR,
    TERTIARY_COLOR,
    TRAIL_ALPHA,
    WARNING_COLOR,
    Config,
    config,
)
from .logging import LOG_DATE_FORMAT, LOG_FORMAT, logger, setup_logging
from .path_validation import (
    PathValidationError,
    is_safe_filename,
    safe_open,
    safe_path_join,
    secure_filename,
    validate_path_within_base,
)
from .profiling import PerformanceMonitor, TimeoutError, _perf_monitor, profile_function, timeout
from .rate_limit import RateLimiter, RateLimitExceeded, SimulationRateLimiter, TokenBucket
from .registry import (
    COMMON_CONSTANT_NAMES,
    COMMON_COORDINATE_NAMES,
    CONSTANT_TYPES,
    COORDINATE_TYPES,
    VariableCategory,
    classify_variable,
    is_constant_type,
    is_coordinate_type,
    is_likely_coordinate,
)
from .validation import (
    AdvancedErrorHandler,
    resource_manager,
    runtime_type_check,
    safe_array_access,
    safe_float_conversion,
    validate_array_safe,
    validate_file_path,
    validate_finite,
    validate_non_negative,
    validate_positive,
    validate_solution_dict,
    validate_time_span,
)

__all__ = [
    # Logging
    "setup_logging",
    "logger",
    "LOG_FORMAT",
    "LOG_DATE_FORMAT",
    # Config
    "Config",
    "config",
    # Caching
    "LRUCache",
    # Profiling
    "PerformanceMonitor",
    "profile_function",
    "timeout",
    "TimeoutError",
    "_perf_monitor",
    # Validation
    "safe_float_conversion",
    "validate_array_safe",
    "safe_array_access",
    "runtime_type_check",
    "validate_finite",
    "validate_positive",
    "validate_non_negative",
    "validate_time_span",
    "validate_solution_dict",
    "validate_file_path",
    "resource_manager",
    "AdvancedErrorHandler",
    # Registry
    "VariableCategory",
    "COORDINATE_TYPES",
    "CONSTANT_TYPES",
    "COMMON_COORDINATE_NAMES",
    "COMMON_CONSTANT_NAMES",
    "is_coordinate_type",
    "is_constant_type",
    "is_likely_coordinate",
    "classify_variable",
    # Rate Limiting
    "RateLimiter",
    "SimulationRateLimiter",
    "RateLimitExceeded",
    "TokenBucket",
    # Path Validation (CWE-22 prevention)
    "PathValidationError",
    "is_safe_filename",
    "secure_filename",
    "validate_path_within_base",
    "safe_open",
    "safe_path_join",
    # Physical Constants
    "STANDARD_GRAVITY",
    "DEFAULT_GRAVITY",
    "SPEED_OF_LIGHT",
    "PLANCK_CONSTANT",
    "HBAR",
    # Numerical Constants
    "DEFAULT_RTOL",
    "DEFAULT_ATOL",
    "ENERGY_TOLERANCE",
    "DEFAULT_NUM_POINTS",
    "SIMPLIFICATION_TIMEOUT",
    "MAX_PARSER_ERRORS",
    "MAX_SOLVER_ITERATIONS",
    "NUMERICAL_INFINITY",
    "SINGULARITY_THRESHOLD",
    # Visualization Constants
    "DEFAULT_TRAIL_LENGTH",
    "DEFAULT_FPS",
    "ANIMATION_INTERVAL_MS",
    "TRAIL_ALPHA",
    "PRIMARY_COLOR",
    "SECONDARY_COLOR",
    "TERTIARY_COLOR",
    # File/Cache Constants
    "MAX_PATH_LENGTH",
    "MAX_FILE_SIZE",
    "DEFAULT_CACHE_SIZE",
    "DEFAULT_CACHE_MEMORY_MB",
]
