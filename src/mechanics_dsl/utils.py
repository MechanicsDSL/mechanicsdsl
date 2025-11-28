"""
Utility functions and classes for MechanicsDSL
"""
import numpy as np
import sympy as sp
import time
import logging
import signal
import sys
import platform
import threading
from typing import List, Dict, Optional, Tuple, Any, Union, Callable
from collections import defaultdict, OrderedDict
from pathlib import Path
from contextlib import contextmanager, ExitStack
from functools import wraps

try:
    import psutil
except ImportError:
    psutil = None  # Optional dependency

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

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(level: int = logging.INFO, 
                 log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        level: Logging level
        log_file: Optional file to write logs to
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger('MechanicsDSL')
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_float_conversion(value: Any) -> float:
    """
    Safely convert any value to Python float with comprehensive error handling.
    
    Args:
        value: Value to convert to float
        
    Returns:
        Converted float value (0.0 on failure)
    """
    if value is None:
        logger.warning("safe_float_conversion: None value, returning 0.0")
        return 0.0
    
    try:
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return 0.0
            if value.size == 1:
                result = float(value.item())
                if not np.isfinite(result):
                    logger.warning("safe_float_conversion: non-finite array value, returning 0.0")
                    return 0.0
                return result

            result = float(value.flat[0])
            if not np.isfinite(result):
                logger.warning("safe_float_conversion: non-finite array value, returning 0.0")
                return 0.0
            return result

        if isinstance(value, (np.integer, np.floating)):
            result = float(value)
            if not np.isfinite(result):
                logger.warning("safe_float_conversion: non-finite numpy value, returning 0.0")
                return 0.0
            return result

        if isinstance(value, np.bool_):
            return float(bool(value))

        if isinstance(value, (int, float)):
            result = float(value)
            if not np.isfinite(result):
                logger.warning(f"safe_float_conversion: non-finite value {value}, returning 0.0")
                return 0.0
            return result

        if isinstance(value, str):
            try:
                result = float(value)
                if not np.isfinite(result):
                    logger.warning(f"safe_float_conversion: non-finite string value '{value}', returning 0.0")
                    return 0.0
                return result
            except (ValueError, TypeError):
                logger.warning(f"safe_float_conversion: cannot convert string '{value}' to float, returning 0.0")
                return 0.0

        # Last resort: try direct conversion
        try:
            result = float(value)
            if not np.isfinite(result):
                logger.warning(f"safe_float_conversion: non-finite value {type(value).__name__}, returning 0.0")
                return 0.0
            return result
        except (ValueError, TypeError, OverflowError) as e:
            logger.warning(f"safe_float_conversion: conversion failed for {type(value).__name__}: {e}, returning 0.0")
            return 0.0
    except Exception as e:
        logger.error(f"safe_float_conversion: unexpected error converting {type(value).__name__}: {e}", exc_info=True)
        return 0.0

# ============================================================================
# ADVANCED UTILITIES
# ============================================================================

class PerformanceMonitor:
    """Advanced performance monitoring with memory and timing tracking"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.memory_snapshots: List[Dict[str, float]] = []
        self.start_times: Dict[str, float] = {}
        
    def start_timer(self, name: str) -> None:
        """Start timing an operation with validation"""
        if not isinstance(name, str) or not name:
            logger.warning(f"PerformanceMonitor.start_timer: invalid name '{name}', using 'unnamed'")
            name = 'unnamed'
        if name in self.start_times:
            logger.warning(f"PerformanceMonitor.start_timer: timer '{name}' already running, overwriting")
        self.start_times[name] = time.perf_counter()
        
    def stop_timer(self, name: str) -> float:
        """Stop timing and record duration with validation"""
        if not isinstance(name, str) or not name:
            logger.warning(f"PerformanceMonitor.stop_timer: invalid name '{name}'")
            return 0.0
        if name not in self.start_times:
            logger.warning(f"PerformanceMonitor.stop_timer: timer '{name}' was not started")
            return 0.0
        try:
            duration = time.perf_counter() - self.start_times[name]
            if duration < 0:
                logger.warning(f"PerformanceMonitor.stop_timer: negative duration for '{name}', clock issue?")
                duration = 0.0
            if duration > 86400:  # More than 24 hours seems wrong
                logger.warning(f"PerformanceMonitor.stop_timer: suspiciously long duration {duration}s for '{name}'")
            self.metrics[name].append(duration)
            del self.start_times[name]
            return duration
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"PerformanceMonitor.stop_timer: error stopping timer '{name}': {e}")
            return 0.0
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB"""
        if psutil is None:
            return {'rss': 0.0, 'vms': 0.0, 'percent': 0.0}
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            return {
                'rss': mem_info.rss / 1024 / 1024,  # Resident Set Size
                'vms': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
                'percent': process.memory_percent()
            }
        except (AttributeError, Exception):
            return {'rss': 0.0, 'vms': 0.0, 'percent': 0.0}
    
    def snapshot_memory(self, label: str = "") -> None:
        """Take a memory snapshot"""
        mem = self.get_memory_usage()
        mem['label'] = label
        mem['timestamp'] = time.time()
        self.memory_snapshots.append(mem)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric with validation"""
        if not isinstance(name, str) or not name:
            logger.warning(f"PerformanceMonitor.get_stats: invalid name '{name}'")
            return {}
        if name not in self.metrics or not self.metrics[name]:
            return {}
        try:
            values = self.metrics[name]
            if not values:
                return {}
            # Filter out invalid values
            valid_values = [v for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
            if not valid_values:
                logger.warning(f"PerformanceMonitor.get_stats: no valid values for '{name}'")
                return {}
            return {
                'count': len(valid_values),
                'total': sum(valid_values),
                'mean': float(np.mean(valid_values)),
                'std': float(np.std(valid_values)),
                'min': float(np.min(valid_values)),
                'max': float(np.max(valid_values))
            }
        except Exception as e:
            logger.error(f"PerformanceMonitor.get_stats: error computing stats for '{name}': {e}")
            return {}
    
    def reset(self) -> None:
        """Reset all metrics"""
        self.metrics.clear()
        self.memory_snapshots.clear()
        self.start_times.clear()

# Global performance monitor
_perf_monitor = PerformanceMonitor()

class LRUCache:
    """Advanced LRU cache with size limits and memory awareness"""
    
    def __init__(self, maxsize: int = 128, max_memory_mb: float = 100.0):
        """Initialize LRU cache with validation"""
        if not isinstance(maxsize, int) or maxsize < 1:
            logger.warning(f"LRUCache: invalid maxsize {maxsize}, using 128")
            maxsize = 128
        if not isinstance(max_memory_mb, (int, float)) or max_memory_mb <= 0:
            logger.warning(f"LRUCache: invalid max_memory_mb {max_memory_mb}, using 100.0")
            max_memory_mb = 100.0
        self.cache: OrderedDict = OrderedDict()
        self.maxsize = maxsize
        self.max_memory_mb = float(max_memory_mb)
        self.hits = 0
        self.misses = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with validation"""
        if not isinstance(key, str):
            logger.warning(f"LRUCache.get: invalid key type {type(key).__name__}, expected str")
            self.misses += 1
            return None
        try:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
        except (KeyError, TypeError, AttributeError) as e:
            logger.error(f"LRUCache.get: error accessing key '{key}': {e}")
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache with eviction if needed and validation"""
        if not isinstance(key, str):
            logger.warning(f"LRUCache.set: invalid key type {type(key).__name__}, expected str")
            return
        if value is None:
            logger.debug(f"LRUCache.set: storing None value for key '{key}'")
        try:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.maxsize:
                    # Remove least recently used
                    try:
                        self.cache.popitem(last=False)
                    except KeyError:
                        pass  # Cache was empty
            self.cache[key] = value
        except (TypeError, AttributeError, MemoryError) as e:
            logger.error(f"LRUCache.set: error setting key '{key}': {e}")
            # Try to free space
            try:
                while len(self.cache) > self.maxsize * 0.5:
                    self.cache.popitem(last=False)
            except Exception:
                pass
        
        # Check memory usage
        try:
            current_mem = self._estimate_memory_mb()
            if current_mem > self.max_memory_mb:
                # Evict oldest items until under limit
                while current_mem > self.max_memory_mb * 0.8 and self.cache:
                    self.cache.popitem(last=False)
                    current_mem = self._estimate_memory_mb()
        except Exception:
            pass  # Memory estimation failed, continue
    
    def _estimate_memory_mb(self) -> float:
        """Estimate cache memory usage"""
        try:
            total = 0
            for value in self.cache.values():
                if isinstance(value, np.ndarray):
                    total += value.nbytes
                elif isinstance(value, (sp.Expr, sp.Matrix)):
                    # Rough estimate for SymPy objects
                    total += sys.getsizeof(str(value))
                else:
                    total += sys.getsizeof(value)
            return total / 1024 / 1024
        except Exception:
            return 0.0
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            'size': len(self.cache),
            'maxsize': self.maxsize,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'memory_mb': self._estimate_memory_mb()
        }

class AdvancedErrorHandler:
    """Advanced error handling with retry and recovery mechanisms"""
    
    @staticmethod
    def retry_on_failure(max_retries: int = 3, delay: float = 0.1, 
                        backoff: float = 2.0, exceptions: Tuple = (Exception,)):
        """Decorator for retrying operations on failure"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                retries = 0
                current_delay = delay
                last_exception = None
                
                while retries < max_retries:
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        retries += 1
                        if retries < max_retries:
                            logger.warning(f"Attempt {retries} failed: {e}. Retrying in {current_delay}s...")
                            time.sleep(current_delay)
                            current_delay *= backoff
                        else:
                            logger.error(f"All {max_retries} attempts failed")
                raise last_exception
            return wrapper
        return decorator
    
    @staticmethod
    def safe_execute(func: Callable, default: Any = None, 
                    log_errors: bool = True) -> Any:
        """Safely execute a function with error handling"""
        try:
            return func()
        except Exception as e:
            if log_errors:
                logger.error(f"Error in safe_execute: {e}", exc_info=True)
            return default

@contextmanager
def resource_manager(*resources):
    """Context manager for multiple resources with validation"""
    if not resources:
        yield
        return
    with ExitStack() as stack:
        for resource in resources:
            if resource is None:
                logger.warning("resource_manager: None resource provided, skipping")
                continue
            try:
                if hasattr(resource, '__enter__') and hasattr(resource, '__exit__'):
                    stack.enter_context(resource)
                else:
                    logger.warning(f"resource_manager: resource {type(resource).__name__} is not a context manager")
            except Exception as e:
                logger.error(f"resource_manager: error adding resource {type(resource).__name__}: {e}")
        yield

def runtime_type_check(value: Any, expected_type: type, name: str = "value") -> bool:
    """Runtime type checking with detailed error messages and validation"""
    if expected_type is None:
        logger.error(f"runtime_type_check: expected_type is None for {name}")
        return False
    if not isinstance(expected_type, type):
        logger.error(f"runtime_type_check: expected_type is not a type: {type(expected_type).__name__}")
        return False
    if not isinstance(name, str):
        name = str(name)
    if not isinstance(value, expected_type):
        actual_type = type(value).__name__
        logger.warning(f"Type mismatch for {name}: expected {expected_type.__name__}, got {actual_type}")
        return False
    return True

def validate_array_safe(arr: Any, name: str = "array", 
                       min_size: int = 0, max_size: Optional[int] = None,
                       check_finite: bool = True) -> bool:
    """
    Comprehensive array validation with extensive checks.
    
    Args:
        arr: Array to validate
        name: Name for error messages
        min_size: Minimum array size
        max_size: Maximum array size (None for no limit)
        check_finite: Whether to check for finite values
        
    Returns:
        True if valid, False otherwise
    """
    if arr is None:
        logger.warning(f"validate_array_safe: {name} is None")
        return False
    if not isinstance(arr, np.ndarray):
        logger.warning(f"validate_array_safe: {name} is not numpy.ndarray, got {type(arr).__name__}")
        return False
    if arr.size < min_size:
        logger.warning(f"validate_array_safe: {name} size {arr.size} < min_size {min_size}")
        return False
    if max_size is not None and arr.size > max_size:
        logger.warning(f"validate_array_safe: {name} size {arr.size} > max_size {max_size}")
        return False
    if check_finite and not np.all(np.isfinite(arr)):
        logger.warning(f"validate_array_safe: {name} contains non-finite values")
        return False
    return True

def safe_array_access(arr: np.ndarray, index: int, default: float = 0.0) -> float:
    """
    Safely access array element with bounds checking.
    
    Args:
        arr: Array to access
        index: Index to access
        default: Default value if access fails
        
    Returns:
        Array element or default value
    """
    if arr is None:
        logger.warning(f"safe_array_access: array is None, returning default {default}")
        return default
    if not isinstance(arr, np.ndarray):
        logger.warning(f"safe_array_access: not an array, got {type(arr).__name__}")
        return default
    if not isinstance(index, int):
        logger.warning(f"safe_array_access: index is not int, got {type(index).__name__}")
        return default
    if index < 0 or index >= arr.size:
        logger.warning(f"safe_array_access: index {index} out of bounds [0, {arr.size})")
        return default
    try:
        value = arr.flat[index]
        result = safe_float_conversion(value)
        if not np.isfinite(result):
            logger.warning(f"safe_array_access: non-finite value at index {index}, returning default")
            return default
        return result
    except (IndexError, TypeError, ValueError) as e:
        logger.warning(f"safe_array_access: error accessing index {index}: {e}, returning default")
        return default

# ============================================================================
# CONFIGURATION
# ============================================================================

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

# ============================================================================
# UTILITY FUNCTIONS AND DECORATORS
# ============================================================================

class TimeoutError(Exception):
    """Raised when an operation times out"""
    pass

@contextmanager
def timeout(seconds: float):
    """
    Cross-platform timeout context manager for timing out operations.
    
    Uses signal.SIGALRM on Unix systems and threading.Timer on Windows.
    Note: Threading-based timeout on Windows cannot interrupt CPU-bound operations.
    
    Args:
        seconds: Maximum time allowed (must be positive)
        
    Raises:
        TimeoutError: If operation exceeds time limit
        ValueError: If seconds is not positive
    """
    if not isinstance(seconds, (int, float)):
        raise TypeError(f"seconds must be numeric, got {type(seconds).__name__}")
    if seconds <= 0:
        raise ValueError(f"seconds must be positive, got {seconds}")
    
    if platform.system() == 'Windows':
        # Windows: Use threading.Timer (cannot interrupt CPU-bound operations)
        timer: Optional[threading.Timer] = None
        timeout_occurred = threading.Event()
        
        def timeout_handler() -> None:
            timeout_occurred.set()
            raise TimeoutError(f"Operation timed out after {seconds} seconds")
        
        timer = threading.Timer(seconds, timeout_handler)
        timer.daemon = True
        timer.start()
        
        try:
            yield
        finally:
            if timer is not None:
                timer.cancel()
                timer.join(timeout=0.1)
    else:
        # Unix: Use signal.SIGALRM (can interrupt operations)
        def timeout_handler(signum: int, frame: Any) -> None:
            raise TimeoutError(f"Operation timed out after {seconds} seconds")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(seconds))
        
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

def profile_function(func):
    """Decorator to profile function execution"""
    def wrapper(*args, **kwargs):
        if config.enable_profiling:
            import cProfile
            import pstats
            from io import StringIO
            
            profiler = cProfile.Profile()
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
            
            s = StringIO()
            stats = pstats.Stats(profiler, stream=s)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions
            
            logger.debug(f"\n{'='*70}\nProfile for {func.__name__}:\n{s.getvalue()}\n{'='*70}")
            return result
        else:
            return func(*args, **kwargs)
    return wrapper

def validate_finite(arr: np.ndarray, name: str = "array") -> bool:
    """
    Validate that array contains only finite values.
    
    Args:
        arr: NumPy array to validate
        name: Name for error messages
        
    Returns:
        True if all finite, False otherwise
        
    Raises:
        TypeError: If arr is not a numpy array
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be numpy.ndarray, got {type(arr).__name__}")
    
    if not np.all(np.isfinite(arr)):
        logger.warning(f"{name} contains non-finite values")
        return False
    return True

def validate_positive(value: Union[int, float], name: str = "value") -> None:
    """
    Validate that a value is positive.
    
    Args:
        value: Value to validate
        name: Name for error messages
        
    Raises:
        TypeError: If value is not numeric
        ValueError: If value is not positive
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")

def validate_non_negative(value: Union[int, float], name: str = "value") -> None:
    """
    Validate that a value is non-negative.
    
    Args:
        value: Value to validate
        name: Name for error messages
        
    Raises:
        TypeError: If value is not numeric
        ValueError: If value is negative
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")

def validate_time_span(t_span: Tuple[float, float]) -> None:
    """
    Validate time span tuple.
    
    Args:
        t_span: Tuple of (t_start, t_end)
        
    Raises:
        TypeError: If t_span is not a tuple or values are not numeric
        ValueError: If t_start >= t_end or values are negative
    """
    if not isinstance(t_span, tuple):
        raise TypeError(f"t_span must be tuple, got {type(t_span).__name__}")
    if len(t_span) != 2:
        raise ValueError(f"t_span must have length 2, got {len(t_span)}")
    
    t_start, t_end = t_span
    
    if not isinstance(t_start, (int, float)) or not isinstance(t_end, (int, float)):
        raise TypeError("t_span values must be numeric")
    
    if t_start < 0 or t_end < 0:
        raise ValueError(f"Time values must be non-negative, got {t_span}")
    
    if t_start >= t_end:
        raise ValueError(f"t_start must be < t_end, got {t_span}")

def validate_solution_dict(solution: dict) -> None:
    """
    Validate solution dictionary structure and content.
    
    Args:
        solution: Solution dictionary from simulation
        
    Raises:
        TypeError: If solution is not a dict
        ValueError: If required keys are missing or values are invalid
    """
    if not isinstance(solution, dict):
        raise TypeError(f"solution must be dict, got {type(solution).__name__}")
    
    if 'success' not in solution:
        raise ValueError("solution must contain 'success' key")
    
    if not isinstance(solution['success'], bool):
        raise TypeError("solution['success'] must be bool")
    
    if solution['success']:
        required_keys = ['t', 'y', 'coordinates']
        for key in required_keys:
            if key not in solution:
                raise ValueError(f"solution missing required key: {key}")
        
        # Validate 't' array
        t = solution['t']
        if not isinstance(t, np.ndarray):
            raise TypeError(f"solution['t'] must be numpy.ndarray, got {type(t).__name__}")
        if len(t) == 0:
            raise ValueError("solution['t'] cannot be empty")
        if not validate_finite(t, "solution['t']"):
            raise ValueError("solution['t'] contains non-finite values")
        
        # Validate 'y' array
        y = solution['y']
        if not isinstance(y, np.ndarray):
            raise TypeError(f"solution['y'] must be numpy.ndarray, got {type(y).__name__}")
        if y.shape[0] == 0:
            raise ValueError("solution['y'] cannot be empty")
        if y.shape[1] != len(t):
            raise ValueError(f"solution['y'] shape mismatch: y.shape[1]={y.shape[1]} != len(t)={len(t)}")
        if not validate_finite(y, "solution['y']"):
            raise ValueError("solution['y'] contains non-finite values")
        
        # Validate 'coordinates'
        coords = solution['coordinates']
        if not isinstance(coords, (list, tuple)):
            raise TypeError(f"solution['coordinates'] must be list or tuple, got {type(coords).__name__}")
        if len(coords) == 0:
            raise ValueError("solution['coordinates'] cannot be empty")
        if y.shape[0] != 2 * len(coords):
            raise ValueError(f"State vector size mismatch: y.shape[0]={y.shape[0]} != 2*len(coords)={2*len(coords)}")

def validate_file_path(filename: str, must_exist: bool = False) -> None:
    """
    Validate file path.
    
    Args:
        filename: File path to validate
        must_exist: Whether file must exist
        
    Raises:
        TypeError: If filename is not a string
        ValueError: If filename is empty or invalid
        FileNotFoundError: If must_exist=True and file doesn't exist
    """
    if not isinstance(filename, str):
        raise TypeError(f"filename must be str, got {type(filename).__name__}")
    
    filename = filename.strip()
    if not filename:
        raise ValueError("filename cannot be empty")
    
    # Check for path traversal attempts
    if '..' in filename:
        raise ValueError(f"filename contains '..' which may be unsafe: {filename}")
    
    if must_exist:
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {filename}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {filename}")
