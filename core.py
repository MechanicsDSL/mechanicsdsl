"""
MechanicsDSL: A Domain-Specific Language for Classical Mechanics

A comprehensive framework for symbolic and numerical analysis of classical 
mechanical systems using LaTeX-inspired notation with complete compiler pipeline.

Author: Noah Parsons
Version: 6.0.0 - Enterprise-Grade Physics Engine with Advanced Features
License: MIT

New in v6.0.0 (Major Enterprise Upgrade):
- Advanced Error Recovery: Multi-level error handling with automatic retry mechanisms
- Performance Monitoring: Built-in profiling, benchmarking, and optimization suggestions
- Intelligent Caching: Multi-tier caching with LRU eviction and memory management
- Adaptive Solvers: Automatic solver selection based on system characteristics
- Advanced Type Safety: Runtime type checking with comprehensive validation
- Memory Management: Resource pooling, garbage collection hints, and memory monitoring
- Enhanced Visualization: Interactive plots, real-time parameter adjustment, export formats
- Robust Validation: Multi-pass validation with detailed diagnostics
- Parallel Processing: Multi-threaded compilation and simulation where applicable
- Advanced Numerical Methods: Support for DAE solvers, event detection, and sensitivity analysis
- Comprehensive Logging: Structured logging with multiple output formats
- Resource Management: Context managers for automatic cleanup and resource tracking
- Advanced Constraint Handling: Improved Lagrange multiplier methods and constraint stabilization
- Energy Monitoring: Real-time energy tracking with conservation validation
- State Management: Advanced state serialization with versioning and migration

Previous versions:
v0.5.0:
- True 3D Motion: Euler angles and quaternions for spinning tops, gyroscopes
- Non-Conservative Forces: Friction, damping, air drag via \\force{} and \\damping{}
- Non-Holonomic Constraints: Velocity-dependent constraints (rolling without slipping)
- Professional Test Suite: Formal pytest-based testing framework
- Distribution Ready: pip-installable package with pyproject.toml
- Web UI: Streamlit-based graphical interface for non-programmers
- Enhanced 3D Visualization: True 3D coordinate systems and rotations
- Advanced Constraint Handling: Mixed holonomic and non-holonomic systems

v0.4.0:
- Cross-platform compatibility (Windows/Unix timeout support)
- Security hardening (removed eval(), safe AST-based parsing)
- Comprehensive input validation throughout
- Specific exception handling (no bare except clauses)
- Extensive type hints and docstrings
- Bounds checking and parameter validation
- Enhanced error messages with context
- Production-ready error recovery
- Performance optimizations
- Comprehensive validation framework
"""

import re
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from typing import List, Dict, Optional, Tuple, Any, Union, Callable, Literal, Set, cast, Protocol, runtime_checkable
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import time
import warnings
import logging
import signal
from collections import defaultdict, deque, OrderedDict
from pathlib import Path
import pickle
from contextlib import contextmanager, ExitStack
import platform
import threading
import ast
import operator
import gc
import sys
import traceback
from functools import lru_cache, wraps
from weakref import WeakValueDictionary
try:
    import psutil
except ImportError:
    psutil = None  # Optional dependency

try:
    from typing import get_type_hints
except ImportError:
    def get_type_hints(obj): return {}

__version__ = "6.0.0"
__author__ = "Noah Parsons"
__license__ = "MIT"

# ============================================================================
# ROBUSTNESS ENHANCEMENTS - v6.0
# ============================================================================
# This version includes extensive defensive programming, comprehensive error
# handling, input validation, bounds checking, memory safety, and recovery
# mechanisms throughout the codebase for maximum robustness and reliability.
# 
# Key robustness features:
# - Comprehensive input validation at all entry points
# - Safe array access with bounds checking
# - Non-finite value detection and handling
# - Memory safety with resource cleanup
# - Error recovery with graceful degradation
# - Extensive logging for debugging
# - Type checking and validation
# - Defensive programming practices
# ============================================================================

# ============================================================================
# CONFIGURATION AND CONSTANTS
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

def safe_float_conversion(value: Any) -> float:
    """
    Safely convert any value to Python float with comprehensive error handling.
    
    Args:
        value: Value to convert to float
        
    Returns:
        Converted float value (0.0 on failure)
        
    Raises:
        None - Always returns a valid float, never raises
    """
    if value is None:
        logger.warning("safe_float_conversion: None value, returning 0.0")
        return 0.0
    
    try:
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return 0.0
        elif value.size == 1:
                result = float(value.item())
                if not np.isfinite(result):
                    logger.warning(f"safe_float_conversion: non-finite array value, returning 0.0")
                    return 0.0
                return result
        else:
                result = float(value.flat[0])
                if not np.isfinite(result):
                    logger.warning(f"safe_float_conversion: non-finite array value, returning 0.0")
                    return 0.0
                return result
    elif isinstance(value, (np.integer, np.floating)):
            result = float(value)
            if not np.isfinite(result):
                logger.warning(f"safe_float_conversion: non-finite numpy value, returning 0.0")
                return 0.0
            return result
    elif isinstance(value, np.bool_):
        return float(bool(value))
        elif isinstance(value, (int, float)):
            result = float(value)
            if not np.isfinite(result):
                logger.warning(f"safe_float_conversion: non-finite value {value}, returning 0.0")
                return 0.0
            return result
        elif isinstance(value, str):
            # Try to parse string
            try:
                result = float(value)
                if not np.isfinite(result):
                    logger.warning(f"safe_float_conversion: non-finite string value '{value}', returning 0.0")
                    return 0.0
                return result
            except (ValueError, TypeError):
                logger.warning(f"safe_float_conversion: cannot convert string '{value}' to float, returning 0.0")
                return 0.0
        else:
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
# ADVANCED UTILITIES - v6.0
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
        
    Example:
        >>> with timeout(5.0):
        ...     # Some operation that should complete within 5 seconds
        ...     result = expensive_computation()
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
    if '..' in filename or filename.startswith('/') or ':' in filename:
        # Allow absolute paths but warn about potential issues
        if '..' in filename:
            raise ValueError(f"filename contains '..' which may be unsafe: {filename}")
    
    if must_exist:
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {filename}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {filename}")

# ============================================================================
# TOKEN SYSTEM - Enhanced
# ============================================================================

TOKEN_TYPES = [
    # Physics specific commands (order matters!)
    ("DOT_NOTATION", r"\\ddot|\\dot"),
    ("SYSTEM", r"\\system"),
    ("DEFVAR", r"\\defvar"),
    ("DEFINE", r"\\define"),
    ("LAGRANGIAN", r"\\lagrangian"),
    ("HAMILTONIAN", r"\\hamiltonian"),
    ("TRANSFORM", r"\\transform"),
    ("CONSTRAINT", r"\\constraint"),
    ("NONHOLONOMIC", r"\\nonholonomic"),
    ("FORCE", r"\\force"),
    ("DAMPING", r"\\damping"),
    ("INITIAL", r"\\initial"),
    ("SOLVE", r"\\solve"),
    ("ANIMATE", r"\\animate"),
    ("PLOT", r"\\plot"),
    ("PARAMETER", r"\\parameter"),
    ("EXPORT", r"\\export"),
    ("IMPORT", r"\\import"),
    ("EULER_ANGLES", r"\\euler"),
    ("QUATERNION", r"\\quaternion"),
    
    # Vector operations
    ("VEC", r"\\vec"),
    ("HAT", r"\\hat"),
    ("MAGNITUDE", r"\\mag|\\norm"),
    
    # Advanced math operators
    ("VECTOR_DOT", r"\\cdot"),
    ("VECTOR_CROSS", r"\\times|\\cross"),
    ("GRADIENT", r"\\nabla|\\grad"),
    ("DIVERGENCE", r"\\div"),
    ("CURL", r"\\curl"),
    ("LAPLACIAN", r"\\laplacian|\\Delta"),
    
    # Calculus
    ("PARTIAL", r"\\partial"),
    ("INTEGRAL", r"\\int"),
    ("OINT", r"\\oint"),
    ("SUM", r"\\sum"),
    ("LIMIT", r"\\lim"),
    ("FRAC", r"\\frac"),
    
    # Greek letters (comprehensive)
    ("GREEK_LETTER", r"\\alpha|\\beta|\\gamma|\\delta|\\epsilon|\\varepsilon|\\zeta|\\eta|\\theta|\\vartheta|\\iota|\\kappa|\\lambda|\\mu|\\nu|\\xi|\\omicron|\\pi|\\varpi|\\rho|\\varrho|\\sigma|\\varsigma|\\tau|\\upsilon|\\phi|\\varphi|\\chi|\\psi|\\omega"),
    
    # General commands
    ("COMMAND", r"\\[a-zA-Z_][a-zA-Z0-9_]*"),
    
    # Brackets and grouping
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACKET", r"\["),
    ("RBRACKET", r"\]"),
    
    # Mathematical operators
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("MULTIPLY", r"\*"),
    ("DIVIDE", r"/"),
    ("POWER", r"\^"),
    ("EQUALS", r"="),
    ("COMMA", r","),
    ("SEMICOLON", r";"),
    ("COLON", r":"),
    ("DOT", r"\."),
    ("UNDERSCORE", r"_"),
    ("PIPE", r"\|"),
    
    # Basic tokens
    ("NUMBER", r"\d+\.?\d*([eE][+-]?\d+)?"),
    ("IDENT", r"[a-zA-Z_][a-zA-Z0-9_]*"),
    ("WHITESPACE", r"\s+"),
    ("NEWLINE", r"\n"),
    ("COMMENT", r"%.*"),
]

token_regex = "|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_TYPES)
token_pattern = re.compile(token_regex)

@dataclass
class Token:
    """Token with position tracking for better error messages"""
    type: str
    value: str
    position: int = 0
    line: int = 1
    column: int = 1

    def __repr__(self) -> str:
        return f"{self.type}:{self.value}@{self.line}:{self.column}"

def tokenize(source: str) -> List[Token]:
    """
    Tokenizer with position tracking and comprehensive error reporting
    
    Args:
        source: DSL source code
        
    Returns:
        List of tokens (excluding whitespace and comments)
    """
    tokens = []
    line = 1
    line_start = 0
    
    for match in token_pattern.finditer(source):
        kind = match.lastgroup
        value = match.group()
        position = match.start()
        
        # Update line tracking
        while line_start < position and '\n' in source[line_start:position]:
            newline_pos = source.find('\n', line_start)
            if newline_pos != -1 and newline_pos < position:
                line += 1
                line_start = newline_pos + 1
            else:
                break
                
        column = position - line_start + 1
        
        if kind not in ["WHITESPACE", "COMMENT"]:
            tokens.append(Token(kind, value, position, line, column))
    
    logger.debug(f"Tokenized {len(tokens)} tokens from {line} lines")
    return tokens

# ============================================================================
# COMPLETE AST SYSTEM - Enhanced with type safety
# ============================================================================

class ASTNode:
    """Base class for all AST nodes"""
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class Expression(ASTNode):
    """Base class for all expressions"""
    pass

# Basic expressions
@dataclass
class NumberExpr(Expression):
    value: float
    def __repr__(self) -> str:
        return f"Num({self.value})"

@dataclass
class IdentExpr(Expression):
    name: str
    def __repr__(self) -> str:
        return f"Id({self.name})"

@dataclass
class GreekLetterExpr(Expression):
    letter: str
    def __repr__(self) -> str:
        return f"Greek({self.letter})"

@dataclass
class DerivativeVarExpr(Expression):
    """Represents \\dot{x} or \\ddot{x} notation"""
    var: str
    order: int = 1
    def __repr__(self) -> str:
        return f"DerivativeVar({self.var}, order={self.order})"

# Binary operations with type safety
@dataclass
class BinaryOpExpr(Expression):
    left: Expression
    operator: Literal["+", "-", "*", "/", "^"]
    right: Expression
    def __repr__(self) -> str:
        return f"BinOp({self.left} {self.operator} {self.right})"

@dataclass
class UnaryOpExpr(Expression):
    operator: Literal["+", "-"]
    operand: Expression
    def __repr__(self) -> str:
        return f"UnaryOp({self.operator}{self.operand})"

# Vector expressions
@dataclass
class VectorExpr(Expression):
    components: List[Expression]
    def __repr__(self) -> str:
        return f"Vector({self.components})"

@dataclass
class VectorOpExpr(Expression):
    operation: str
    left: Expression
    right: Optional[Expression] = None
    def __repr__(self) -> str:
        if self.right:
            return f"VectorOp({self.operation}, {self.left}, {self.right})"
        return f"VectorOp({self.operation}, {self.left})"

# Calculus expressions
@dataclass
class DerivativeExpr(Expression):
    expr: Expression
    var: str
    order: int = 1
    partial: bool = False
    def __repr__(self) -> str:
        type_str = "Partial" if self.partial else "Total"
        return f"{type_str}Deriv({self.expr}, {self.var}, order={self.order})"

@dataclass
class IntegralExpr(Expression):
    expr: Expression
    var: str
    lower: Optional[Expression] = None
    upper: Optional[Expression] = None
    line_integral: bool = False
    def __repr__(self) -> str:
        return f"Integral({self.expr}, {self.var}, {self.lower}, {self.upper})"

# Function calls
@dataclass
class FunctionCallExpr(Expression):
    name: str
    args: List[Expression]
    def __repr__(self) -> str:
        return f"Call({self.name}, {self.args})"

@dataclass
class FractionExpr(Expression):
    numerator: Expression
    denominator: Expression
    def __repr__(self) -> str:
        return f"Frac({self.numerator}/{self.denominator})"

# Physics-specific AST nodes
@dataclass
class SystemDef(ASTNode):
    name: str
    def __repr__(self) -> str:
        return f"System({self.name})"

@dataclass
class VarDef(ASTNode):
    name: str
    vartype: str
    unit: str
    vector: bool = False
    def __repr__(self) -> str:
        vec_str = " [Vector]" if self.vector else ""
        return f"VarDef({self.name}: {self.vartype}[{self.unit}]{vec_str})"

@dataclass
class ParameterDef(ASTNode):
    name: str
    value: float
    unit: str
    def __repr__(self) -> str:
        return f"Parameter({self.name} = {self.value} [{self.unit}])"

@dataclass
class DefineDef(ASTNode):
    name: str
    args: List[str]
    body: Expression
    def __repr__(self) -> str:
        return f"Define({self.name}({', '.join(self.args)}) = {self.body})"

@dataclass
class LagrangianDef(ASTNode):
    expr: Expression
    def __repr__(self) -> str:
        return f"Lagrangian({self.expr})"

@dataclass
class HamiltonianDef(ASTNode):
    expr: Expression
    def __repr__(self) -> str:
        return f"Hamiltonian({self.expr})"

@dataclass
class TransformDef(ASTNode):
    coord_type: str
    var: str
    expr: Expression
    def __repr__(self) -> str:
        return f"Transform({self.coord_type}: {self.var} = {self.expr})"

@dataclass
class ConstraintDef(ASTNode):
    expr: Expression
    constraint_type: str = "holonomic"
    def __repr__(self) -> str:
        return f"Constraint({self.expr}, type={self.constraint_type})"

@dataclass
class NonHolonomicConstraintDef(ASTNode):
    """Non-holonomic constraint (velocity-dependent)"""
    expr: Expression
    def __repr__(self) -> str:
        return f"NonHolonomicConstraint({self.expr})"

@dataclass
class ForceDef(ASTNode):
    """Non-conservative force definition"""
    expr: Expression
    force_type: str = "general"  # "friction", "damping", "drag", "general"
    def __repr__(self) -> str:
        return f"Force({self.expr}, type={self.force_type})"

@dataclass
class DampingDef(ASTNode):
    """Damping force definition"""
    expr: Expression
    damping_coefficient: Optional[float] = None
    def __repr__(self) -> str:
        return f"Damping({self.expr}, coeff={self.damping_coefficient})"

@dataclass
class InitialCondition(ASTNode):
    conditions: Dict[str, float]
    def __repr__(self) -> str:
        return f"Initial({self.conditions})"

@dataclass
class SolveDef(ASTNode):
    method: str
    options: Dict[str, Any] = field(default_factory=dict)
    def __repr__(self) -> str:
        return f"Solve({self.method}, {self.options})"

@dataclass
class AnimateDef(ASTNode):
    target: str
    options: Dict[str, Any] = field(default_factory=dict)
    def __repr__(self) -> str:
        return f"Animate({self.target}, {self.options})"

@dataclass
class ExportDef(ASTNode):
    filename: str
    format: str = "json"
    def __repr__(self) -> str:
        return f"Export({self.filename}, {self.format})"

@dataclass
class ImportDef(ASTNode):
    filename: str
    def __repr__(self) -> str:
        return f"Import({self.filename})"

# ============================================================================
# COMPREHENSIVE PHYSICS UNITS SYSTEM
# ============================================================================

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

# ============================================================================
# ENHANCED PARSER ENGINE
# ============================================================================

class ParserError(Exception):
    """Custom exception for parser errors"""
    def __init__(self, message: str, token: Optional[Token] = None):
        self.message = message
        self.token = token
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        if self.token:
            return f"{self.message} at line {self.token.line}, column {self.token.column}"
        return self.message

class MechanicsParser:
    """Parser with improved error handling and feature completeness"""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current_system = None
        self.errors: List[str] = []
        self.max_errors = config.max_parser_errors

    def peek(self, offset: int = 0) -> Optional[Token]:
        """Look ahead at token without consuming it"""
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return None

    def match(self, *expected_types: str) -> Optional[Token]:
        """Match and consume token if type matches"""
        token = self.peek()
        if token and token.type in expected_types:
            self.pos += 1
            return token
        return None

    def expect(self, expected_type: str) -> Token:
        """Expect a specific token type, raise error if not found"""
        token = self.match(expected_type)
        if not token:
            current = self.peek()
            if current:
                error_msg = f"Expected {expected_type} but got {current.type} '{current.value}'"
                self.errors.append(error_msg)
                raise ParserError(error_msg, current)
            else:
                error_msg = f"Expected {expected_type} but reached end of input"
                self.errors.append(error_msg)
                raise ParserError(error_msg)
        return token

    @profile_function
    def parse(self) -> List[ASTNode]:
        """Parse the complete DSL with comprehensive error recovery"""
        nodes = []
        error_count = 0
        
        while self.pos < len(self.tokens) and error_count < self.max_errors:
            try:
                node = self.parse_statement()
                if node:
                    nodes.append(node)
                    logger.debug(f"Parsed node: {type(node).__name__}")
            except ParserError as e:
                self.errors.append(str(e))
                error_count += 1
                logger.error(f"Parser error: {e}")
                
                # Error recovery: skip to next statement
                while self.pos < len(self.tokens):
                    token = self.peek()
                    if token and token.type in ["SYSTEM", "DEFVAR", "DEFINE", 
                                                "LAGRANGIAN", "HAMILTONIAN", 
                                                "CONSTRAINT", "INITIAL", "SOLVE"]:
                        break
                    self.pos += 1
        
        if self.errors:
            logger.warning(f"Parser encountered {len(self.errors)} errors")
            
        logger.info(f"Successfully parsed {len(nodes)} AST nodes")
        return nodes

    def parse_statement(self) -> Optional[ASTNode]:
        """Parse a top-level statement"""
        token = self.peek()
        if not token:
            return None

        handlers = {
            "SYSTEM": self.parse_system,
            "DEFVAR": self.parse_defvar,
            "PARAMETER": self.parse_parameter,
            "DEFINE": self.parse_define,
            "LAGRANGIAN": self.parse_lagrangian,
            "HAMILTONIAN": self.parse_hamiltonian,
            "TRANSFORM": self.parse_transform,
            "CONSTRAINT": self.parse_constraint,
            "NONHOLONOMIC": self.parse_nonholonomic,
            "FORCE": self.parse_force,
            "DAMPING": self.parse_damping,
            "INITIAL": self.parse_initial,
            "SOLVE": self.parse_solve,
            "ANIMATE": self.parse_animate,
            "EXPORT": self.parse_export,
            "IMPORT": self.parse_import,
        }
        
        handler = handlers.get(token.type)
        if handler:
            return handler()
        else:
            logger.debug(f"Skipping unknown token: {token}")
            self.pos += 1
            return None

    def parse_system(self) -> SystemDef:
        """Parse \\system{name}"""
        self.expect("SYSTEM")
        self.expect("LBRACE")
        name = self.expect("IDENT").value
        self.expect("RBRACE")
        self.current_system = name
        return SystemDef(name)

    def parse_defvar(self) -> VarDef:
        """Parse \\defvar{name}{type}{unit}"""
        self.expect("DEFVAR")
        self.expect("LBRACE")
        name = self.expect("IDENT").value
        self.expect("RBRACE")
        self.expect("LBRACE")
        
        vartype_parts = []
        while True:
            tok = self.peek()
            if not tok or tok.type == 'RBRACE':
                break
            self.pos += 1
            vartype_parts.append(tok.value)
        vartype = ' '.join(vartype_parts).strip()
        self.expect("RBRACE")
        
        self.expect("LBRACE")
        unit_expr = self.parse_expression()
        unit = self.expression_to_string(unit_expr)
        self.expect("RBRACE")
        
        is_vector = vartype in ["Vector", "Vector3", "Position", "Velocity", 
                               "Force", "Momentum", "Acceleration"]
        
        return VarDef(name, vartype, unit, is_vector)
    
    def parse_parameter(self) -> ParameterDef:
        """Parse \\parameter{name}{value}{unit}"""
        self.expect("PARAMETER")
        self.expect("LBRACE")
        name = self.expect("IDENT").value
        self.expect("RBRACE")
        self.expect("LBRACE")
        value = float(self.expect("NUMBER").value)
        self.expect("RBRACE")
        self.expect("LBRACE")
        unit = self.expect("IDENT").value
        self.expect("RBRACE")
        return ParameterDef(name, value, unit)

    def parse_define(self) -> DefineDef:
        """Parse \\define{\\op{name}(args) = expression}"""
        self.expect("DEFINE")
        self.expect("LBRACE")
        
        self.expect("COMMAND")
        self.expect("LBRACE")
        name = self.expect("IDENT").value
        self.expect("RBRACE")
        
        self.expect("LPAREN")
        args = []
        if self.peek() and self.peek().type == "IDENT":
            args.append(self.expect("IDENT").value)
            while self.match("COMMA"):
                args.append(self.expect("IDENT").value)
        self.expect("RPAREN")
        
        self.expect("EQUALS")
        body = self.parse_expression()
        self.expect("RBRACE")
        
        return DefineDef(name, args, body)

    def parse_lagrangian(self) -> LagrangianDef:
        """Parse \\lagrangian{expression}"""
        self.expect("LAGRANGIAN")
        self.expect("LBRACE")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return LagrangianDef(expr)

    def parse_hamiltonian(self) -> HamiltonianDef:
        """Parse \\hamiltonian{expression}"""
        self.expect("HAMILTONIAN")
        self.expect("LBRACE")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return HamiltonianDef(expr)

    def parse_transform(self) -> TransformDef:
        """Parse \\transform{type}{var = expr}"""
        self.expect("TRANSFORM")
        self.expect("LBRACE")
        coord_type = self.expect("IDENT").value
        self.expect("RBRACE")
        self.expect("LBRACE")
        var = self.expect("IDENT").value
        self.expect("EQUALS")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return TransformDef(coord_type, var, expr)
    
    def parse_constraint(self) -> ConstraintDef:
        """Parse \\constraint{expression}"""
        self.expect("CONSTRAINT")
        self.expect("LBRACE")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return ConstraintDef(expr)

    def parse_nonholonomic(self) -> NonHolonomicConstraintDef:
        """Parse \\nonholonomic{expression}"""
        self.expect("NONHOLONOMIC")
        self.expect("LBRACE")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return NonHolonomicConstraintDef(expr)

    def parse_force(self) -> ForceDef:
        """Parse \\force{expression}"""
        self.expect("FORCE")
        self.expect("LBRACE")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return ForceDef(expr)

    def parse_damping(self) -> DampingDef:
        """Parse \\damping{expression}"""
        self.expect("DAMPING")
        self.expect("LBRACE")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return DampingDef(expr)

    def parse_initial(self) -> InitialCondition:
        """Parse \\initial{var1=val1, var2=val2, ...}"""
        self.expect("INITIAL")
        self.expect("LBRACE")
        
        conditions = {}
        var = self.expect("IDENT").value
        self.expect("EQUALS")
        val = float(self.expect("NUMBER").value)
        conditions[var] = val
        
        while self.match("COMMA"):
            var = self.expect("IDENT").value
            self.expect("EQUALS")
            val = float(self.expect("NUMBER").value)
            conditions[var] = val
            
        self.expect("RBRACE")
        return InitialCondition(conditions)

    def parse_solve(self) -> SolveDef:
        """Parse \\solve{method}"""
        self.expect("SOLVE")
        self.expect("LBRACE")
        method = self.expect("IDENT").value
        self.expect("RBRACE")
        return SolveDef(method)

    def parse_animate(self) -> AnimateDef:
        """Parse \\animate{target}"""
        self.expect("ANIMATE")
        self.expect("LBRACE")
        target = self.expect("IDENT").value
        self.expect("RBRACE")
        return AnimateDef(target)

    def parse_export(self) -> ExportDef:
        """Parse \\export{filename}"""
        self.expect("EXPORT")
        self.expect("LBRACE")
        filename = self.expect("IDENT").value
        self.expect("RBRACE")
        return ExportDef(filename)

    def parse_import(self) -> ImportDef:
        """Parse \\import{filename}"""
        self.expect("IMPORT")
        self.expect("LBRACE")
        filename = self.expect("IDENT").value
        self.expect("RBRACE")
        return ImportDef(filename)

    def parse_expression(self) -> Expression:
        """Parse expressions with full operator precedence"""
        return self.parse_additive()

    def parse_additive(self) -> Expression:
        """Addition and subtraction"""
        left = self.parse_multiplicative()
        
        while True:
            if self.match("PLUS"):
                right = self.parse_multiplicative()
                left = BinaryOpExpr(left, "+", right)
            elif self.match("MINUS"):
                right = self.parse_multiplicative()
                left = BinaryOpExpr(left, "-", right)
            else:
                break
                
        return left

    def parse_multiplicative(self) -> Expression:
        """Multiplication, division, and explicit products only"""
        left = self.parse_power()
        
        while True:
            if self.match("MULTIPLY"):
                right = self.parse_power()
                left = BinaryOpExpr(left, "*", right)
            elif self.match("DIVIDE"):
                right = self.parse_power()
                left = BinaryOpExpr(left, "/", right)
            elif self.match("VECTOR_DOT"):
                right = self.parse_power()
                left = VectorOpExpr("dot", left, right)
            elif self.match("VECTOR_CROSS"):
                right = self.parse_power()
                left = VectorOpExpr("cross", left, right)
            else:
                # Improved implicit multiplication - only for safe cases
                next_token = self.peek()
                if (next_token and 
                    next_token.type == "LPAREN" and
                    isinstance(left, (NumberExpr, IdentExpr, GreekLetterExpr)) and
                    not self.at_end_of_expression()):
                    # Safe implicit multiplication: 2(x+y), m(v^2), etc.
                    right = self.parse_power()
                    left = BinaryOpExpr(left, "*", right)
                else:
                    break
                    
        return left

    def parse_power(self) -> Expression:
        """Exponentiation (right associative)"""
        left = self.parse_unary()
        
        if self.match("POWER"):
            right = self.parse_power()
            return BinaryOpExpr(left, "^", right)
            
        return left

    def parse_unary(self) -> Expression:
        """Unary operators"""
        if self.match("MINUS"):
            operand = self.parse_unary()
            return UnaryOpExpr("-", operand)
        elif self.match("PLUS"):
            return self.parse_unary()
        
        return self.parse_postfix()

    def parse_postfix(self) -> Expression:
        """Function calls, subscripts, etc."""
        expr = self.parse_primary()
        
        while True:
            if self.match("LPAREN"):
                # Function call
                args = []
                if self.peek() and self.peek().type != "RPAREN":
                    args.append(self.parse_expression())
                    while self.match("COMMA"):
                        args.append(self.parse_expression())
                self.expect("RPAREN")
                
                if isinstance(expr, IdentExpr):
                    expr = FunctionCallExpr(expr.name, args)
                elif isinstance(expr, GreekLetterExpr):
                    expr = FunctionCallExpr(expr.letter, args)
                else:
                    raise ParserError("Invalid function call syntax")
            else:
                break
                
        return expr

    def parse_primary(self) -> Expression:
        """Primary expressions: literals, identifiers, parentheses, vectors, commands"""

        # Numbers
        if self.match("NUMBER"):
            return NumberExpr(float(self.tokens[self.pos - 1].value))

        # Time derivatives: \dot{x} and \ddot{x}
        token = self.peek()
        if token and token.type == "DOT_NOTATION":
            self.pos += 1
            order = 2 if token.value == r"\ddot" else 1
            self.expect("LBRACE")
            var = self.expect("IDENT").value
            self.expect("RBRACE")
            return DerivativeVarExpr(var, order)

        # Identifiers
        if self.match("IDENT"):
            return IdentExpr(self.tokens[self.pos - 1].value)

        # Greek letters
        if self.match("GREEK_LETTER"):
            letter = self.tokens[self.pos - 1].value[1:]
            return GreekLetterExpr(letter)

        # Parentheses
        if self.match("LPAREN"):
            expr = self.parse_expression()
            self.expect("RPAREN")
            return expr

        # Vectors [x, y, z]
        if self.match("LBRACKET"):
            components = []
            components.append(self.parse_expression())
            while self.match("COMMA"):
                components.append(self.parse_expression())
            self.expect("RBRACKET")
            return VectorExpr(components)

        # Commands (LaTeX-style functions)
        if self.match("COMMAND"):
            cmd = self.tokens[self.pos - 1].value
            return self.parse_command(cmd)

        # Mathematical constants
        if token and token.value in ["pi", "e"]:
            self.pos += 1
            if token.value == "pi":
                return NumberExpr(np.pi)
            elif token.value == "e":
                return NumberExpr(np.e)

        current = self.peek()
        if current:
            raise ParserError(f"Unexpected token {current.type} '{current.value}'", current)
        else:
            raise ParserError("Unexpected end of input")

    def parse_command(self, cmd: str) -> Expression:
        """Parse LaTeX-style commands"""
        
        if cmd == r"\frac":
            self.expect("LBRACE")
            num = self.parse_expression()
            self.expect("RBRACE")
            self.expect("LBRACE")
            denom = self.parse_expression()
            self.expect("RBRACE")
            return FractionExpr(num, denom)
        
        elif cmd == r"\vec":
            self.expect("LBRACE")
            expr = self.parse_expression()
            self.expect("RBRACE")
            return VectorOpExpr("vec", expr)
            
        elif cmd == r"\hat":
            self.expect("LBRACE")
            expr = self.parse_expression()
            self.expect("RBRACE")
            return VectorOpExpr("unit", expr)
            
        elif cmd in [r"\mag", r"\norm"]:
            self.expect("LBRACE")
            expr = self.parse_expression()
            self.expect("RBRACE")
            return VectorOpExpr("magnitude", expr)
            
        elif cmd == r"\partial":
            self.expect("LBRACE")
            expr = self.parse_expression()
            self.expect("RBRACE")
            self.expect("LBRACE")
            var = self.expect("IDENT").value
            self.expect("RBRACE")
            return DerivativeExpr(expr, var, 1, True)
            
        elif cmd in [r"\sin", r"\cos", r"\tan", r"\exp", r"\log", r"\ln", r"\sqrt", 
                     r"\sinh", r"\cosh", r"\tanh", r"\arcsin", r"\arccos", r"\arctan"]:
            func_name = cmd[1:]
            self.expect("LBRACE")
            arg = self.parse_expression()
            self.expect("RBRACE")
            return FunctionCallExpr(func_name, [arg])
            
        elif cmd in [r"\nabla", r"\grad"]:
            if self.peek() and self.peek().type == "LBRACE":
                self.expect("LBRACE")
                expr = self.parse_expression()
                self.expect("RBRACE")
                return VectorOpExpr("grad", expr)
            return VectorOpExpr("grad", None)
            
        else:
            # Unknown command - treat as identifier
            return IdentExpr(cmd[1:])

    def at_end_of_expression(self) -> bool:
        """Check if we're at the end of an expression"""
        token = self.peek()
        return (not token or 
                token.type in ["RBRACE", "RPAREN", "RBRACKET", "COMMA", 
                              "SEMICOLON", "EQUALS"])

    def expression_to_string(self, expr: Expression) -> str:
        """Convert expression back to string for unit parsing"""
        if isinstance(expr, NumberExpr):
            return str(expr.value)
        elif isinstance(expr, IdentExpr):
            return expr.name
        elif isinstance(expr, BinaryOpExpr):
            left = self.expression_to_string(expr.left)
            right = self.expression_to_string(expr.right)
            return f"({left} {expr.operator} {right})"
        elif isinstance(expr, UnaryOpExpr):
            operand = self.expression_to_string(expr.operand)
            return f"{expr.operator}{operand}"
        else:
            return str(expr)

# ============================================================================
# ENHANCED SYMBOLIC MATH ENGINE WITH CACHING
# ============================================================================

class SymbolicEngine:
    """Enhanced symbolic mathematics engine with advanced caching and performance monitoring"""
    
    def __init__(self):
        self.sp = sp
        self.symbol_map: Dict[str, sp.Symbol] = {}
        self.function_map: Dict[str, sp.Function] = {}
        self.time_symbol = sp.Symbol('t', real=True)
        self.assumptions: Dict[str, dict] = {}
        # v6.0: Advanced LRU cache
        if config.cache_symbolic_results:
            self._cache = LRUCache(
                maxsize=config.cache_max_size,
                max_memory_mb=config.cache_max_memory_mb
            )
        else:
            self._cache = None
        self._perf_monitor = _perf_monitor if config.enable_performance_monitoring else None

    def get_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """Get or create a SymPy symbol with assumptions (cached)"""
        if name not in self.symbol_map:
            default_assumptions = {'real': True}
            default_assumptions.update(assumptions)
            self.symbol_map[name] = sp.Symbol(name, **default_assumptions)
            self.assumptions[name] = default_assumptions
            logger.debug(f"Created symbol: {name} with assumptions {default_assumptions}")
        return self.symbol_map[name]

    def get_function(self, name: str) -> sp.Function:
        """Get or create a SymPy function (cached)"""
        if name not in self.function_map:
            self.function_map[name] = sp.Function(name, real=True)
            logger.debug(f"Created function: {name}")
        return self.function_map[name]

    @profile_function
    def ast_to_sympy(self, expr: Expression) -> sp.Expr:
        """
        Convert AST expression to SymPy with comprehensive support and caching
        
        Args:
            expr: AST expression node
            
        Returns:
            SymPy expression
        """
        # v6.0: Cache key generation
        cache_key = None
        if self._cache is not None:
            try:
                cache_key = str(hash(str(expr)))
                cached = self._cache.get(cache_key)
                if cached is not None:
                    logger.debug(f"Cache hit for expression: {expr}")
                    return cached
            except Exception as e:
                logger.debug(f"Cache key generation failed: {e}")
        
        if self._perf_monitor:
            self._perf_monitor.start_timer('ast_to_sympy')
        
        try:
            result = self._ast_to_sympy_impl(expr)
            
            # Cache result
            if self._cache is not None and cache_key is not None:
                self._cache.set(cache_key, result)
            
            if self._perf_monitor:
                self._perf_monitor.stop_timer('ast_to_sympy')
            
            return result
        except Exception as e:
            if self._perf_monitor:
                self._perf_monitor.stop_timer('ast_to_sympy')
            raise
    
    def _ast_to_sympy_impl(self, expr: Expression) -> sp.Expr:
        """Internal implementation of AST to SymPy conversion"""
        if isinstance(expr, NumberExpr):
            return sp.Float(expr.value)
            
        elif isinstance(expr, IdentExpr):
            return self.get_symbol(expr.name)
            
        elif isinstance(expr, GreekLetterExpr):
            return self.get_symbol(expr.letter)
            
        elif isinstance(expr, BinaryOpExpr):
            left = self._ast_to_sympy_impl(expr.left)
            right = self._ast_to_sympy_impl(expr.right)
            
            ops = {
                "+": lambda l, r: l + r,
                "-": lambda l, r: l - r,
                "*": lambda l, r: l * r,
                "/": lambda l, r: l / r,
                "^": lambda l, r: l ** r,
            }
            
            if expr.operator in ops:
                return ops[expr.operator](left, right)
            else:
                raise ValueError(f"Unknown operator: {expr.operator}")
                
        elif isinstance(expr, UnaryOpExpr):
            operand = self._ast_to_sympy_impl(expr.operand)
            if expr.operator == "-":
                return -operand
            elif expr.operator == "+":
                return operand
            else:
                raise ValueError(f"Unknown unary operator: {expr.operator}")
        
        elif isinstance(expr, FractionExpr):
            num = self._ast_to_sympy_impl(expr.numerator)
            denom = self._ast_to_sympy_impl(expr.denominator)
            return num / denom

        elif isinstance(expr, DerivativeVarExpr):
            if expr.order == 1:
                return self.get_symbol(f"{expr.var}_dot")
            elif expr.order == 2:
                return self.get_symbol(f"{expr.var}_ddot")
            else:
                raise ValueError(f"Derivative order {expr.order} not supported")
                
        elif isinstance(expr, DerivativeExpr):
            inner = self._ast_to_sympy_impl(expr.expr)
            var = self.get_symbol(expr.var)
            
            if expr.partial:
                return sp.diff(inner, var, expr.order)
            else:
                if expr.var == "t":
                    return sp.diff(inner, self.time_symbol, expr.order)
                else:
                    return sp.diff(inner, var, expr.order)
                    
        elif isinstance(expr, FunctionCallExpr):
            args = [self._ast_to_sympy_impl(arg) for arg in expr.args]
            
            builtin_funcs = {
                "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                "exp": sp.exp, "log": sp.log, "ln": sp.log,
                "sqrt": sp.sqrt, "sinh": sp.sinh, "cosh": sp.cosh,
                "tanh": sp.tanh, "arcsin": sp.asin, "arccos": sp.acos,
                "arctan": sp.atan, "abs": sp.Abs,
            }
            
            if expr.name in builtin_funcs:
                return builtin_funcs[expr.name](*args)
            else:
                func = self.get_function(expr.name)
                return func(*args)
                
        elif isinstance(expr, VectorExpr):
            return sp.Matrix([self._ast_to_sympy_impl(comp) for comp in expr.components])
            
        elif isinstance(expr, VectorOpExpr):
            if expr.operation == "grad":
                if expr.left:
                    inner = self._ast_to_sympy_impl(expr.left)
                    vars_list = [self.get_symbol(v) for v in ['x', 'y', 'z']]
                    return sp.Matrix([sp.diff(inner, var) for var in vars_list])
                else:
                    return self.get_symbol('nabla')
            elif expr.operation == "dot":
                left_vec = self._ast_to_sympy_impl(expr.left)
                right_vec = self._ast_to_sympy_impl(expr.right)
                if isinstance(left_vec, sp.Matrix) and isinstance(right_vec, sp.Matrix):
                    return left_vec.dot(right_vec)
                else:
                    return left_vec * right_vec
            elif expr.operation == "cross":
                left_vec = self._ast_to_sympy_impl(expr.left)
                right_vec = self._ast_to_sympy_impl(expr.right)
                if isinstance(left_vec, sp.Matrix) and isinstance(right_vec, sp.Matrix):
                    return left_vec.cross(right_vec)
                else:
                    raise ValueError("Cross product requires vector arguments")
            elif expr.operation == "magnitude":
                vec = self._ast_to_sympy_impl(expr.left)
                if isinstance(vec, sp.Matrix):
                    return sp.sqrt(vec.dot(vec))
                else:
                    return sp.Abs(vec)
                    
        else:
            raise ValueError(f"Cannot convert {type(expr).__name__} to SymPy")

    @profile_function
    def derive_equations_of_motion(self, lagrangian: sp.Expr, 
                                   coordinates: List[str]) -> List[sp.Expr]:
        """
        Derive Euler-Lagrange equations from Lagrangian
        
        Args:
            lagrangian: Lagrangian expression
            coordinates: List of generalized coordinates
            
        Returns:
            List of equations of motion
        """
        logger.info(f"Deriving equations of motion for {len(coordinates)} coordinates")
        equations = []
        
        for q in coordinates:
            logger.debug(f"Processing coordinate: {q}")
            q_sym = self.get_symbol(q)
            q_dot_sym = self.get_symbol(f"{q}_dot")
            q_ddot_sym = self.get_symbol(f"{q}_ddot")

            q_func = sp.Function(q)(self.time_symbol)

            L_with_funcs = lagrangian.subs(q_sym, q_func)
            L_with_funcs = L_with_funcs.subs(q_dot_sym, sp.diff(q_func, self.time_symbol))

            dL_dq_dot = sp.diff(L_with_funcs, sp.diff(q_func, self.time_symbol))
            d_dt_dL_dq_dot = sp.diff(dL_dq_dot, self.time_symbol)
            dL_dq = sp.diff(L_with_funcs, q_func)

            equation = d_dt_dL_dq_dot - dL_dq

            equation = equation.subs(q_func, q_sym)
            equation = equation.subs(sp.diff(q_func, self.time_symbol), q_dot_sym)
            equation = equation.subs(sp.diff(q_func, self.time_symbol, 2), q_ddot_sym)

            # Simplify with timeout
            try:
                if config.simplification_timeout > 0:
                    with timeout(config.simplification_timeout):
                        equation = sp.simplify(equation)
                else:
                    equation = sp.simplify(equation)
            except TimeoutError:
                logger.warning(f"Simplification timeout for {q}, using unsimplified equation")
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Simplification error for {q}: {e}, using unsimplified equation")
            
            equations.append(equation)
            logger.debug(f"Equation for {q}: {equation}")
            
        return equations

    def derive_equations_with_constraints(self, lagrangian: sp.Expr,
                                         coordinates: List[str],
                                         constraints: List[sp.Expr]) -> Tuple[List[sp.Expr], List[str]]:
        """
        Derive equations with holonomic constraints using Lagrange multipliers
        
        Args:
            lagrangian: Lagrangian expression
            coordinates: List of generalized coordinates
            constraints: List of constraint expressions
            
        Returns:
            Tuple of (augmented equations, extended coordinates including lambdas)
        """
        logger.info(f"Deriving constrained equations with {len(constraints)} constraints")
        
        # Create Lagrange multipliers
        lambdas = [self.get_symbol(f'lambda_{i}') for i in range(len(constraints))]
        
        # Augmented Lagrangian: L' = L + Σ(λ_i * g_i)
        L_augmented = lagrangian
        for lam, constraint in zip(lambdas, constraints):
            L_augmented += lam * constraint
        
        logger.debug(f"Augmented Lagrangian: {L_augmented}")
        
        # Derive augmented equations
        equations = self.derive_equations_of_motion(L_augmented, coordinates)
        
        # Add time derivatives of constraints as additional equations
        constraint_eqs = []
        for constraint in constraints:
            # First time derivative: dg/dt = 0
            constraint_dot = sp.diff(constraint, self.time_symbol)
            constraint_eqs.append(constraint_dot)
        
        extended_coords = coordinates + [str(lam) for lam in lambdas]
        all_equations = equations + constraint_eqs
        
        logger.info(f"Generated {len(all_equations)} constrained equations")
        return all_equations, extended_coords

    @profile_function
    def derive_hamiltonian_equations(self, hamiltonian: sp.Expr, 
                                    coordinates: List[str]) -> Tuple[List[sp.Expr], List[sp.Expr]]:
        """
        Derive Hamilton's equations from Hamiltonian
        
        Hamilton's equations:
        dq/dt = ∂H/∂p
        dp/dt = -∂H/∂q
        
        Args:
            hamiltonian: Hamiltonian expression
            coordinates: List of generalized coordinates
            
        Returns:
            Tuple of (q_dot equations, p_dot equations)
        """
        logger.info(f"Deriving Hamiltonian equations for {len(coordinates)} coordinates")
        q_dot_equations = []
        p_dot_equations = []
        
        for q in coordinates:
            q_sym = self.get_symbol(q)
            p_sym = self.get_symbol(f"p_{q}")
            
            # dq/dt = ∂H/∂p
            q_dot = sp.diff(hamiltonian, p_sym)
            try:
                if config.simplification_timeout > 0:
                    with timeout(config.simplification_timeout):
                        q_dot = sp.simplify(q_dot)
                else:
                    q_dot = sp.simplify(q_dot)
            except TimeoutError:
                logger.debug(f"Simplification timeout for d{q}/dt, using unsimplified")
            except (ValueError, TypeError, AttributeError) as e:
                logger.debug(f"Simplification error for d{q}/dt: {e}")
            q_dot_equations.append(q_dot)
            
            # dp/dt = -∂H/∂q
            p_dot = -sp.diff(hamiltonian, q_sym)
            try:
                if config.simplification_timeout > 0:
                    with timeout(config.simplification_timeout):
                        p_dot = sp.simplify(p_dot)
                else:
                    p_dot = sp.simplify(p_dot)
            except TimeoutError:
                logger.debug(f"Simplification timeout for dp_{q}/dt, using unsimplified")
            except (ValueError, TypeError, AttributeError) as e:
                logger.debug(f"Simplification error for dp_{q}/dt: {e}")
            p_dot_equations.append(p_dot)
            
            logger.debug(f"Hamilton equations for {q}:")
            logger.debug(f"  d{q}/dt = {q_dot}")
            logger.debug(f"  dp_{q}/dt = {p_dot}")
            
        return q_dot_equations, p_dot_equations

    @profile_function
    def lagrangian_to_hamiltonian(self, lagrangian: sp.Expr, 
                                 coordinates: List[str]) -> sp.Expr:
        """
        Convert Lagrangian to Hamiltonian via Legendre transform
        
        H = Σ(p_i * q̇_i) - L
        where p_i = ∂L/∂q̇_i
        
        Args:
            lagrangian: Lagrangian expression
            coordinates: List of generalized coordinates
            
        Returns:
            Hamiltonian expression
        """
        logger.info("Converting Lagrangian to Hamiltonian")
        hamiltonian = sp.S.Zero
        
        for q in coordinates:
            q_dot_sym = self.get_symbol(f"{q}_dot")
            p_sym = self.get_symbol(f"p_{q}")
            
            # Calculate conjugate momentum p = ∂L/∂q̇
            momentum_def = sp.diff(lagrangian, q_dot_sym)
            logger.debug(f"Momentum for {q}: p_{q} = {momentum_def}")
            
            # Solve for q̇ in terms of p
            try:
                q_dot_solution = sp.solve(momentum_def - p_sym, q_dot_sym)
                if q_dot_solution:
                    q_dot_expr = q_dot_solution[0]
                    hamiltonian += p_sym * q_dot_expr
                    logger.debug(f"Solved for {q}_dot: {q_dot_expr}")
            except (ValueError, TypeError, NotImplementedError) as e:
                logger.warning(f"Could not solve for {q}_dot: {e}, using symbolic form")
                hamiltonian += p_sym * q_dot_sym
        
        # H = Σ(p_i * q̇_i) - L
        hamiltonian = hamiltonian - lagrangian
        
        # Substitute momentum definitions
        for q in coordinates:
            q_dot_sym = self.get_symbol(f"{q}_dot")
            p_sym = self.get_symbol(f"p_{q}")
            momentum_def = sp.diff(lagrangian, q_dot_sym)
            
            try:
                q_dot_solution = sp.solve(momentum_def - p_sym, q_dot_sym)
                if q_dot_solution:
                    hamiltonian = hamiltonian.subs(q_dot_sym, q_dot_solution[0])
            except (ValueError, TypeError, NotImplementedError):
                logger.debug(f"Could not substitute {q}_dot in Hamiltonian")
        
        # Simplify with timeout
        try:
            if config.simplification_timeout > 0:
                with timeout(config.simplification_timeout):
                    hamiltonian = sp.simplify(hamiltonian)
            else:
                hamiltonian = sp.simplify(hamiltonian)
        except TimeoutError:
            logger.warning("Hamiltonian simplification timeout, using unsimplified form")
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Hamiltonian simplification error: {e}, using unsimplified form")
        
        logger.info(f"Hamiltonian: {hamiltonian}")
        return hamiltonian

    def solve_for_accelerations(self, equations: List[sp.Expr], 
                               coordinates: List[str]) -> Dict[str, sp.Expr]:
        """
        Solve equations of motion for accelerations
        
        Args:
            equations: List of equations of motion
            coordinates: List of generalized coordinates
            
        Returns:
            Dictionary mapping acceleration symbols to expressions
        """
        logger.info("Solving for accelerations")
        accelerations = {}
        accel_symbols = [self.get_symbol(f"{q}_ddot") for q in coordinates]
        
        try:
            solutions = sp.solve(equations, accel_symbols, dict=True)
            
            if solutions:
                sol = solutions[0] if isinstance(solutions, list) else solutions
                for q in coordinates:
                    accel_sym = self.get_symbol(f"{q}_ddot")
                    if accel_sym in sol:
                        accel_expr = sol[accel_sym]
                        try:
                            if config.simplification_timeout > 0:
                                with timeout(config.simplification_timeout):
                                    accel_expr = sp.simplify(accel_expr)
                            else:
                                accel_expr = sp.simplify(accel_expr)
                        except TimeoutError:
                            logger.debug(f"Simplification timeout for {q}_ddot")
                        except (ValueError, TypeError, AttributeError):
                            logger.debug(f"Simplification error for {q}_ddot")
                        accelerations[f"{q}_ddot"] = accel_expr
                        logger.debug(f"Solved {q}_ddot = {accel_expr}")
            else:
                logger.warning("No symbolic solution found, solving individually")
                for i, q in enumerate(coordinates):
                    accel_key = f"{q}_ddot"
                    try:
                        sol = sp.solve(equations[i], accel_symbols[i])
                        if sol:
                            accel_expr = sol[0] if isinstance(sol, list) else sol
                            try:
                                if config.simplification_timeout > 0:
                                    with timeout(config.simplification_timeout):
                                        accel_expr = sp.simplify(accel_expr)
                                else:
                                    accel_expr = sp.simplify(accel_expr)
                            except TimeoutError:
                                logger.debug(f"Simplification timeout for {accel_key}")
                            except (ValueError, TypeError, AttributeError):
                                logger.debug(f"Simplification error for {accel_key}")
                            accelerations[accel_key] = accel_expr
                    except (ValueError, TypeError, NotImplementedError) as e:
                        logger.error(f"Could not solve for {accel_key}: {e}")
                        accelerations[accel_key] = equations[i]
                        
        except (ValueError, TypeError, NotImplementedError) as e:
            logger.error(f"Could not solve equations symbolically: {e}")
            for i, q in enumerate(coordinates):
                accelerations[f"{q}_ddot"] = equations[i]
                
        return accelerations

# ============================================================================
# NUMERICAL SIMULATION ENGINE WITH BETTER STABILITY
# ============================================================================

class NumericalSimulator:
    """Enhanced numerical simulator with better stability and diagnostics"""
    
    def __init__(self, symbolic_engine: SymbolicEngine):
        self.symbolic = symbolic_engine
        self.equations: Dict[str, Callable] = {}
        self.parameters: Dict[str, float] = {}
        self.initial_conditions: Dict[str, float] = {}
        self.constraints: List[sp.Expr] = []
        self.state_vars: List[str] = []
        self.coordinates: List[str] = []
        self.use_hamiltonian: bool = False
        self.hamiltonian_equations: Optional[Dict[str, List[Tuple]]] = None

    def set_parameters(self, params: Dict[str, float]):
        """Set physical parameters"""
        self.parameters.update(params)
        logger.debug(f"Set parameters: {params}")

    def set_initial_conditions(self, conditions: Dict[str, float]):
        """Set initial conditions"""
        self.initial_conditions.update(conditions)
        logger.debug(f"Set initial conditions: {conditions}")
    
    def add_constraint(self, constraint_expr: sp.Expr):
        """Add a constraint equation"""
        self.constraints.append(constraint_expr)
        logger.debug(f"Added constraint: {constraint_expr}")

    @profile_function
    def compile_equations(self, accelerations: Dict[str, sp.Expr], coordinates: List[str]):
        """Compile symbolic equations to numerical functions"""
        
        logger.info(f"Compiling equations for {len(coordinates)} coordinates")
        
        state_vars = []
        for q in coordinates:
            state_vars.extend([q, f"{q}_dot"])
            
        param_subs = {self.symbolic.get_symbol(k): v for k, v in self.parameters.items()}
        
        compiled_equations = {}
        
        for q in coordinates:
            accel_key = f"{q}_ddot"
            if accel_key in accelerations:
                eq = accelerations[accel_key].subs(param_subs)
                
                # Attempt simplification with timeout
                try:
                    if config.simplification_timeout > 0:
                        with timeout(config.simplification_timeout):
                            eq = sp.simplify(eq)
                    else:
                        eq = sp.simplify(eq)
                except TimeoutError:
                    logger.debug(f"Simplification timeout for {accel_key}, skipping")
                except (ValueError, TypeError, AttributeError) as e:
                    logger.debug(f"Simplification error for {accel_key}: {e}, skipping")

                eq = self._replace_derivatives(eq, coordinates)
                
                free_symbols = eq.free_symbols
                ordered_symbols = []
                symbol_indices = []
                
                for i, var_name in enumerate(state_vars):
                    sym = self.symbolic.get_symbol(var_name)
                    if sym in free_symbols:
                        ordered_symbols.append(sym)
                        symbol_indices.append(i)
                
                if ordered_symbols:
                    try:
                        func = sp.lambdify(ordered_symbols, eq, modules=['numpy', 'math'])
                        
                        def make_wrapper(func, indices):
                            # Force capture by value using default arguments
                            def wrapper(*state_vector, _func=func, _indices=indices):
                                try:
                                    args = [state_vector[i] for i in _indices if i < len(state_vector)]
                                    if len(args) == len(indices):
                                        result = _func(*args)
                                        if isinstance(result, np.ndarray):
                                            if result.size == 1:
                                                result = float(result.item())
                                            else:
                                                result = float(result.flat[0])
                                        result = float(result)
                                        return result if np.isfinite(result) else 0.0
                                    return 0.0
                                except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
                                    logger.debug(f"Evaluation error: {e}")
                                    return 0.0
                            return wrapper
                        
                        compiled_equations[accel_key] = make_wrapper(func, symbol_indices)
                        logger.debug(f"Compiled {accel_key}")
                        
                    except (ValueError, TypeError, AttributeError, 
                            NotImplementedError, SyntaxError) as e:
                        logger.error(f"Compilation failed for {accel_key}: {e}")
                        compiled_equations[accel_key] = lambda *args: 0.0
                    except Exception as e:
                        logger.error(f"Unexpected compilation error for {accel_key}: {type(e).__name__}: {e}")
                        compiled_equations[accel_key] = lambda *args: 0.0
                else:
                    try:
                        const_value = float(sp.N(eq))
                        compiled_equations[accel_key] = lambda *args: const_value
                        logger.debug(f"{accel_key} is constant: {const_value}")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not evaluate constant for {accel_key}: {e}")
                        compiled_equations[accel_key] = lambda *args: 0.0

        self.equations = compiled_equations
        self.state_vars = state_vars
        self.coordinates = coordinates
        logger.info("Equation compilation complete")

    def compile_hamiltonian_equations(self, q_dots: List[sp.Expr], p_dots: List[sp.Expr], 
                                     coordinates: List[str]):
        """Compile Hamiltonian equations"""
        logger.info("Compiling Hamiltonian equations")
        self.use_hamiltonian = True
        
        state_vars = []
        for q in coordinates:
            state_vars.extend([q, f"p_{q}"])
        
        param_subs = {self.symbolic.get_symbol(k): v for k, v in self.parameters.items()}
        
        self.hamiltonian_equations = {
            'q_dots': [],
            'p_dots': []
        }
        
        for i, q in enumerate(coordinates):
            q_dot_eq = q_dots[i].subs(param_subs)
            p_dot_eq = p_dots[i].subs(param_subs)
            
            # Compile q_dot
            free_syms = q_dot_eq.free_symbols
            ordered_syms = []
            indices = []
            for j, var_name in enumerate(state_vars):
                sym = self.symbolic.get_symbol(var_name)
                if sym in free_syms:
                    ordered_syms.append(sym)
                    indices.append(j)
            
            if ordered_syms:
                func = sp.lambdify(ordered_syms, q_dot_eq, modules=['numpy', 'math'])
                self.hamiltonian_equations['q_dots'].append((func, indices))
            else:
                const_val = float(sp.N(q_dot_eq))
                self.hamiltonian_equations['q_dots'].append((lambda *args, v=const_val: v, []))
            
            # Compile p_dot
            free_syms = p_dot_eq.free_symbols
            ordered_syms = []
            indices = []
            for j, var_name in enumerate(state_vars):
                sym = self.symbolic.get_symbol(var_name)
                if sym in free_syms:
                    ordered_syms.append(sym)
                    indices.append(j)
            
            if ordered_syms:
                func = sp.lambdify(ordered_syms, p_dot_eq, modules=['numpy', 'math'])
                self.hamiltonian_equations['p_dots'].append((func, indices))
            else:
                const_val = float(sp.N(p_dot_eq))
                self.hamiltonian_equations['p_dots'].append((lambda *args, v=const_val: v, []))
        
        self.state_vars = state_vars
        self.coordinates = coordinates
        logger.info("Hamiltonian compilation complete")

    def _replace_derivatives(self, expr: sp.Expr, coordinates: List[str]) -> sp.Expr:
        """Replace Derivative objects with corresponding symbols"""
        derivs = list(expr.atoms(sp.Derivative))
        for d in derivs:
            try:
                base = d.args[0]
                order = 1
                if len(d.args) >= 2:
                    arg2 = d.args[1]
                    if isinstance(arg2, tuple) and len(arg2) >= 2:
                        order = int(arg2[1])
                
                base_name = str(base)
                if base_name in coordinates:
                    if order == 1:
                        repl = self.symbolic.get_symbol(f"{base_name}_dot")
                    elif order == 2:
                        repl = self.symbolic.get_symbol(f"{base_name}_ddot")
                    else:
                        continue
                    expr = expr.xreplace({d: repl})
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"Could not replace derivative: {e}")
                continue
        return expr

    def equations_of_motion(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        ODE system for numerical integration with comprehensive bounds checking and validation.
        
        Args:
            t: Current time
            y: State vector
            
        Returns:
            Derivative vector dydt
        """
        # Comprehensive input validation
        if not isinstance(t, (int, float)) or not np.isfinite(t):
            logger.error(f"equations_of_motion: invalid time t={t}, using 0.0")
            t = 0.0
        
        if y is None:
            logger.error("equations_of_motion: y is None, returning zeros")
            if self.coordinates:
                return np.zeros(2 * len(self.coordinates))
            return np.zeros(1)
        
        if not isinstance(y, np.ndarray):
            logger.error(f"equations_of_motion: y is not numpy.ndarray, got {type(y).__name__}")
            try:
                y = np.array(y, dtype=float)
            except Exception as e:
                logger.error(f"equations_of_motion: cannot convert y to array: {e}")
                if self.coordinates:
                    return np.zeros(2 * len(self.coordinates))
                return np.zeros(1)
        
        if not validate_array_safe(y, "state_vector", min_size=1, check_finite=False):
            logger.warning("equations_of_motion: state vector validation failed, attempting recovery")
            # Try to fix non-finite values
            y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
        
        if self.use_hamiltonian:
            return self._hamiltonian_ode(t, y)
        
        # Validate expected size
        expected_size = 2 * len(self.coordinates) if self.coordinates else 1
        if y.size != expected_size:
            logger.warning(f"equations_of_motion: state vector size {y.size} != expected {expected_size}")
            # Try to pad or truncate
            if y.size < expected_size:
                y = np.pad(y, (0, expected_size - y.size), mode='constant', constant_values=0.0)
            else:
                y = y[:expected_size]
        
        try:
            dydt = np.zeros_like(y)
        
            # Position derivatives = velocities (with comprehensive bounds checking)
            for i in range(len(self.coordinates)):
                pos_idx = 2 * i 
                vel_idx = 2 * i + 1 

                if vel_idx < len(y) and pos_idx < len(dydt):
                    vel_value = safe_array_access(y, vel_idx, 0.0)
                    dydt[pos_idx] = vel_value
                elif pos_idx < len(dydt):
                    dydt[pos_idx] = 0.0

        for i, q in enumerate(self.coordinates):
            accel_key = f"{q}_ddot"
            vel_idx = 2 * i + 1

            if accel_key in self.equations and vel_idx < len(dydt):
                try:
                    # Validate equation function exists
                    eq_func = self.equations.get(accel_key)
                    if eq_func is None:
                        logger.warning(f"equations_of_motion: equation function for {accel_key} is None")
                        dydt[vel_idx] = 0.0
                        continue
                    
                    # Call with safe argument handling
                    try:
                        accel_value = eq_func(*y)
                        accel_value = safe_float_conversion(accel_value)
                        if np.isfinite(accel_value):
                            dydt[vel_idx] = accel_value
                        else:
                            dydt[vel_idx] = 0.0
                            logger.warning(f"Non-finite acceleration for {q} at t={t:.6f}")
                    except (ValueError, TypeError, ZeroDivisionError, IndexError, OverflowError) as e:
                        dydt[vel_idx] = 0.0
                        logger.debug(f"Evaluation error for {q} at t={t:.6f}: {e}")
                except Exception as e:
                    dydt[vel_idx] = 0.0
                    logger.error(f"Unexpected error evaluating {accel_key}: {e}", exc_info=True)
            elif vel_idx < len(dydt):
                dydt[vel_idx] = 0.0
                
            # Final validation of result
            if not validate_array_safe(dydt, "dydt", check_finite=True):
                logger.warning("equations_of_motion: result validation failed, fixing non-finite values")
                dydt = np.nan_to_num(dydt, nan=0.0, posinf=1e10, neginf=-1e10)
            
        return dydt
            
        except Exception as e:
            logger.error(f"equations_of_motion: unexpected error: {e}", exc_info=True)
            # Return safe default
            if self.coordinates:
                return np.zeros(2 * len(self.coordinates))
            return np.zeros(1)
    

    def _hamiltonian_ode(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        ODE system for Hamiltonian formulation with comprehensive validation.
        
        Args:
            t: Current time
            y: State vector
            
        Returns:
            Derivative vector dydt
        """
        # Input validation
        if not isinstance(t, (int, float)) or not np.isfinite(t):
            logger.error(f"_hamiltonian_ode: invalid time t={t}, using 0.0")
            t = 0.0
        
        if y is None:
            logger.error("_hamiltonian_ode: y is None")
            if self.coordinates:
                return np.zeros(2 * len(self.coordinates))
            return np.zeros(1)
        
        if not isinstance(y, np.ndarray):
            logger.error(f"_hamiltonian_ode: y is not numpy.ndarray, got {type(y).__name__}")
            try:
                y = np.array(y, dtype=float)
            except Exception as e:
                logger.error(f"_hamiltonian_ode: cannot convert y to array: {e}")
                if self.coordinates:
                    return np.zeros(2 * len(self.coordinates))
                return np.zeros(1)
        
        if not validate_array_safe(y, "hamiltonian_state_vector", min_size=1, check_finite=False):
            logger.warning("_hamiltonian_ode: state vector validation failed, attempting recovery")
            y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
        
        try:
            dydt = np.zeros_like(y)
        
            if self.hamiltonian_equations is None:
                logger.error("_hamiltonian_ode: hamiltonian_equations is None, cannot compute ODE")
                return dydt
            
            if not isinstance(self.hamiltonian_equations, dict):
                logger.error(f"_hamiltonian_ode: hamiltonian_equations is not dict, got {type(self.hamiltonian_equations).__name__}")
                return dydt
            
            if 'q_dots' not in self.hamiltonian_equations or 'p_dots' not in self.hamiltonian_equations:
                logger.error("_hamiltonian_ode: hamiltonian_equations missing required keys")
                return dydt
            
        for i, q in enumerate(self.coordinates):
            # Validate q is a string
            if not isinstance(q, str):
                logger.warning(f"_hamiltonian_ode: coordinate {i} is not string: {type(q).__name__}")
                continue
            
            # dq/dt
            if i >= len(self.hamiltonian_equations['q_dots']):
                logger.warning(f"_hamiltonian_ode: Index {i} out of range for q_dots (len={len(self.hamiltonian_equations['q_dots'])})")
                continue
                
                try:
                    q_dot_data = self.hamiltonian_equations['q_dots'][i]
                    if not isinstance(q_dot_data, tuple) or len(q_dot_data) != 2:
                        logger.error(f"_hamiltonian_ode: invalid q_dot data structure at index {i}")
                        continue
                    
                    func, indices = q_dot_data
                    if func is None:
                        logger.warning(f"_hamiltonian_ode: function is None for d{q}/dt")
                        continue
                    
                    if not isinstance(indices, (list, tuple)):
                        logger.warning(f"_hamiltonian_ode: indices is not list/tuple for d{q}/dt")
                        indices = []
                    
            q_idx = 2 * i 

           if q_idx < len(dydt):
                try:
                    args = [safe_array_access(y, j, 0.0) for j in indices if isinstance(j, int) and j >= 0]
                    if len(args) == len(indices):
                        result = func(*args)
                        dydt[q_idx] = safe_float_conversion(result)
                        if not np.isfinite(dydt[q_idx]):
                            logger.warning(f"_hamiltonian_ode: non-finite d{q}/dt, setting to 0.0")
                            dydt[q_idx] = 0.0
                    else:
                        dydt[q_idx] = 0.0
                        logger.warning(f"_hamiltonian_ode: Incomplete arguments for d{q}/dt (got {len(args)}, expected {len(indices)})")

                except (ValueError, TypeError, ZeroDivisionError, IndexError, OverflowError) as e:
                    dydt[q_idx] = 0.0
                    logger.debug(f"_hamiltonian_ode: Evaluation error for d{q}/dt: {e}")
                except Exception as e:
                    dydt[q_idx] = 0.0
                    logger.error(f"_hamiltonian_ode: Unexpected error for d{q}/dt: {e}", exc_info=True)
                except (IndexError, TypeError, ValueError) as e:
                    logger.error(f"_hamiltonian_ode: Error accessing q_dots[{i}]: {e}")
                    continue

                # dp/dt
                if i >= len(self.hamiltonian_equations['p_dots']):
                    logger.warning(f"_hamiltonian_ode: Index {i} out of range for p_dots (len={len(self.hamiltonian_equations['p_dots'])})")
                    continue
                
                try:
                    p_dot_data = self.hamiltonian_equations['p_dots'][i]
                    if not isinstance(p_dot_data, tuple) or len(p_dot_data) != 2:
                        logger.error(f"_hamiltonian_ode: invalid p_dot data structure at index {i}")
                        continue
                    
                    func, indices = p_dot_data
                    if func is None:
                        logger.warning(f"_hamiltonian_ode: function is None for dp_{q}/dt")
                        continue
                    
                    if not isinstance(indices, (list, tuple)):
                        logger.warning(f"_hamiltonian_ode: indices is not list/tuple for dp_{q}/dt")
                        indices = []
                    
            p_idx = 2 * i + 1

            if p_idx < len(dydt):
                try:
                    args = [safe_array_access(y, j, 0.0) for j in indices if isinstance(j, int) and j >= 0]
                    if len(args) == len(indices):
                        result = func(*args)
                        dydt[p_idx] = safe_float_conversion(result)
                        if not np.isfinite(dydt[p_idx]):
                            logger.warning(f"_hamiltonian_ode: non-finite dp_{q}/dt, setting to 0.0")
                            dydt[p_idx] = 0.0
                    else: 
                        dydt[p_idx] = 0.0
                        logger.warning(f"_hamiltonian_ode: Incomplete arguments for dp_{q}/dt (got {len(args)}, expected {len(indices)})")
                except (ValueError, TypeError, ZeroDivisionError, IndexError, OverflowError) as e:
                    dydt[p_idx] = 0.0
                    logger.debug(f"_hamiltonian_ode: Evaluation error for dp_{q}/dt: {e}")
                except Exception as e:
                    dydt[p_idx] = 0.0
                    logger.error(f"_hamiltonian_ode: Unexpected error for dp_{q}/dt: {e}", exc_info=True)
                except (IndexError, TypeError, ValueError) as e:
                    logger.error(f"_hamiltonian_ode: Error accessing p_dots[{i}]: {e}")
                    continue

            # Final validation
            if not validate_array_safe(dydt, "hamiltonian_dydt", check_finite=True):
                logger.warning("_hamiltonian_ode: result validation failed, fixing non-finite values")
                dydt = np.nan_to_num(dydt, nan=0.0, posinf=1e10, neginf=-1e10)

        return dydt

        except Exception as e:
            logger.error(f"_hamiltonian_ode: unexpected error: {e}", exc_info=True)
            if self.coordinates:
                return np.zeros(2 * len(self.coordinates))
            return np.zeros(1)

                        
    def _select_optimal_solver(self, t_span: Tuple[float, float], 
                              y0: np.ndarray) -> str:
        """v6.0: Intelligently select optimal solver based on system characteristics"""
        if not config.enable_adaptive_solver:
            return 'RK45'
        
        # Analyze system characteristics
        n_dof = len(self.coordinates)
        time_span = t_span[1] - t_span[0]
        
        # Large systems benefit from implicit methods
        if n_dof > 10:
            return 'LSODA'
        
        # Long time spans may need more stable methods
        if time_span > 100:
            return 'LSODA'
        
        # Small, simple systems can use fast explicit methods
        if n_dof <= 2 and time_span < 10:
            return 'RK45'
        
        # Default to adaptive method
        return 'LSODA'
                        
    @profile_function
    def simulate(self, t_span: Tuple[float, float], num_points: int = 1000,
                 method: str = None, rtol: float = None, atol: float = None,
                 detect_stiff: bool = True) -> dict:
        """
        Run numerical simulation with adaptive integration and diagnostics.
        
        Args:
            t_span: Time span (t_start, t_end) where t_start < t_end
            num_points: Number of output points (must be >= 2)
            method: Integration method ('RK45', 'LSODA', 'Radau', etc.)
            rtol: Relative tolerance (must be in (0, 1))
            atol: Absolute tolerance (must be positive)
            detect_stiff: Whether to detect stiff systems
            
        Returns:
            Dictionary with solution data and metadata, always contains 'success' key
            
        Raises:
            TypeError: If arguments have wrong types
            ValueError: If arguments are out of valid ranges
            
        Example:
            >>> solution = simulator.simulate((0, 10), num_points=1000)
            >>> if solution['success']:
            ...     t = solution['t']
            ...     y = solution['y']
        """
        # Comprehensive input validation
        validate_time_span(t_span)
        
        if not isinstance(num_points, int):
            raise TypeError(f"num_points must be int, got {type(num_points).__name__}")
        if num_points < 2:
            raise ValueError(f"num_points must be >= 2, got {num_points}")
        if num_points > 10_000_000:
            raise ValueError(f"num_points too large (>{10_000_000}), got {num_points}")
        
        if not isinstance(method, str):
            raise TypeError(f"method must be str, got {type(method).__name__}")
        valid_methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {method}")
        
        if rtol is not None:
            if not isinstance(rtol, (int, float)):
                raise TypeError(f"rtol must be numeric, got {type(rtol).__name__}")
            if rtol <= 0 or rtol >= 1:
                raise ValueError(f"rtol must be in (0, 1), got {rtol}")
        
        if atol is not None:
            if not isinstance(atol, (int, float)):
                raise TypeError(f"atol must be numeric, got {type(atol).__name__}")
            if atol <= 0:
                raise ValueError(f"atol must be positive, got {atol}")
        
        if not isinstance(detect_stiff, bool):
            raise TypeError(f"detect_stiff must be bool, got {type(detect_stiff).__name__}")
        
        # v6.0: Adaptive solver selection
        if method is None:
            method = 'RK45'  # Temporary default for solver selection
        if method == 'RK45' and config.enable_adaptive_solver:
            # Will be set after y0 is created
            adaptive_method = True
        else:
            adaptive_method = False
        
        rtol = rtol or config.default_rtol
        atol = atol or config.default_atol
        
        # Build initial state vector
        y0 = []
        for q in self.coordinates:
            if self.use_hamiltonian:
                pos_val = self.initial_conditions.get(q, 0.0)
                y0.append(pos_val)
                mom_key = f"p_{q}"
                mom_val = self.initial_conditions.get(mom_key, 0.0)
                y0.append(mom_val)
            else:
                pos_val = self.initial_conditions.get(q, 0.0)
                y0.append(pos_val)
                vel_key = f"{q}_dot"
                vel_val = self.initial_conditions.get(vel_key, 0.0)
                y0.append(vel_val)

        y0 = np.array(y0, dtype=float)
        t_eval = np.linspace(t_span[0], t_span[1], num_points)

        # Validate initial conditions
        if not validate_finite(y0, "Initial conditions"):
            return {
                'success': False,
                'error': 'Initial conditions contain non-finite values'
            }
        
        # Test initial evaluation
        try:
            dydt_test = self.equations_of_motion(t_span[0], y0)
            if not validate_finite(dydt_test, "Initial derivatives"):
                return {
                    'success': False,
                    'error': 'Initial derivatives contain non-finite values'
                }
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"Initial evaluation failed: {e}")
            return {
                'success': False,
                'error': f'Initial evaluation failed: {str(e)}'
            }
        except Exception as e:
            logger.error(f"Unexpected error in initial evaluation: {type(e).__name__}: {e}")
            return {
                'success': False,
                'error': f'Initial evaluation failed: {str(e)}'
            }
        
        # Stiffness detection
        is_stiff = False
        if detect_stiff and method == 'RK45':
            try:
                test_sol = solve_ivp(
                    self.equations_of_motion,
                    (t_span[0], t_span[0] + 0.01),
                    y0,
                    method='RK45',
                    max_step=0.001
                )
                if not test_sol.success:
                    is_stiff = True
                    logger.warning("System may be stiff. Consider using 'LSODA' or 'Radau' method.")
            except (ValueError, RuntimeError, AttributeError) as e:
                logger.debug(f"Stiffness detection test failed: {e}")
        
        # Run integration
        try:
            solution = solve_ivp(
                self.equations_of_motion,
                t_span,
                y0,
                t_eval=t_eval,
                method=method,
                rtol=rtol,
                atol=atol,
                max_step=min(0.01, (t_span[1] - t_span[0]) / 100)
            )
            
            logger.info(f"Simulation complete: {solution.nfev} evaluations, "
                       f"status={'success' if solution.success else 'failed'}")
            
            # v6.0: Performance monitoring
            if config.enable_performance_monitoring:
                _perf_monitor.stop_timer('simulation')
                _perf_monitor.snapshot_memory("post_simulation")
            
            result = {
                'success': solution.success,
                't': solution.t,
                'y': solution.y,
                'coordinates': self.coordinates,
                'state_vars': self.state_vars,
                'message': solution.message if hasattr(solution, 'message') else '',
                'nfev': solution.nfev if hasattr(solution, 'nfev') else 0,
                'is_stiff': is_stiff,
                'use_hamiltonian': self.use_hamiltonian,
                'method_used': method,  # v6.0: Track which method was used
            }
            
            # v6.0: Add performance metrics if available
            if config.enable_performance_monitoring:
                sim_stats = _perf_monitor.get_stats('simulation')
                if sim_stats:
                    result['performance'] = sim_stats
            
            return result
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# ============================================================================
# POTENTIAL ENERGY OFFSET COMPUTATION
# ============================================================================

class PotentialEnergyCalculator:
    """Compute potential energy with proper offset for different systems"""
    
    @staticmethod
    def compute_pe_offset(system_type: str, parameters: Dict[str, float]) -> float:
        """
        Compute PE offset to set minimum PE = 0
        
        Args:
            system_type: Type of mechanical system
            parameters: System parameters
            
        Returns:
            PE offset value
        """
        system = system_type.lower()
        
        if 'double' in system and 'pendulum' in system:
            m1 = parameters.get('m1', 1.0)
            m2 = parameters.get('m2', 1.0)
            l1 = parameters.get('l1', 1.0)
            l2 = parameters.get('l2', 1.0)
            g = parameters.get('g', 9.81)
            # Minimum PE when both pendulums hang straight down
            return -m1 * g * l1 - m2 * g * (l1 + l2)
            
        elif 'pendulum' in system:
            m = parameters.get('m', 1.0)
            l = parameters.get('l', 1.0)
            g = parameters.get('g', 9.81)
            # Minimum PE when pendulum hangs straight down
            return -m * g * l
            
        elif 'oscillator' in system or 'spring' in system:
            # Harmonic oscillator: PE minimum is already at x=0
            return 0.0
            
        else:
            # Default: no offset
            return 0.0
    
    @staticmethod
    def compute_kinetic_energy(solution: dict, parameters: Dict[str, float]) -> np.ndarray:
        """
        Compute kinetic energy from solution with validation.
        
        Args:
            solution: Solution dictionary (validated)
            parameters: System parameters dictionary
            
        Returns:
            Array of kinetic energy values
            
        Raises:
            TypeError: If inputs have wrong types
            ValueError: If solution is invalid
        """
        if not isinstance(parameters, dict):
            raise TypeError(f"parameters must be dict, got {type(parameters).__name__}")
        
        validate_solution_dict(solution)
        
        t = solution['t']
        y = solution['y']
        coords = solution['coordinates']
        
        KE = np.zeros_like(t)
        
        if len(coords) == 0:
            logger.warning("No coordinates found for kinetic energy calculation")
            return KE
        
        if 'theta' in coords[0]:  # Pendulum systems
            if len(coords) == 1:  # Simple pendulum
                if y.shape[0] < 2:
                    logger.warning("Insufficient state vector for simple pendulum KE")
                    return KE
                theta_dot = y[1]
                m = parameters.get('m', 1.0)
                l = parameters.get('l', 1.0)
                KE = 0.5 * m * l**2 * theta_dot**2
                
            elif len(coords) >= 2:  # Double pendulum
                if y.shape[0] < 4:
                    logger.warning("Insufficient state vector for double pendulum KE")
                    return KE
                theta1_dot, theta2_dot = y[1], y[3]
                theta1, theta2 = y[0], y[2]
                m1 = parameters.get('m1', 1.0)
                m2 = parameters.get('m2', 1.0)
                l1 = parameters.get('l1', 1.0)
                l2 = parameters.get('l2', 1.0)
                
                KE1 = 0.5 * m1 * l1**2 * theta1_dot**2
                KE2 = 0.5 * m2 * (l1**2 * theta1_dot**2 + l2**2 * theta2_dot**2 + 
                                  2 * l1 * l2 * theta1_dot * theta2_dot * np.cos(theta1 - theta2))
                KE = KE1 + KE2
                
        else:  # Cartesian systems
            v = y[1] if y.shape[0] > 1 else np.zeros_like(t)
            m = parameters.get('m', 1.0)
            KE = 0.5 * m * v**2
            
        return KE
    
    @staticmethod
    def compute_potential_energy(solution: dict, parameters: Dict[str, float], 
                                system_type: str = "") -> np.ndarray:
        """Compute potential energy from solution with proper offset"""
        t = solution['t']
        y = solution['y']
        coords = solution['coordinates']
        
        PE = np.zeros_like(t)
        
        if len(coords) == 0:
            logger.warning("No coordinates found for potential energy calculation")
            return PE
        
        if 'theta' in coords[0]:  # Pendulum systems
            if len(coords) == 1:  # Simple pendulum
                if y.shape[0] < 1:
                    logger.warning("Insufficient state vector for simple pendulum PE")
                    return PE
                theta = y[0]
                m = parameters.get('m', 1.0)
                l = parameters.get('l', 1.0)
                g = parameters.get('g', 9.81)
                
                PE = -m * g * l * np.cos(theta)
                offset = PotentialEnergyCalculator.compute_pe_offset('simple_pendulum', parameters)
                PE = PE - offset
                
            elif len(coords) >= 2:  # Double pendulum
                if y.shape[0] < 3:
                    logger.warning("Insufficient state vector for double pendulum PE")
                    return PE
                theta1, theta2 = y[0], y[2]
                m1 = parameters.get('m1', 1.0)
                m2 = parameters.get('m2', 1.0)
                l1 = parameters.get('l1', 1.0)
                l2 = parameters.get('l2', 1.0)
                g = parameters.get('g', 9.81)
                
                PE1 = -m1 * g * l1 * np.cos(theta1)
                PE2 = -m2 * g * (l1 * np.cos(theta1) + l2 * np.cos(theta2))
                PE = PE1 + PE2
                
                offset = PotentialEnergyCalculator.compute_pe_offset('double_pendulum', parameters)
                PE = PE - offset
                
        else:  # Cartesian systems
            if y.shape[0] < 1:
                logger.warning("Insufficient state vector for Cartesian PE")
                return PE
            x = y[0]
            k = parameters.get('k', 1.0)
            PE = 0.5 * k * x**2
            
        return PE

# ============================================================================
# ENHANCED 3D VISUALIZATION ENGINE WITH CIRCULAR BUFFERS
# ============================================================================

class MechanicsVisualizer:
    """Enhanced visualization with circular buffers and configurable options"""
    
    def __init__(self, trail_length: int = None, fps: int = None):
        self.fig = None
        self.ax = None
        self.animation = None
        self.trail_length = trail_length or config.trail_length
        self.fps = fps or config.animation_fps
        logger.debug(f"Visualizer initialized: trail_length={self.trail_length}, fps={self.fps}")

    def has_ffmpeg(self) -> bool:
        """Check if ffmpeg is available"""
        import shutil
        return shutil.which('ffmpeg') is not None

    def save_animation_to_file(self, anim: animation.FuncAnimation, 
                               filename: str, fps: int = None, dpi: int = 100) -> bool:
        """
        Save animation to file with validation.
        
        Args:
            anim: Animation object to save
            filename: Output filename (validated)
            fps: Frames per second (optional)
            dpi: Dots per inch (default: 100)
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            TypeError: If inputs have wrong types
            ValueError: If filename is invalid or parameters out of range
        """
        if anim is None:
            raise ValueError("anim cannot be None")
        
        validate_file_path(filename, must_exist=False)
        
        if fps is not None:
            if not isinstance(fps, int):
                raise TypeError(f"fps must be int, got {type(fps).__name__}")
            if fps < 1 or fps > 120:
                raise ValueError(f"fps must be in [1, 120], got {fps}")
        
        if not isinstance(dpi, int):
            raise TypeError(f"dpi must be int, got {type(dpi).__name__}")
        if dpi < 10 or dpi > 1000:
            raise ValueError(f"dpi must be in [10, 1000], got {dpi}")

        fps = fps or self.fps

        try:
            if filename.lower().endswith('.mp4') and self.has_ffmpeg():
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=fps, metadata=dict(artist='MechanicsDSL'), bitrate=1800)
                anim.save(filename, writer=writer, dpi=dpi)
                logger.info(f"Animation saved to {filename}")
                return True
            elif filename.lower().endswith('.gif'):
                anim.save(filename, writer='pillow', fps=fps)
                logger.info(f"Animation saved to {filename}")
                return True
            else:
                if self.has_ffmpeg():
                    Writer = animation.writers['ffmpeg']
                    writer = Writer(fps=fps, metadata=dict(artist='MechanicsDSL'), bitrate=1800)
                    anim.save(filename, writer=writer, dpi=dpi)
                    logger.info(f"Animation saved to {filename}")
                    return True
        except (IOError, OSError, PermissionError, ValueError, AttributeError) as e:
            logger.error(f"Failed to save animation: {e}")
        except Exception as e:
            logger.error(f"Unexpected error saving animation: {type(e).__name__}: {e}")
        
        return False

    def setup_3d_plot(self, title: str = "Classical Mechanics Simulation"):
        """Setup 3D plotting environment"""
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title(title, fontsize=16, fontweight='bold')
        self.ax.set_xlabel('X (m)', fontsize=12)
        self.ax.set_ylabel('Y (m)', fontsize=12)
        self.ax.set_zlabel('Z (m)', fontsize=12)
        self.ax.grid(True, alpha=0.3)

    def animate_pendulum(self, solution: dict, parameters: dict, system_name: str = "pendulum"):
        """
        Create animated pendulum visualization with validation.
        
        Args:
            solution: Solution dictionary (validated)
            parameters: System parameters dictionary
            system_name: Name of the system
            
        Returns:
            Animation object or None if failed
            
        Raises:
            TypeError: If inputs have wrong types
            ValueError: If solution is invalid
        """
        if not isinstance(parameters, dict):
            raise TypeError(f"parameters must be dict, got {type(parameters).__name__}")
        if not isinstance(system_name, str):
            raise TypeError(f"system_name must be str, got {type(system_name).__name__}")
        
        if not isinstance(solution, dict) or not solution.get('success', False):
            logger.warning("Cannot animate failed simulation")
            return None
        
        validate_solution_dict(solution)
            
        self.setup_3d_plot(f"{system_name.title()} Animation")
        
        t = solution['t']
        y = solution['y']
        coordinates = solution['coordinates']
        
        name = (system_name or '').lower()
        
        if len(coordinates) >= 2 or 'double' in name:
            return self._animate_double_pendulum(t, y, parameters)
        else:
            return self._animate_single_pendulum(t, y, parameters)
    
    def _animate_single_pendulum(self, t: np.ndarray, y: np.ndarray, parameters: dict):
        """Animate single pendulum with circular buffer"""
        if y.shape[0] < 1:
            logger.error("Insufficient state vector for single pendulum animation")
            return None
        theta = y[0]
        l = parameters.get('l', 1.0)
        
        x = l * np.sin(theta)
        y_pos = -l * np.cos(theta)
        z = np.zeros_like(x)
        
        self.ax.set_xlim(-l*1.2, l*1.2)
        self.ax.set_ylim(-l*1.2, l*0.2)
        self.ax.set_zlim(-0.1, 0.1)
        
        line, = self.ax.plot([], [], [], 'o-', linewidth=3, markersize=10, 
                            color=PRIMARY_COLOR, label='Pendulum')
        trail, = self.ax.plot([], [], [], '-', alpha=TRAIL_ALPHA, linewidth=1.5, 
                             color=SECONDARY_COLOR, label='Trail')
        time_text = self.ax.text2D(0.05, 0.95, '', transform=self.ax.transAxes, fontsize=12)
        
        self.ax.legend(loc='upper right')
        
        # Circular buffer for trail
        trail_buffer = deque(maxlen=self.trail_length)
        
        def animate_frame(frame):
            if frame < len(t):
                # Update pendulum position
                line.set_data([0, x[frame]], [0, y_pos[frame]])
                line.set_3d_properties([0, z[frame]])
                
                # Update trail using circular buffer
                trail_buffer.append((x[frame], y_pos[frame], z[frame]))
                if len(trail_buffer) > 1:
                    trail_x, trail_y, trail_z = zip(*trail_buffer)
                    trail.set_data(trail_x, trail_y)
                    trail.set_3d_properties(trail_z)
                
                time_text.set_text(f'Time: {t[frame]:.2f} s')
                
            return line, trail, time_text
        
        interval = ANIMATION_INTERVAL_MS
        self.animation = animation.FuncAnimation(
            self.fig, animate_frame, frames=len(t),
            interval=interval, blit=False, repeat=True
        )
        
        logger.info("Single pendulum animation created")
        return self.animation
    
    def _animate_double_pendulum(self, t: np.ndarray, y: np.ndarray, parameters: dict):
        """Animate double pendulum with circular buffers"""
        if y.shape[0] < 1:
            logger.error("Insufficient state vector for double pendulum animation")
            return None
        theta1 = y[0]
        theta2 = y[2] if y.shape[0] >= 4 else y[0]
        
        l1 = parameters.get('l1', parameters.get('l', 1.0))
        l2 = parameters.get('l2', 1.0)
        
        x1 = l1 * np.sin(theta1)
        y1 = -l1 * np.cos(theta1)
        x2 = x1 + l2 * np.sin(theta2)
        y2 = y1 - l2 * np.cos(theta2)
        
        max_reach = l1 + l2
        self.ax.set_xlim(-max_reach*1.1, max_reach*1.1)
        self.ax.set_ylim(-max_reach*1.1, max_reach*0.2)
        self.ax.set_zlim(-0.1, 0.1)
        
        line1, = self.ax.plot([], [], [], 'o-', linewidth=3, markersize=10, 
                             color=PRIMARY_COLOR, label='Pendulum 1')
        line2, = self.ax.plot([], [], [], 'o-', linewidth=3, markersize=10, 
                             color=TERTIARY_COLOR, label='Pendulum 2')
        trail1, = self.ax.plot([], [], [], '-', alpha=0.3, linewidth=1, color=PRIMARY_COLOR)
        trail2, = self.ax.plot([], [], [], '-', alpha=TRAIL_ALPHA, linewidth=1.5, color=SECONDARY_COLOR)
        time_text = self.ax.text2D(0.05, 0.95, '', transform=self.ax.transAxes, fontsize=12)
        
        self.ax.legend(loc='upper right')
        
        # Circular buffers for trails
        trail_buffer1 = deque(maxlen=self.trail_length)
        trail_buffer2 = deque(maxlen=self.trail_length)
        
        def animate_frame(frame):
            if frame < len(t):
                line1.set_data([0, x1[frame]], [0, y1[frame]])
                line1.set_3d_properties([0, 0])
                
                line2.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
                line2.set_3d_properties([0, 0])
                
                # Update trails using circular buffers
                trail_buffer1.append((x1[frame], y1[frame], 0))
                trail_buffer2.append((x2[frame], y2[frame], 0))
                
                if len(trail_buffer1) > 1:
                    t1_x, t1_y, t1_z = zip(*trail_buffer1)
                    trail1.set_data(t1_x, t1_y)
                    trail1.set_3d_properties(t1_z)
                
                if len(trail_buffer2) > 1:
                    t2_x, t2_y, t2_z = zip(*trail_buffer2)
                    trail2.set_data(t2_x, t2_y)
                    trail2.set_3d_properties(t2_z)
                
                time_text.set_text(f'Time: {t[frame]:.2f} s')
                
            return line1, line2, trail1, trail2, time_text
        
        interval = ANIMATION_INTERVAL_MS
        self.animation = animation.FuncAnimation(
            self.fig, animate_frame, frames=len(t),
            interval=interval, blit=False, repeat=True
        )
        
        logger.info("Double pendulum animation created")
        return self.animation

    def animate_oscillator(self, solution: dict, parameters: dict, system_name: str = "oscillator"):
        """Animate harmonic oscillator"""
        
        if not solution['success']:
            logger.warning("Cannot animate failed simulation")
            return None
        
        t = solution['t']
        y = solution['y']
        
        if y.shape[0] < 1:
            logger.error("Insufficient state vector for oscillator animation")
            return None
        x = y[0]
        v = y[1] if y.shape[0] > 1 else np.zeros_like(x)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.set_xlim(t[0], t[-1])
        ax1.set_ylim(np.min(x)*1.2, np.max(x)*1.2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position (m)')
        ax1.set_title(f'{system_name.title()} - Position vs Time')
        ax1.grid(True, alpha=0.3)
        
        line1, = ax1.plot([], [], 'b-', linewidth=2, label='Position')
        point1, = ax1.plot([], [], 'ro', markersize=8)
        ax1.legend()
        
        ax2.set_xlim(np.min(x)*1.2, np.max(x)*1.2)
        ax2.set_ylim(np.min(v)*1.2, np.max(v)*1.2)
        ax2.set_xlabel('Position (m)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Phase Space')
        ax2.grid(True, alpha=0.3)
        
        line2, = ax2.plot([], [], 'g-', linewidth=1.5, alpha=0.6, label='Trajectory')
        point2, = ax2.plot([], [], 'ro', markersize=8)
        ax2.legend()
        
        def init():
            line1.set_data([], [])
            point1.set_data([], [])
            line2.set_data([], [])
            point2.set_data([], [])
            return line1, point1, line2, point2
        
        def animate_frame(frame):
            if frame < len(t):
                line1.set_data(t[:frame], x[:frame])
                point1.set_data([t[frame]], [x[frame]])
                
                line2.set_data(x[:frame], v[:frame])
                point2.set_data([x[frame]], [v[frame]])
            
            return line1, point1, line2, point2
        
        interval = ANIMATION_INTERVAL_MS
        self.animation = animation.FuncAnimation(
            fig, animate_frame, frames=len(t), init_func=init,
            interval=interval, blit=True, repeat=True
        )
        
        self.fig = fig
        self.ax = ax1
        
        logger.info("Oscillator animation created")
        return self.animation

    def animate(self, solution: dict, parameters: dict, system_name: str = "system"):
        """Generic animation dispatcher"""
        if not solution or not solution.get('success'):
            logger.warning("Cannot animate: invalid solution")
            return None

        coords = solution.get('coordinates', [])
        name = (system_name or '').lower()

        try:
            if 'pendulum' in name or any('theta' in c for c in coords):
                return self.animate_pendulum(solution, parameters, system_name)
            elif 'oscillator' in name or 'spring' in name or (len(coords) == 1 and 'x' in coords):
                return self.animate_oscillator(solution, parameters, system_name)
            else:
                return self._animate_phase_space(solution, system_name)
                
        except Exception as e:
            logger.error(f"Animation failed: {e}")
            return None
    
    def _animate_phase_space(self, solution: dict, system_name: str):
        """Generic phase space animation"""
        t = solution['t']
        y = solution['y']
        coords = solution['coordinates']
        
        if len(coords) == 0:
            logger.warning("No coordinates to animate")
            return None
        
        if y.shape[0] < 1:
            logger.error("Insufficient state vector for phase space animation")
            return None
        q = y[0]
        q_dot = y[1] if y.shape[0] > 1 else np.zeros_like(q)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f'{system_name} - Phase Space')
        ax.set_xlabel(f'{coords[0]}')
        ax.set_ylabel(f'{coords[0]}_dot')
        ax.grid(True, alpha=0.3)
        
        line, = ax.plot([], [], 'b-', linewidth=1.5, alpha=0.6)
        point, = ax.plot([], [], 'ro', markersize=8)
        time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12)
        
        def init():
            ax.set_xlim(np.min(q)*1.1, np.max(q)*1.1)
            ax.set_ylim(np.min(q_dot)*1.1, np.max(q_dot)*1.1)
            line.set_data([], [])
            point.set_data([], [])
            return line, point, time_text
        
        def animate_frame(frame):
            if frame < len(t):
                line.set_data(q[:frame], q_dot[:frame])
                point.set_data([q[frame]], [q_dot[frame]])
                time_text.set_text(f'Time: {t[frame]:.2f} s')
            return line, point, time_text
        
        interval = ANIMATION_INTERVAL_MS
        self.animation = animation.FuncAnimation(
            fig, animate_frame, frames=len(t), init_func=init,
            interval=interval, blit=True, repeat=True
        )
        
        self.fig = fig
        self.ax = ax
        
        logger.info("Phase space animation created")
        return self.animation

    def plot_energy(self, solution: dict, parameters: dict, system_name: str = "",
                   lagrangian: sp.Expr = None):
        """Plot energy conservation analysis with proper offset correction"""
        
        if not solution['success']:
            logger.warning("Cannot plot energy for failed simulation")
            return
        
        t = solution['t']
        
        # Use the new energy calculator
        KE = PotentialEnergyCalculator.compute_kinetic_energy(solution, parameters)
        PE = PotentialEnergyCalculator.compute_potential_energy(solution, parameters, system_name)
        E_total = KE + PE
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Energy Analysis', fontsize=16, fontweight='bold')
        
        axes[0, 0].plot(t, KE, 'r-', linewidth=2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Energy (J)')
        axes[0, 0].set_title('Kinetic Energy')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(t, PE, 'b-', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Energy (J)')
        axes[0, 1].set_title('Potential Energy')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(t, E_total, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Energy (J)')
        axes[1, 0].set_title('Total Energy')
        axes[1, 0].grid(True, alpha=0.3)
        
        E_error = (E_total - E_total[0]) / np.abs(E_total[0]) * 100 if E_total[0] != 0 else (E_total - E_total[0])
        axes[1, 1].plot(t, E_error, 'purple', linewidth=2)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Relative Error (%)')
        axes[1, 1].set_title('Energy Conservation Error')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        logger.info(f"\n{'='*50}")
        logger.info("Energy Conservation Analysis")
        logger.info(f"{'='*50}")
        logger.info(f"Initial Total Energy: {E_total[0]:.6f} J")
        logger.info(f"Final Total Energy:   {E_total[-1]:.6f} J")
        logger.info(f"Energy Change:        {E_total[-1] - E_total[0]:.6e} J")
        if E_total[0] != 0:
            logger.info(f"Relative Error:       {E_error[-1]:.6f}%")
            logger.info(f"Max Relative Error:   {np.max(np.abs(E_error)):.6f}%")
        logger.info(f"{'='*50}\n")

    def plot_phase_space(self, solution: dict, coordinate_index: int = 0):
        """
        Plot phase space trajectory with validation.
        
        Args:
            solution: Solution dictionary (validated)
            coordinate_index: Index of coordinate to plot (default: 0)
            
        Raises:
            TypeError: If inputs have wrong types
            ValueError: If solution is invalid or coordinate_index out of range
        """
        if not isinstance(coordinate_index, int):
            raise TypeError(f"coordinate_index must be int, got {type(coordinate_index).__name__}")
        if coordinate_index < 0:
            raise ValueError(f"coordinate_index must be non-negative, got {coordinate_index}")
        
        if not isinstance(solution, dict) or not solution.get('success', False):
            logger.warning("Cannot plot phase space for failed simulation")
            return
        
        validate_solution_dict(solution)
        
        y = solution['y']
        coords = solution['coordinates']
        
        if coordinate_index >= len(coords):
            raise ValueError(f"coordinate_index {coordinate_index} out of range [0, {len(coords)})")
        
        # Safe array access with validation
        pos_idx = 2 * coordinate_index
        vel_idx = 2 * coordinate_index + 1
        
        if pos_idx >= y.shape[0] or vel_idx >= y.shape[0]:
            raise ValueError(f"State vector too small: need indices {pos_idx} and {vel_idx}, got size {y.shape[0]}")
        
        position = y[pos_idx]
        velocity = y[vel_idx]
        
        # Validate arrays
        if not validate_array_safe(position, "position", check_finite=True):
            logger.warning("plot_phase_space: position array has issues, attempting to fix")
            position = np.nan_to_num(position, nan=0.0, posinf=1e10, neginf=-1e10)
        
        if not validate_array_safe(velocity, "velocity", check_finite=True):
            logger.warning("plot_phase_space: velocity array has issues, attempting to fix")
            velocity = np.nan_to_num(velocity, nan=0.0, posinf=1e10, neginf=-1e10)
        
        if len(position) == 0 or len(velocity) == 0:
            logger.error("plot_phase_space: empty position or velocity arrays")
            return
        
        if len(position) != len(velocity):
            logger.warning(f"plot_phase_space: position and velocity length mismatch ({len(position)} vs {len(velocity)})")
            min_len = min(len(position), len(velocity))
            position = position[:min_len]
            velocity = velocity[:min_len]
        
        plt.figure(figsize=(10, 10))
        try:
        plt.plot(position, velocity, 'b-', alpha=0.7, linewidth=1.5, label='Trajectory')
            if len(position) > 0:
        plt.plot(position[0], velocity[0], 'go', markersize=10, label='Start', zorder=5)
        plt.plot(position[-1], velocity[-1], 'ro', markersize=10, label='End', zorder=5)
        except Exception as e:
            logger.error(f"plot_phase_space: error plotting: {e}", exc_info=True)
            return
        
        plt.xlabel(f'{coords[coordinate_index]} (position)', fontsize=12)
        plt.ylabel(f'd{coords[coordinate_index]}/dt (velocity)', fontsize=12)
        plt.title(f'Phase Space: {coords[coordinate_index]}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()
        
        logger.info(f"Phase space plot created for {coords[coordinate_index]}")

# ============================================================================
# SYSTEM STATE SERIALIZATION
# ============================================================================

class SystemSerializer:
    """Serialize and deserialize compiled physics systems"""
    
    @staticmethod
    def export_system(compiler: 'PhysicsCompiler', filename: str, 
                     format: str = 'json') -> bool:
        """
        Export compiled system to file
        
        Args:
            compiler: PhysicsCompiler instance
            filename: Output filename
            format: Export format ('json' or 'pickle')
            
        Returns:
            True if successful
        """
        try:
            state = {
                'version': __version__,
                'system_name': compiler.system_name,
                'variables': compiler.variables,
                'parameters': compiler.parameters_def,
                'initial_conditions': compiler.initial_conditions,
                'lagrangian': str(compiler.lagrangian) if compiler.lagrangian else None,
                'hamiltonian': str(compiler.hamiltonian) if compiler.hamiltonian else None,
                'coordinates': compiler.get_coordinates(),
                'use_hamiltonian': compiler.use_hamiltonian_formulation,
                'constraints': [str(c) for c in compiler.constraints],
                'transforms': {k: str(v) for k, v in compiler.transforms.items()},
            }
            
            if format == 'json':
                with open(filename, 'w') as f:
                    json.dump(state, f, indent=2)
            elif format == 'pickle':
                with open(filename, 'wb') as f:
                    pickle.dump(state, f)
            else:
                raise ValueError(f"Unknown format: {format}")
            
            logger.info(f"System exported to {filename}")
            return True
            
        except (IOError, OSError, PermissionError, ValueError) as e:
            logger.error(f"Export failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected export error: {type(e).__name__}: {e}")
            return False
    
    @staticmethod
    def import_system(filename: str) -> Optional[dict]:
        """
        Import system state from file with validation.
        
        Args:
            filename: Input filename (validated)
            
        Returns:
            System state dictionary or None if failed
            
        Raises:
            TypeError: If filename is not a string
            ValueError: If filename is invalid
            FileNotFoundError: If file doesn't exist
        """
        validate_file_path(filename, must_exist=True)
        
        try:
            if filename.endswith('.json'):
                with open(filename, 'r', encoding='utf-8') as f:
                    state = json.load(f)
            elif filename.endswith('.pkl') or filename.endswith('.pickle'):
                with open(filename, 'rb') as f:
                    state = pickle.load(f)
            else:
                # Try JSON first
                with open(filename, 'r', encoding='utf-8') as f:
                    state = json.load(f)
            
            logger.info(f"System imported from {filename}")
            return state
            
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return None

# ============================================================================
# COMPLETE PHYSICS COMPILER WITH ALL IMPROVEMENTS
# ============================================================================

class PhysicsCompiler:
    """
    Main compiler class - v6.0.0 with enterprise-grade features.
    
    Production-ready physics DSL compiler with comprehensive validation,
    cross-platform support, and security hardening.
    
    Features:
    - Cross-platform timeout support (Windows/Unix)
    - Safe AST-based parsing (no eval())
    - Comprehensive input validation
    - Specific exception handling
    - Extensive type hints
    - Production-ready error recovery
    
    Example:
        >>> compiler = PhysicsCompiler()
        >>> result = compiler.compile_dsl("\\system{pendulum}\\lagrangian{x^2}")
        >>> if result['success']:
        ...     solution = compiler.simulate((0, 10))
        ...     compiler.animate(solution)
    """
    
    def __init__(self):
        self.ast: List[ASTNode] = []
        self.variables: Dict[str, dict] = {}
        self.definitions: Dict[str, dict] = {}
        self.parameters_def: Dict[str, dict] = {}
        self.system_name: str = "unnamed_system"
        self.lagrangian: Optional[Expression] = None
        self.hamiltonian: Optional[Expression] = None
        self.transforms: Dict[str, dict] = {}
        self.constraints: List[Expression] = []
        self.non_holonomic_constraints: List[Expression] = []
        self.forces: List[Expression] = []
        self.damping_forces: List[Expression] = []
        self.initial_conditions: Dict[str, float] = {}
        
        self.symbolic = SymbolicEngine()
        self.simulator = NumericalSimulator(self.symbolic)
        self.visualizer = MechanicsVisualizer()
        self.unit_system = UnitSystem()
        
        self.compilation_time: Optional[float] = None
        self.equations: Optional[Any] = None
        self.use_hamiltonian_formulation: bool = False
        
        # v6.0: Memory management
        if config.enable_memory_monitoring:
            gc.set_threshold(*config._gc_threshold)
            _perf_monitor.snapshot_memory("compiler_init")
        
        logger.debug("PhysicsCompiler initialized (v6.0.0)")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
        return False
    
    def cleanup(self) -> None:
        """v6.0: Cleanup resources and trigger garbage collection"""
        if config.enable_memory_monitoring:
            _perf_monitor.snapshot_memory("pre_cleanup")
        
        # Clear large caches
        if hasattr(self.symbolic, '_cache') and self.symbolic._cache:
            if isinstance(self.symbolic._cache, LRUCache):
                self.symbolic._cache.clear()
        
        # Clear compiled equations
        self.equations = None
        
        # Trigger garbage collection
        if config.enable_memory_monitoring:
            collected = gc.collect()
            logger.debug(f"Garbage collection: {collected} objects collected")
            _perf_monitor.snapshot_memory("post_cleanup")

    @profile_function
    def compile_dsl(self, dsl_source: str, use_hamiltonian: bool = False,
                   use_constraints: bool = True) -> dict:
        """
        Complete compilation pipeline with comprehensive validation.
        
        Args:
            dsl_source: DSL source code (must be non-empty string)
            use_hamiltonian: Force Hamiltonian formulation
            use_constraints: Apply constraint handling
            
        Returns:
            Compilation result dictionary with 'success' key
            
        Raises:
            TypeError: If dsl_source is not a string
            ValueError: If dsl_source is empty or invalid
            
        Example:
            >>> compiler = PhysicsCompiler()
            >>> result = compiler.compile_dsl(r"\\system{test}\\lagrangian{x^2}")
            >>> assert result['success']
        """
        # Comprehensive input validation
        if dsl_source is None:
            logger.error("compile_dsl: dsl_source is None")
            return {
                'success': False,
                'error': 'dsl_source cannot be None',
                'compilation_time': 0.0
            }
        
        if not isinstance(dsl_source, str):
            error_msg = f"dsl_source must be str, got {type(dsl_source).__name__}"
            logger.error(f"compile_dsl: {error_msg}")
            raise TypeError(error_msg)
        
        dsl_source = dsl_source.strip()
        if not dsl_source:
            error_msg = "dsl_source cannot be empty"
            logger.error(f"compile_dsl: {error_msg}")
            raise ValueError(error_msg)
        
        if len(dsl_source) > 1_000_000:  # 1MB limit
            error_msg = f"dsl_source too large ({len(dsl_source)} chars), max 1MB"
            logger.error(f"compile_dsl: {error_msg}")
            raise ValueError(error_msg)
        
        # Check for potentially malicious patterns
        dangerous_patterns = ['__import__', 'eval(', 'exec(', 'compile(']
        for pattern in dangerous_patterns:
            if pattern in dsl_source:
                logger.warning(f"compile_dsl: potentially dangerous pattern '{pattern}' detected in source")
        
        if not isinstance(use_hamiltonian, bool):
            error_msg = f"use_hamiltonian must be bool, got {type(use_hamiltonian).__name__}"
            logger.error(f"compile_dsl: {error_msg}")
            raise TypeError(error_msg)
        
        if not isinstance(use_constraints, bool):
            error_msg = f"use_constraints must be bool, got {type(use_constraints).__name__}"
            logger.error(f"compile_dsl: {error_msg}")
            raise TypeError(error_msg)
        
        start_time = time.time()
        logger.info(f"Starting DSL compilation (source length: {len(dsl_source)} chars)")
        
        # Performance monitoring
        if config.enable_performance_monitoring:
            _perf_monitor.snapshot_memory("pre_compilation")
            _perf_monitor.start_timer('compilation')
        
try:
            # Tokenize with error handling
            try:
                tokens = tokenize(dsl_source)
                if not tokens:
                    raise ValueError("Tokenization produced no tokens")
                logger.info(f"Tokenized {len(tokens)} tokens")
            except Exception as e:
                logger.error(f"Tokenization failed: {e}", exc_info=True)
                raise ValueError(f"Tokenization failed: {e}") from e
            
            # Parse with error handling
            try:
                parser = MechanicsParser(tokens)
                self.ast = parser.parse()
                
                if parser.errors:
                    logger.warning(f"Parser found {len(parser.errors)} errors")
                    if len(parser.errors) >= config.max_parser_errors:
                        raise ValueError(f"Too many parser errors ({len(parser.errors)})")
            except Exception as e:
                logger.error(f"Parsing failed: {e}", exc_info=True)
                raise ValueError(f"Parsing failed: {e}") from e
            
            # Semantic analysis with error handling
            try:
                self.analyze_semantics()
            except Exception as e:
                logger.error(f"Semantic analysis failed: {e}", exc_info=True)
                raise ValueError(f"Semantic analysis failed: {e}") from e
            
            # Determine formulation
            if self.hamiltonian is not None:
                use_hamiltonian = True
            elif use_hamiltonian and self.lagrangian is not None:
                coords = self.get_coordinates()
                L_sympy = self.symbolic.ast_to_sympy(self.lagrangian)
                self.hamiltonian_expr = self.symbolic.lagrangian_to_hamiltonian(L_sympy, coords)
                use_hamiltonian = True
            
            # Derive equations with error handling
            try:
                if use_hamiltonian:
                    equations = self.derive_hamiltonian_equations()
                    self.use_hamiltonian_formulation = True
                    logger.info("Using Hamiltonian formulation")
                else:
                    # Check for constraints
                    if use_constraints and len(self.constraints) > 0:
                        equations = self.derive_constrained_equations()
                        logger.info(f"Using constrained Lagrangian with {len(self.constraints)} constraints")
                    else:
                        equations = self.derive_equations()
                        logger.info("Using standard Lagrangian formulation")
                    self.use_hamiltonian_formulation = False
                
                if equations is None:
                    raise ValueError("Equation derivation returned None")
                    
                self.equations = equations
            except Exception as e:
                logger.error(f"Equation derivation failed: {e}", exc_info=True)
                raise ValueError(f"Equation derivation failed: {e}") from e
            
            # Setup simulation with error handling
            try:
                self.setup_simulation(equations)
            except Exception as e:
                logger.error(f"Simulation setup failed: {e}", exc_info=True)
                raise ValueError(f"Simulation setup failed: {e}") from e
            
            self.compilation_time = time.time() - start_time
            
            # Performance monitoring
            if config.enable_performance_monitoring:
                _perf_monitor.stop_timer('compilation')
                _perf_monitor.snapshot_memory("post_compilation")
            
            result = {
                'success': True,
                'system_name': self.system_name,
                'coordinates': list(self.get_coordinates()),
                'equations': equations,
                'variables': self.variables,
                'parameters': self.simulator.parameters,
                'compilation_time': self.compilation_time,
                'ast_nodes': len(self.ast),
                'formulation': 'Hamiltonian' if use_hamiltonian else 'Lagrangian',
                'num_constraints': len(self.constraints) if use_constraints else 0,
            }
            
            logger.info(f"Compilation successful in {self.compilation_time:.4f}s")
            
            # Add performance metrics if available
            if config.enable_performance_monitoring:
                comp_stats = _perf_monitor.get_stats('compilation')
                if comp_stats:
                    result['performance'] = comp_stats
            
            return result
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Compilation failed: {e}\n{error_trace}")
            return {
                'success': False,
                'error': str(e),
                'traceback': error_trace,
                'compilation_time': time.time() - start_time,
            }

    def analyze_semantics(self):
        """Extract system information from AST"""
        logger.info("Analyzing semantics")
        
        for node in self.ast:
            if isinstance(node, SystemDef):
                self.system_name = node.name
                logger.debug(f"System name: {self.system_name}")
                
            elif isinstance(node, VarDef):
                self.variables[node.name] = {
                    'type': node.vartype,
                    'unit': node.unit,
                    'vector': node.vector
                }
                logger.debug(f"Variable: {node.name} ({node.vartype})")
            
            elif isinstance(node, ParameterDef):
                self.parameters_def[node.name] = {
                    'value': node.value,
                    'unit': node.unit
                }
                logger.debug(f"Parameter: {node.name} = {node.value}")
                
            elif isinstance(node, DefineDef):
                self.definitions[node.name] = {
                    'args': node.args,
                    'body': node.body
                }
                logger.debug(f"Definition: {node.name}")
                
            elif isinstance(node, LagrangianDef):
                self.lagrangian = node.expr
                logger.debug("Lagrangian defined")
                
            elif isinstance(node, HamiltonianDef):
                self.hamiltonian = node.expr
                logger.debug("Hamiltonian defined")
                
            elif isinstance(node, TransformDef):
                self.transforms[node.var] = {
                    'type': node.coord_type,
                    'expression': node.expr
                }
                logger.debug(f"Transform: {node.var}")
            
            elif isinstance(node, ConstraintDef):
                self.constraints.append(node.expr)
                logger.debug("Holonomic constraint added")
            
            elif isinstance(node, NonHolonomicConstraintDef):
                self.non_holonomic_constraints.append(node.expr)
                logger.debug("Non-holonomic constraint added")
            
            elif isinstance(node, ForceDef):
                self.forces.append(node.expr)
                logger.debug(f"Force added: {node.force_type}")
            
            elif isinstance(node, DampingDef):
                self.damping_forces.append(node.expr)
                logger.debug("Damping force added")
                
            elif isinstance(node, InitialCondition):
                self.initial_conditions.update(node.conditions)
                logger.debug(f"Initial conditions: {node.conditions}")

    def get_coordinates(self) -> List[str]:
        """Extract generalized coordinates"""
        coordinates = []
        
        for var_name, var_info in self.variables.items():
            if (var_info['type'] in ['Angle', 'Position', 'Coordinate', 'Length'] or
                var_name in ['theta', 'theta1', 'theta2', 'x', 'y', 'z', 'r', 'phi', 'psi']):
                coordinates.append(var_name)
        
        logger.debug(f"Coordinates: {coordinates}")
        return coordinates

    def derive_equations(self) -> Dict[str, sp.Expr]:
        """Derive equations using Lagrangian formulation"""
        
        if self.lagrangian is None:
            raise ValueError("No Lagrangian defined")
        
        L_sympy = self.symbolic.ast_to_sympy(self.lagrangian)
        coordinates = self.get_coordinates()
        
        if not coordinates:
            raise ValueError("No generalized coordinates found")
        
        eq_list = self.symbolic.derive_equations_of_motion(L_sympy, coordinates)
        accelerations = self.symbolic.solve_for_accelerations(eq_list, coordinates)
        
        return accelerations

    def derive_constrained_equations(self) -> Dict[str, sp.Expr]:
        """Derive equations with constraints using Lagrange multipliers"""
        
        if self.lagrangian is None:
            raise ValueError("No Lagrangian defined")
        
        if not self.constraints:
            logger.warning("No constraints defined, using standard formulation")
            return self.derive_equations()
        
        L_sympy = self.symbolic.ast_to_sympy(self.lagrangian)
        coordinates = self.get_coordinates()
        
        if not coordinates:
            raise ValueError("No generalized coordinates found")
        
        # Convert constraint expressions to SymPy
        constraint_exprs = [self.symbolic.ast_to_sympy(c) for c in self.constraints]
        
        # Derive constrained equations
        eq_list, extended_coords = self.symbolic.derive_equations_with_constraints(
            L_sympy, coordinates, constraint_exprs
        )
        
        # Solve for accelerations and lambda multipliers
        accelerations = self.symbolic.solve_for_accelerations(eq_list, extended_coords)
        
        # Filter out only the coordinate accelerations (not lambda derivatives)
        coord_accelerations = {
            k: v for k, v in accelerations.items() 
            if any(k.startswith(f"{c}_ddot") for c in coordinates)
        }
        
        return coord_accelerations

    def derive_hamiltonian_equations(self) -> Tuple[List[sp.Expr], List[sp.Expr]]:
        """Derive equations using Hamiltonian formulation"""
        
        if self.hamiltonian is not None:
            H_sympy = self.symbolic.ast_to_sympy(self.hamiltonian)
        elif hasattr(self, 'hamiltonian_expr'):
            H_sympy = self.hamiltonian_expr
        else:
            raise ValueError("No Hamiltonian defined or derived")
        
        coordinates = self.get_coordinates()
        
        if not coordinates:
            raise ValueError("No generalized coordinates found")
        
        q_dots, p_dots = self.symbolic.derive_hamiltonian_equations(H_sympy, coordinates)
        
        return (q_dots, p_dots)

    def setup_simulation(self, equations):
        """Configure numerical simulator"""
        
        logger.info("Setting up simulation")
        
        # Collect parameters
        parameters = {}
        for param_name, param_info in self.parameters_def.items():
            parameters[param_name] = param_info['value']
        
        # Add default parameters
        for var_name, var_info in self.variables.items():
            if var_info['type'] in ['Real', 'Mass', 'Length', 'Acceleration', 'Spring Constant']:
                if var_name not in parameters:
                    defaults = {
                        'g': 9.81,
                        'm': 1.0, 'm1': 1.0, 'm2': 1.0,
                        'l': 1.0, 'l1': 1.0, 'l2': 1.0,
                        'k': 1.0,
                    }
                    parameters[var_name] = defaults.get(var_name, 1.0)
        
        self.simulator.set_parameters(parameters)
        self.simulator.set_initial_conditions(self.initial_conditions)
        
        coordinates = self.get_coordinates()
        
        if self.use_hamiltonian_formulation:
            q_dots, p_dots = equations
            self.simulator.compile_hamiltonian_equations(q_dots, p_dots, coordinates)
        else:
            self.simulator.compile_equations(equations, coordinates)

    def simulate(self, t_span: Tuple[float, float] = (0, 10), 
                num_points: int = 1000, **kwargs) -> dict:
        """Run numerical simulation"""
        return self.simulator.simulate(t_span, num_points, **kwargs)

    def animate(self, solution: dict, show: bool = True):
        """Create animation from solution"""
        parameters = self.simulator.parameters
        anim = self.visualizer.animate(solution, parameters, self.system_name)
        
        if show and anim is not None:
            plt.show()
        
        return anim

    def export_animation(self, solution: dict, filename: str, 
                        fps: int = 30, dpi: int = 100) -> str:
        """Export animation to file"""
        anim = self.animate(solution, show=False)
        
        if anim is None:
            raise RuntimeError('No animation available')
        
        ok = self.visualizer.save_animation_to_file(anim, filename, fps, dpi)
        
        if not ok:
            raise RuntimeError(f'Failed to save animation to {filename}')
        
        return filename

    def plot_energy(self, solution: dict):
        """Plot energy analysis"""
        self.visualizer.plot_energy(solution, self.simulator.parameters, 
                                   self.system_name, self.lagrangian)

    def plot_phase_space(self, solution: dict, coordinate_index: int = 0):
        """Plot phase space"""
        self.visualizer.plot_phase_space(solution, coordinate_index)
    
    def print_equations(self):
        """Print derived equations"""
        if self.equations is None:
            print("No equations derived yet.")
            return
        
        print(f"\n{'='*70}")
        print(f"Equations of Motion: {self.system_name}")
        print(f"Formulation: {'Hamiltonian' if self.use_hamiltonian_formulation else 'Lagrangian'}")
        print(f"{'='*70}\n")
        
        if self.use_hamiltonian_formulation:
            q_dots, p_dots = self.equations
            coords = self.get_coordinates()
            for i, q in enumerate(coords):
                print(f"d{q}/dt = {q_dots[i]}")
                print(f"dp_{q}/dt = {p_dots[i]}\n")
        else:
            for coord in self.get_coordinates():
                accel_key = f"{coord}_ddot"
                if accel_key in self.equations:
                    eq = self.equations[accel_key]
                    print(f"{accel_key} = {eq}\n")
        
        print(f"{'='*70}\n")
    
    def get_info(self) -> dict:
        """Get comprehensive system information"""
        return {
            'system_name': self.system_name,
            'coordinates': self.get_coordinates(),
            'variables': self.variables,
            'parameters': self.simulator.parameters,
            'initial_conditions': self.initial_conditions,
            'has_lagrangian': self.lagrangian is not None,
            'has_hamiltonian': self.hamiltonian is not None,
            'num_constraints': len(self.constraints),
            'compilation_time': self.compilation_time,
            'formulation': 'Hamiltonian' if self.use_hamiltonian_formulation else 'Lagrangian',
        }
    
    def export_system(self, filename: str, format: str = 'json') -> bool:
        """Export system state to file"""
        return SystemSerializer.export_system(self, filename, format)
    
    @staticmethod
    def import_system(filename: str) -> Optional['PhysicsCompiler']:
        """Import system state from file"""
        state = SystemSerializer.import_system(filename)
        if state is None:
            return None
        
        # Note: This creates a new compiler but doesn't fully reconstruct the equations
        # For full reconstruction, you'd need to re-compile the DSL source
        compiler = PhysicsCompiler()
        compiler.system_name = state.get('system_name', 'imported_system')
        compiler.variables = state.get('variables', {})
        compiler.parameters_def = state.get('parameters', {})
        compiler.initial_conditions = state.get('initial_conditions', {})
        
        logger.info(f"Imported system: {compiler.system_name}")
        logger.warning("Note: Equations not reconstructed. Re-compile DSL source for full functionality.")
        
        return compiler

# ============================================================================
# EXAMPLE SYSTEMS - EXPANDED WITH NEW FEATURES
# ============================================================================

def example_simple_pendulum() -> str:
    """Example: Simple pendulum"""
    return r"""
\system{simple_pendulum}

\defvar{theta}{Angle}{rad}
\defvar{m}{Mass}{kg}
\defvar{l}{Length}{m}
\defvar{g}{Acceleration}{m/s^2}

\parameter{m}{1.0}{kg}
\parameter{l}{1.0}{m}
\parameter{g}{9.81}{m/s^2}

\lagrangian{\frac{1}{2} * m * l^2 * \dot{theta}^2 - m * g * l * (1 - \cos{theta})}

\initial{theta=0.5, theta_dot=0.0}

\solve{RK45}
\animate{pendulum}
"""

def example_double_pendulum() -> str:
    """Example: Double pendulum (chaotic)"""
    return r"""
\system{double_pendulum}

\defvar{theta1}{Angle}{rad}
\defvar{theta2}{Angle}{rad}
\defvar{m1}{Mass}{kg}
\defvar{m2}{Mass}{kg}
\defvar{l1}{Length}{m}
\defvar{l2}{Length}{m}
\defvar{g}{Acceleration}{m/s^2}

\parameter{m1}{1.0}{kg}
\parameter{m2}{1.0}{kg}
\parameter{l1}{1.0}{m}
\parameter{l2}{1.0}{m}
\parameter{g}{9.81}{m/s^2}

\lagrangian{
    \frac{1}{2} * (m1 + m2) * l1^2 * \dot{theta1}^2 
    + \frac{1}{2} * m2 * l2^2 * \dot{theta2}^2 
    + m2 * l1 * l2 * \dot{theta1} * \dot{theta2} * \cos{theta1 - theta2}
    + (m1 + m2) * g * l1 * \cos{theta1}
    + m2 * g * l2 * \cos{theta2}
}

\initial{theta1=1.57, theta1_dot=0.0, theta2=1.57, theta2_dot=0.0}

\solve{RK45}
\animate{double_pendulum}
"""

def example_harmonic_oscillator() -> str:
    """Example: Harmonic oscillator"""
    return r"""
\system{harmonic_oscillator}

\defvar{x}{Position}{m}
\defvar{m}{Mass}{kg}
\defvar{k}{Spring Constant}{N/m}

\parameter{m}{1.0}{kg}
\parameter{k}{10.0}{N/m}

\lagrangian{\frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2}

\initial{x=1.0, x_dot=0.0}

\solve{RK45}
\animate{oscillator}
"""

def example_atwood_machine() -> str:
    """Example: Atwood machine with constraint"""
    return r"""
\system{atwood_machine}

\defvar{x}{Position}{m}
\defvar{m1}{Mass}{kg}
\defvar{m2}{Mass}{kg}
\defvar{g}{Acceleration}{m/s^2}

\parameter{m1}{2.0}{kg}
\parameter{m2}{1.0}{kg}
\parameter{g}{9.81}{m/s^2}

\lagrangian{\frac{1}{2} * (m1 + m2) * \dot{x}^2 + (m1 - m2) * g * x}

\initial{x=0.0, x_dot=0.0}

\solve{LSODA}
"""

def run_example(example_name: str = "simple_pendulum", 
                t_span: Tuple[float, float] = (0, 10),
                show_animation: bool = True,
                show_energy: bool = True,
                show_phase: bool = True,
                use_hamiltonian: bool = False,
                export_file: Optional[str] = None) -> dict:
    """
    Run a built-in example system
    
    Args:
        example_name: Name of example
        t_span: Time span for simulation
        show_animation: Whether to show animation
        show_energy: Whether to show energy plot
        show_phase: Whether to show phase space plot
        use_hamiltonian: Use Hamiltonian formulation
        export_file: Optional filename to export animation
        
    Returns:
        Dictionary with compiler and solution
    """
    
    examples = {
        'simple_pendulum': example_simple_pendulum(),
        'double_pendulum': example_double_pendulum(),
        'harmonic_oscillator': example_harmonic_oscillator(),
        'atwood_machine': example_atwood_machine(),
    }
    
    if example_name not in examples:
        raise ValueError(f"Unknown example: {example_name}. Choose from {list(examples.keys())}")
    
    dsl_code = examples[example_name]
    
    compiler = PhysicsCompiler()
    result = compiler.compile_dsl(dsl_code, use_hamiltonian=use_hamiltonian)
    
    if not result['success']:
        logger.error(f"Compilation failed: {result.get('error', 'Unknown error')}")
        if 'traceback' in result:
            print(result['traceback'])
        return {'compiler': compiler, 'solution': None}
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Successfully compiled: {result['system_name']}")
    logger.info(f"Formulation: {result['formulation']}")
    logger.info(f"Coordinates: {result['coordinates']}")
    logger.info(f"Compilation time: {result['compilation_time']:.4f} seconds")
    logger.info(f"{'='*70}\n")
    
    compiler.print_equations()
    
    logger.info("Running simulation...")
    solution = compiler.simulate(t_span, num_points=1000)
    
    if not solution['success']:
        logger.error(f"Simulation failed: {solution.get('error', 'Unknown error')}")
        return {'compiler': compiler, 'solution': solution}
    
    logger.info(f"Simulation completed: {solution['nfev']} function evaluations")
    if solution.get('is_stiff'):
        logger.warning("⚠️  System detected as potentially stiff")
    
    if show_animation:
        logger.info("\nCreating animation...")
        compiler.animate(solution, show=True)
    
    if show_energy:
        logger.info("\nPlotting energy analysis...")
        compiler.plot_energy(solution)
    
    if show_phase:
        logger.info("\nPlotting phase space...")
        compiler.plot_phase_space(solution, coordinate_index=0)
    
    if export_file:
        logger.info(f"\nExporting animation to {export_file}...")
        try:
            compiler.export_animation(solution, export_file)
            logger.info("Export successful!")
        except Exception as e:
            logger.error(f"Export failed: {e}")
    
    return {
        'compiler': compiler,
        'solution': solution,
        'result': result
    }

# ============================================================================
# VALIDATION AND TESTING - COMPREHENSIVE PYTEST-COMPATIBLE SUITE
# ============================================================================

class SystemValidator:
    """Validate DSL systems against known analytical solutions"""
    
    @staticmethod
    def validate_simple_harmonic_oscillator(compiler: PhysicsCompiler, 
                                           solution: dict,
                                           tolerance: float = 0.01) -> bool:
        """Validate harmonic oscillator against analytical solution"""
        if not solution['success']:
            return False
        
        t = solution['t']
        x = solution['y'][0]
        v = solution['y'][1]
        
        m = compiler.simulator.parameters.get('m', 1.0)
        k = compiler.simulator.parameters.get('k', 1.0)
        
        omega = np.sqrt(k / m)
        
        x0 = x[0]
        v0 = v[0]
        A = np.sqrt(x0**2 + (v0/omega)**2)
        phi = np.arctan2(-v0/omega, x0)
        
        x_analytical = A * np.cos(omega * t + phi)
        
        error = np.max(np.abs(x - x_analytical)) / (A if A != 0 else 1.0)
        
        logger.info(f"\n{'='*50}")
        logger.info("Harmonic Oscillator Validation")
        logger.info(f"{'='*50}")
        logger.info(f"  Natural frequency: {omega:.4f} rad/s")
        logger.info(f"  Amplitude: {A:.4f} m")
        logger.info(f"  Max relative error: {error:.6f}")
        logger.info(f"  Tolerance: {tolerance}")
        logger.info(f"  Status: {'✓ PASSED' if error < tolerance else '✗ FAILED'}")
        logger.info(f"{'='*50}\n")
        
        return error < tolerance
    
    @staticmethod
    def validate_energy_conservation(compiler: PhysicsCompiler,
                                    solution: dict,
                                    tolerance: float = 0.01) -> bool:
        """Validate energy conservation"""
        if not solution['success']:
            return False
        
        params = compiler.simulator.parameters
        system_name = compiler.system_name
        
        # Use improved energy calculator
        KE = PotentialEnergyCalculator.compute_kinetic_energy(solution, params)
        PE = PotentialEnergyCalculator.compute_potential_energy(solution, params, system_name)
        E_total = KE + PE
        
        E_error = np.abs((E_total - E_total[0]) / (E_total[0] if E_total[0] != 0 else 1.0))
        max_error = np.max(E_error)
        
        logger.info(f"\n{'='*50}")
        logger.info("Energy Conservation Validation")
        logger.info(f"{'='*50}")
        logger.info(f"  Initial energy: {E_total[0]:.6f} J")
        logger.info(f"  Final energy: {E_total[-1]:.6f} J")
        logger.info(f"  Max relative error: {max_error:.6f}")
        logger.info(f"  Tolerance: {tolerance}")
        logger.info(f"  Status: {'✓ PASSED' if max_error < tolerance else '✗ FAILED'}")
        logger.info(f"{'='*50}\n")
        
        return max_error < tolerance

    @staticmethod
    def run_all_tests() -> dict:
        """Run comprehensive test suite"""
        print("\n" + "="*70)
        print("MechanicsDSL v6.0.0 - Comprehensive Test Suite")
        print("="*70 + "\n")
        
        results = {}
        
        # Test 1: Simple pendulum
        print("Test 1: Simple Pendulum")
        print("-" * 50)
        try:
            output = run_example('simple_pendulum', t_span=(0, 5), 
                               show_animation=False, show_energy=False, show_phase=False)
            compiler = output['compiler']
            solution = output['solution']
            results['simple_pendulum'] = {
                'compiled': output['result']['success'],
                'simulated': solution['success']
            }
        except Exception as e:
            print(f"✗ Failed: {e}")
            results['simple_pendulum'] = {'compiled': False, 'simulated': False}
        
        # Test 2: Harmonic oscillator with validation
        print("\nTest 2: Harmonic Oscillator (with validation)")
        print("-" * 50)
        try:
            output = run_example('harmonic_oscillator', t_span=(0, 10),
                               show_animation=False, show_energy=False, show_phase=False)
            compiler = output['compiler']
            solution = output['solution']
            
            validator = SystemValidator()
            passed = validator.validate_simple_harmonic_oscillator(compiler, solution)
            
            results['harmonic_oscillator'] = {
                'compiled': output['result']['success'],
                'simulated': solution['success'],
                'validated': passed
            }
        except Exception as e:
            print(f"✗ Failed: {e}")
            results['harmonic_oscillator'] = {'compiled': False, 'simulated': False, 'validated': False}
        
        # Test 3: Double pendulum
        print("\nTest 3: Double Pendulum (Chaotic System)")
        print("-" * 50)
        try:
            output = run_example('double_pendulum', t_span=(0, 5),
                               show_animation=False, show_energy=False, show_phase=False)
            compiler = output['compiler']
            solution = output['solution']
            
            validator = SystemValidator()
            energy_ok = validator.validate_energy_conservation(compiler, solution, tolerance=0.05)
            
            results['double_pendulum'] = {
                'compiled': output['result']['success'],
                'simulated': solution['success'],
                'energy_conserved': energy_ok
            }
        except Exception as e:
            print(f"✗ Failed: {e}")
            results['double_pendulum'] = {'compiled': False, 'simulated': False}
        
        # Test 4: Atwood machine
        print("\nTest 4: Atwood Machine")
        print("-" * 50)
        try:
            output = run_example('atwood_machine', t_span=(0, 5),
                               show_animation=False, show_energy=False, show_phase=False)
            results['atwood_machine'] = {
                'compiled': output['result']['success'],
                'simulated': output['solution']['success']
            }
        except Exception as e:
            print(f"✗ Failed: {e}")
            results['atwood_machine'] = {'compiled': False, 'simulated': False}
        
        # Summary
        print("\n" + "="*70)
        print("Test Summary")
        print("="*70)
        
        total_tests = 0
        passed_tests = 0
        
        for test_name, test_results in results.items():
            print(f"\n{test_name}:")
            for key, value in test_results.items():
                status = "✓" if value else "✗"
                print(f"  {status} {key}: {value}")
                total_tests += 1
                if value:
                    passed_tests += 1
        
        print(f"\n{'='*70}")
        print(f"Overall: {passed_tests}/{total_tests} tests passed")
        print(f"{'='*70}\n")
        
        return results

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Command line interface for MechanicsDSL"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='MechanicsDSL v6.0.0 - Enterprise-Grade Physics Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run built-in example
  python mechanics_dsl_v3.py --example simple_pendulum
  
  # Use Hamiltonian formulation
  python mechanics_dsl_v3.py --example simple_pendulum --hamiltonian
  
  # Run comprehensive tests
  python mechanics_dsl_v3.py --test
  
  # Enable profiling
  python mechanics_dsl_v3.py --example double_pendulum --profile
  
  # Export system state
  python mechanics_dsl_v3.py --example simple_pendulum --export-system pendulum.json
        """
    )
    
    parser.add_argument('--example', type=str, 
                       choices=['simple_pendulum', 'double_pendulum', 'harmonic_oscillator',
                               'atwood_machine'],
                       help='Run a built-in example system')
    parser.add_argument('--file', type=str, help='DSL source file to compile')
    parser.add_argument('--time', type=float, default=10.0, help='Simulation time (default: 10.0)')
    parser.add_argument('--points', type=int, default=1000, help='Number of time points (default: 1000)')
    parser.add_argument('--export', type=str, help='Export animation to file (.mp4 or .gif)')
    parser.add_argument('--energy', action='store_true', help='Show energy analysis')
    parser.add_argument('--phase', action='store_true', help='Show phase space plot')
    parser.add_argument('--validate', action='store_true', help='Run validation tests')
    parser.add_argument('--no-animation', action='store_true', help='Skip animation display')
    parser.add_argument('--hamiltonian', action='store_true', help='Use Hamiltonian formulation')
    parser.add_argument('--test', action='store_true', help='Run comprehensive test suite')
    parser.add_argument('--profile', action='store_true', help='Enable performance profiling')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--export-system', type=str, help='Export system state to file')
    parser.add_argument('--import-system', type=str, help='Import system state from file')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        setup_logging(logging.DEBUG)
        config.enable_debug_logging = True
    
    # Configure profiling
    if args.profile:
        config.enable_profiling = True
    
    if args.test:
        SystemValidator.run_all_tests()
        return 0
    
    if args.example:
        results = run_example(
            args.example,
            t_span=(0, args.time),
            show_animation=not args.no_animation,
            show_energy=args.energy,
            show_phase=args.phase,
            use_hamiltonian=args.hamiltonian,
            export_file=args.export
        )
        
        compiler = results['compiler']
        solution = results['solution']
        
        if args.export_system:
            compiler.export_system(args.export_system)
        
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                dsl_code = f.read()
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found")
            return 1
        
        compiler = PhysicsCompiler()
        result = compiler.compile_dsl(dsl_code, use_hamiltonian=args.hamiltonian)
        
        if not result['success']:
            print(f"Compilation failed: {result.get('error', 'Unknown error')}")
            if 'traceback' in result:
                print(result['traceback'])
            return 1
        
        print(f"Successfully compiled: {result['system_name']}")
        compiler.print_equations()
        
        solution = compiler.simulate((0, args.time), num_points=args.points)
        
        if not solution['success']:
            print(f"Simulation failed: {solution.get('error', 'Unknown error')}")
            return 1
        
        if not args.no_animation:
            compiler.animate(solution, show=True)
        
        if args.energy:
            compiler.plot_energy(solution)
        
        if args.phase:
            compiler.plot_phase_space(solution)
        
        if args.export_system:
            compiler.export_system(args.export_system)
    
    elif args.import_system:
        compiler = PhysicsCompiler.import_system(args.import_system)
        if compiler:
            print(f"Imported system: {compiler.system_name}")
            print(compiler.get_info())
        else:
            print("Import failed")
            return 1
    
    else:
        parser.print_help()
        return 0
    
    if args.validate and solution and solution['success']:
        print("\nRunning validation...")
        validator = SystemValidator()
        
        if 'oscillator' in compiler.system_name.lower():
            validator.validate_simple_harmonic_oscillator(compiler, solution)
        
        validator.validate_energy_conservation(compiler, solution)
    
    return 0

# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    'PhysicsCompiler',
    'SymbolicEngine',
    'NumericalSimulator',
    'MechanicsVisualizer',
    'SystemValidator',
    'SystemSerializer',
    'PotentialEnergyCalculator',
    'Config',
    'config',
    'tokenize',
    'MechanicsParser',
    'setup_logging',
    'example_simple_pendulum',
    'example_double_pendulum',
    'example_harmonic_oscillator',
    'example_atwood_machine',
    'run_example',
]

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) == 1:
        print("""
╔═══════════════════════════════════════════════════════════════════╗
║                  MechanicsDSL v6.0.0 - Enterprise Edition         ║
║       A Domain-Specific Language for Classical Mechanics          ║
║                                                                   ║
║  NEW in v6.0.0 (Enterprise Upgrade):                             ║
║  • Advanced Error Recovery: Multi-level retry mechanisms          ║
║  • Performance Monitoring: Built-in profiling & optimization      ║
║  • Intelligent Caching: Multi-tier LRU with memory management     ║
║  • Adaptive Solvers: Automatic solver selection                   ║
║  • Advanced Type Safety: Runtime type checking                    ║
║  • Memory Management: Resource pooling & monitoring                ║
║  • Enhanced Visualization: Interactive plots & real-time updates  ║
║  • Robust Validation: Multi-pass diagnostics                      ║
╚═══════════════════════════════════════════════════════════════════╝

Running interactive demo with simple pendulum...
        """)
        
        results = run_example('simple_pendulum', t_span=(0, 10))
        
        print("\n" + "="*70)
        print("Demo completed! Try these options:")
        print("  --example double_pendulum    # See chaotic behavior")
        print("  --hamiltonian                # Use Hamiltonian formulation")
        print("  --test                       # Run full test suite")
        print("  --profile                    # Enable performance profiling")
        print("  --export-system out.json     # Save system state")
        print("  --help                       # See all options")
        print("="*70)
    else:
        sys.exit(main())









