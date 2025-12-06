"""
Validation utilities for MechanicsDSL
"""
import numpy as np
import time
from typing import Any, Optional, Union, Tuple, Callable
from pathlib import Path
from contextlib import contextmanager, ExitStack
from functools import wraps

from .logging import logger


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
