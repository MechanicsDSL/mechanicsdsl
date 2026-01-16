"""
Enhanced Error Handling Module
==============================

Provides comprehensive error handling with:
- Rich error context
- Error chains
- Recovery strategies
- Error reporting
"""

import sys
import traceback
import logging
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
import json

from .utils import logger


# =============================================================================
# Error Context
# =============================================================================

@dataclass
class ErrorContext:
    """Rich context for error reporting."""
    
    timestamp: datetime = field(default_factory=datetime.now)
    operation: Optional[str] = None
    component: Optional[str] = None
    user_message: Optional[str] = None
    technical_message: Optional[str] = None
    recovery_hint: Optional[str] = None
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'operation': self.operation,
            'component': self.component,
            'user_message': self.user_message,
            'technical_message': self.technical_message,
            'recovery_hint': self.recovery_hint,
            'error_code': self.error_code,
            'metadata': self.metadata,
        }


# =============================================================================
# Exception Hierarchy
# =============================================================================

class MechanicsDSLError(Exception):
    """Base exception for all MechanicsDSL errors."""
    
    error_code = "MDSL-0000"
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None,
                 cause: Optional[Exception] = None):
        super().__init__(message)
        self.context = context or ErrorContext()
        self.context.technical_message = message
        self.cause = cause
        self.__cause__ = cause
    
    def with_recovery_hint(self, hint: str) -> 'MechanicsDSLError':
        """Add recovery hint."""
        self.context.recovery_hint = hint
        return self
    
    def with_user_message(self, message: str) -> 'MechanicsDSLError':
        """Add user-friendly message."""
        self.context.user_message = message
        return self
    
    def with_metadata(self, **kwargs) -> 'MechanicsDSLError':
        """Add metadata."""
        self.context.metadata.update(kwargs)
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'error_type': type(self).__name__,
            'error_code': self.error_code,
            'message': str(self),
            'context': self.context.to_dict(),
            'cause': str(self.cause) if self.cause else None,
        }
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.context.recovery_hint:
            msg += f" (Hint: {self.context.recovery_hint})"
        return msg


class ParseError(MechanicsDSLError):
    """Error during DSL parsing."""
    error_code = "MDSL-1001"


class CompilationError(MechanicsDSLError):
    """Error during compilation."""
    error_code = "MDSL-1002"


class SimulationError(MechanicsDSLError):
    """Error during simulation."""
    error_code = "MDSL-2001"


class NumericalError(MechanicsDSLError):
    """Numerical computation error."""
    error_code = "MDSL-2002"


class CodegenError(MechanicsDSLError):
    """Error during code generation."""
    error_code = "MDSL-3001"


class ConfigurationError(MechanicsDSLError):
    """Configuration error."""
    error_code = "MDSL-4001"


class SecurityError(MechanicsDSLError):
    """Security-related error."""
    error_code = "MDSL-5001"


class ResourceError(MechanicsDSLError):
    """Resource limit or availability error."""
    error_code = "MDSL-6001"


# =============================================================================
# Error Handlers
# =============================================================================

T = TypeVar('T')


def handle_errors(
    error_map: Optional[Dict[Type[Exception], Type[MechanicsDSLError]]] = None,
    default_error: Type[MechanicsDSLError] = MechanicsDSLError,
    log_level: int = logging.ERROR,
    reraise: bool = True
):
    """
    Decorator for consistent error handling.
    
    Args:
        error_map: Map of exception types to MechanicsDSL error types
        default_error: Default error type for unmapped exceptions
        log_level: Logging level for errors
        reraise: Whether to reraise the error
    """
    error_map = error_map or {}
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except MechanicsDSLError:
                # Already our error type, just log and reraise
                logger.log(log_level, f"Error in {func.__name__}", exc_info=True)
                if reraise:
                    raise
            except Exception as e:
                # Convert to appropriate error type
                for exc_type, mdsl_type in error_map.items():
                    if isinstance(e, exc_type):
                        new_error = mdsl_type(str(e), cause=e)
                        break
                else:
                    new_error = default_error(str(e), cause=e)
                
                new_error.context.operation = func.__name__
                logger.log(log_level, f"Error in {func.__name__}: {e}", exc_info=True)
                
                if reraise:
                    raise new_error from e
                return None
        
        return wrapper
    return decorator


@contextmanager
def error_boundary(operation: str, component: str = "unknown"):
    """Context manager for error boundaries."""
    context = ErrorContext(operation=operation, component=component)
    
    try:
        yield context
    except MechanicsDSLError as e:
        e.context.operation = operation
        e.context.component = component
        raise
    except Exception as e:
        error = MechanicsDSLError(str(e), context=context, cause=e)
        logger.error(f"Error in {operation}: {e}", exc_info=True)
        raise error from e


# =============================================================================
# Recovery Strategies
# =============================================================================

def retry(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for retry with exponential backoff.
    
    Args:
        max_attempts: Maximum retry attempts
        delay_seconds: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Exceptions to catch and retry
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            import time
            
            last_exception = None
            delay = delay_seconds
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                            f"Retrying in {delay:.1f}s"
                        )
                        time.sleep(delay)
                        delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


def fallback(default_value: T = None, log_error: bool = True):
    """
    Decorator to return default value on error.
    
    Args:
        default_value: Value to return on error
        log_error: Whether to log the error
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.warning(f"{func.__name__} failed, using fallback: {e}")
                return default_value
        
        return wrapper
    return decorator


# =============================================================================
# Error Reporting
# =============================================================================

class ErrorReporter:
    """Collects and reports errors."""
    
    def __init__(self, max_errors: int = 1000):
        self.errors: List[Dict[str, Any]] = []
        self.max_errors = max_errors
    
    def report(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Report an error."""
        error_data = {
            'timestamp': datetime.now().isoformat(),
            'type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {},
        }
        
        if isinstance(error, MechanicsDSLError):
            error_data['error_code'] = error.error_code
            error_data['details'] = error.to_dict()
        
        self.errors.append(error_data)
        
        # Trim if too many
        if len(self.errors) > self.max_errors:
            self.errors = self.errors[-self.max_errors:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get error summary."""
        if not self.errors:
            return {'total': 0, 'by_type': {}}
        
        by_type = {}
        for error in self.errors:
            t = error['type']
            by_type[t] = by_type.get(t, 0) + 1
        
        return {
            'total': len(self.errors),
            'by_type': by_type,
            'recent': self.errors[-10:],
        }
    
    def clear(self):
        """Clear reported errors."""
        self.errors.clear()
    
    def export_json(self, filepath: str):
        """Export errors to JSON file."""
        with open(filepath, 'w') as f:
            json.dump({
                'summary': self.get_summary(),
                'errors': self.errors,
            }, f, indent=2, default=str)


# Global error reporter
error_reporter = ErrorReporter()


def report_error(error: Exception, **context):
    """Global function to report errors."""
    error_reporter.report(error, context)
