"""
Structured Logging Module
=========================

Provides structured, context-rich logging for MechanicsDSL.

Features:
- Structured JSON logging for production
- Colored console logging for development
- Correlation IDs for request tracing
- Performance metrics logging
- Security event logging
"""

import json
import logging
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Union

# =============================================================================
# Log Record Extensions
# =============================================================================


@dataclass
class LogContext:
    """Extended context for log records."""

    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    system_name: Optional[str] = None
    operation: Optional[str] = None
    duration_ms: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None


_thread_local = threading.local()


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return getattr(_thread_local, "correlation_id", None)


def set_correlation_id(correlation_id: str):
    """Set the correlation ID for the current thread."""
    _thread_local.correlation_id = correlation_id


def clear_correlation_id():
    """Clear the correlation ID."""
    if hasattr(_thread_local, "correlation_id"):
        del _thread_local.correlation_id


@contextmanager
def correlation_context(correlation_id: Optional[str] = None):
    """Context manager for correlation ID."""
    cid = correlation_id or str(uuid.uuid4())
    old_cid = get_correlation_id()
    set_correlation_id(cid)
    try:
        yield cid
    finally:
        if old_cid:
            set_correlation_id(old_cid)
        else:
            clear_correlation_id()


# =============================================================================
# Formatters
# =============================================================================


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation ID if present
        correlation_id = get_correlation_id()
        if correlation_id:
            log_data["correlation_id"] = correlation_id

        # Add extra fields
        for key in [
            "duration_ms",
            "operation",
            "system_name",
            "user_id",
            "session_id",
            "error",
            "stack_trace",
            "metrics",
        ]:
            if hasattr(record, key):
                log_data[key] = getattr(record, key)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(log_data, default=str)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        reset = self.COLORS["RESET"]

        # Format timestamp
        timestamp = self.formatTime(record, "%H:%M:%S")

        # Format level
        level = f"{color}{record.levelname:8}{reset}"

        # Format message
        message = record.getMessage()

        # Basic format
        output = f"{timestamp} [{level}] {record.name}: {message}"

        # Add correlation ID if present
        correlation_id = get_correlation_id()
        if correlation_id:
            output = f"{timestamp} [{level}] [{correlation_id[:8]}] {record.name}: {message}"

        # Add duration if present
        if hasattr(record, "duration_ms"):
            output += f" ({record.duration_ms:.2f}ms)"

        # Add exception if present
        if record.exc_info:
            output += f"\n{self.formatException(record.exc_info)}"

        return output


# =============================================================================
# Handlers
# =============================================================================


class SecurityEventHandler(logging.Handler):
    """Handler for security-related events."""

    def __init__(self, log_file: Optional[str] = None):
        super().__init__(level=logging.WARNING)
        self.log_file = log_file
        self.events = []

    def emit(self, record: logging.LogRecord):
        if hasattr(record, "security_event"):
            event = {
                "timestamp": self.format(record),
                "event_type": getattr(record, "security_event", "unknown"),
                "message": record.getMessage(),
                "correlation_id": get_correlation_id(),
            }
            self.events.append(event)

            if self.log_file:
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(event) + "\n")


class MetricsHandler(logging.Handler):
    """Handler for collecting metrics from logs."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.metrics = {
            "request_count": 0,
            "error_count": 0,
            "total_duration_ms": 0.0,
            "operations": {},
        }
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord):
        with self._lock:
            self.metrics["request_count"] += 1

            if record.levelno >= logging.ERROR:
                self.metrics["error_count"] += 1

            if hasattr(record, "duration_ms"):
                self.metrics["total_duration_ms"] += record.duration_ms

            if hasattr(record, "operation"):
                op = record.operation
                if op not in self.metrics["operations"]:
                    self.metrics["operations"][op] = {"count": 0, "total_time": 0.0}
                self.metrics["operations"][op]["count"] += 1
                if hasattr(record, "duration_ms"):
                    self.metrics["operations"][op]["total_time"] += record.duration_ms

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.metrics)


# =============================================================================
# Logger Configuration
# =============================================================================


def configure_logging(
    level: Union[str, int] = logging.INFO,
    structured: bool = False,
    log_file: Optional[str] = None,
    security_log: Optional[str] = None,
    enable_metrics: bool = False,
) -> logging.Logger:
    """
    Configure the MechanicsDSL logger.

    Args:
        level: Logging level
        structured: Use JSON structured logging
        log_file: Path to log file
        security_log: Path to security event log
        enable_metrics: Enable metrics collection

    Returns:
        Configured logger
    """
    logger = logging.getLogger("MechanicsDSL")
    logger.setLevel(level if isinstance(level, int) else getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)

    if structured:
        console.setFormatter(StructuredFormatter())
    else:
        console.setFormatter(ColoredFormatter())

    logger.addHandler(console)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter())
        logger.addHandler(file_handler)

    # Security event handler
    if security_log:
        security_handler = SecurityEventHandler(security_log)
        logger.addHandler(security_handler)

    # Metrics handler
    if enable_metrics:
        metrics_handler = MetricsHandler()
        logger.addHandler(metrics_handler)
        logger.metrics_handler = metrics_handler

    return logger


# =============================================================================
# Decorators
# =============================================================================


def log_operation(name: Optional[str] = None, level: int = logging.DEBUG):
    """Decorator to log function entry, exit, and duration."""

    def decorator(func):
        operation_name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger("MechanicsDSL")

            logger.debug(f"Starting {operation_name}", extra={"operation": operation_name})

            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000

                logger.log(
                    level,
                    f"Completed {operation_name}",
                    extra={"operation": operation_name, "duration_ms": duration_ms},
                )

                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000

                logger.error(
                    f"Failed {operation_name}: {e}",
                    extra={"operation": operation_name, "duration_ms": duration_ms},
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator


def log_security_event(event_type: str):
    """Decorator to log security events."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger("MechanicsDSL")

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                logger.warning(
                    f"Security event: {event_type} - {e}", extra={"security_event": event_type}
                )
                raise

        return wrapper

    return decorator


# =============================================================================
# Convenience Functions
# =============================================================================


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a child logger."""
    base = logging.getLogger("MechanicsDSL")
    if name:
        return base.getChild(name)
    return base


def log_metrics() -> Dict[str, Any]:
    """Get collected metrics."""
    logger = logging.getLogger("MechanicsDSL")
    if hasattr(logger, "metrics_handler"):
        return logger.metrics_handler.get_metrics()
    return {}
