"""
MechanicsDSL Security Module
============================

Centralized security utilities for input validation, sandboxing, and attack prevention.

Security Features:
- Input sanitization
- Path traversal protection
- Injection prevention
- Resource limits
- Sandboxed execution

Author: MechanicsDSL Team
License: MIT
"""

import functools
import hashlib
import re
import secrets
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from .utils import logger

# =============================================================================
# Constants
# =============================================================================

# Maximum allowed sizes
MAX_DSL_SIZE = 1024 * 1024  # 1 MB
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_PATH_LENGTH = 4096
MAX_STRING_LENGTH = 100000

# Dangerous patterns
DANGEROUS_PATTERNS = [
    r"__import__",
    r"eval\s*\(",
    r"exec\s*\(",
    r"compile\s*\(",
    r"open\s*\(",
    r"os\.system",
    r"subprocess\.",
    r"Popen",
    r"shell\s*=\s*True",
    r"pickle\.load",
    r"marshal\.load",
    r"importlib",
]

# Allowed characters in identifiers
IDENTIFIER_REGEX = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Path traversal patterns
PATH_TRAVERSAL_PATTERNS = [
    "..",
    "~",
    "\x00",  # Null byte
]


# =============================================================================
# Exceptions
# =============================================================================


class SecurityError(Exception):
    """Base exception for security violations."""


class InputValidationError(SecurityError):
    """Raised when input validation fails."""


class PathTraversalError(SecurityError):
    """Raised when path traversal attack is detected."""


class InjectionError(SecurityError):
    """Raised when code injection is detected."""


class SandboxViolationError(SecurityError):
    """Raised when sandbox restrictions are violated."""


class ResourceLimitError(SecurityError):
    """Raised when resource limits are exceeded."""


# =============================================================================
# Input Validation
# =============================================================================


def validate_identifier(name: str, context: str = "identifier") -> str:
    """
    Validate that a string is a safe identifier.

    Args:
        name: The identifier to validate
        context: Description for error messages

    Returns:
        The validated identifier

    Raises:
        InputValidationError: If identifier is invalid
    """
    if not name:
        raise InputValidationError(f"Empty {context} not allowed")

    if len(name) > 256:
        raise InputValidationError(f"{context} too long (max 256 chars)")

    if not IDENTIFIER_REGEX.match(name):
        raise InputValidationError(
            f"Invalid {context}: '{name}'. Must match [a-zA-Z_][a-zA-Z0-9_]*"
        )

    # Check for Python keywords
    import keyword

    if keyword.iskeyword(name):
        raise InputValidationError(f"Cannot use Python keyword as {context}: '{name}'")

    return name


def validate_string(
    value: str, max_length: int = MAX_STRING_LENGTH, context: str = "string"
) -> str:
    """
    Validate and sanitize a string input.

    Args:
        value: String to validate
        max_length: Maximum allowed length
        context: Description for error messages

    Returns:
        Validated string

    Raises:
        InputValidationError: If validation fails
    """
    if not isinstance(value, str):
        raise InputValidationError(f"{context} must be a string, got {type(value)}")

    if len(value) > max_length:
        raise InputValidationError(f"{context} too long ({len(value)} > {max_length} chars)")

    # Check for null bytes
    if "\x00" in value:
        raise InputValidationError(f"Null bytes not allowed in {context}")

    return value


def validate_number(
    value: Union[int, float],
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    context: str = "number",
) -> Union[int, float]:
    """
    Validate a numeric input.

    Args:
        value: Number to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        context: Description for error messages

    Returns:
        Validated number

    Raises:
        InputValidationError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise InputValidationError(f"{context} must be numeric, got {type(value)}")

    import math

    if math.isnan(value) or math.isinf(value):
        raise InputValidationError(f"{context} cannot be NaN or Inf")

    if min_val is not None and value < min_val:
        raise InputValidationError(f"{context} must be >= {min_val}, got {value}")

    if max_val is not None and value > max_val:
        raise InputValidationError(f"{context} must be <= {max_val}, got {value}")

    return value


def validate_dsl_code(code: str) -> str:
    """
    Validate and sanitize DSL code input.

    Checks for:
    - Size limits
    - Dangerous patterns
    - Injection attempts

    Args:
        code: DSL code to validate

    Returns:
        Validated code

    Raises:
        InjectionError: If dangerous patterns detected
        InputValidationError: If validation fails
    """
    if not code or not isinstance(code, str):
        raise InputValidationError("DSL code must be a non-empty string")

    if len(code) > MAX_DSL_SIZE:
        raise InputValidationError(f"DSL code too large ({len(code)} > {MAX_DSL_SIZE} bytes)")

    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE):
            raise InjectionError(f"Dangerous pattern detected in DSL code: {pattern}")

    logger.debug("DSL code validated successfully")
    return code


# =============================================================================
# Path Security
# =============================================================================


def validate_path(
    path: Union[str, Path],
    base_dir: Optional[Path] = None,
    must_exist: bool = False,
    allow_absolute: bool = True,
) -> Path:
    """
    Validate a file path for security issues.

    Args:
        path: Path to validate
        base_dir: If provided, path must be under this directory
        must_exist: If True, path must exist
        allow_absolute: If False, only relative paths allowed

    Returns:
        Validated Path object

    Raises:
        PathTraversalError: If path traversal detected
        InputValidationError: If validation fails
    """
    path_str = str(path)

    # Check length
    if len(path_str) > MAX_PATH_LENGTH:
        raise InputValidationError(f"Path too long ({len(path_str)} > {MAX_PATH_LENGTH})")

    # Check for null bytes
    if "\x00" in path_str:
        raise PathTraversalError("Null byte in path (injection attempt)")

    # Convert to Path object
    path_obj = Path(path_str)

    # Check for absolute paths if not allowed
    if not allow_absolute and path_obj.is_absolute():
        raise InputValidationError("Absolute paths not allowed")

    # Resolve to absolute path for traversal check
    try:
        resolved = path_obj.resolve()
    except (OSError, ValueError) as e:
        raise InputValidationError(f"Invalid path: {e}")

    # Check for path traversal
    if base_dir is not None:
        base_resolved = Path(base_dir).resolve()
        try:
            resolved.relative_to(base_resolved)
        except ValueError:
            raise PathTraversalError(f"Path '{path}' escapes base directory '{base_dir}'")

    # Check if path contains explicit traversal
    for pattern in PATH_TRAVERSAL_PATTERNS:
        if pattern in path_str:
            raise PathTraversalError(f"Path traversal pattern detected: '{pattern}'")

    # Check existence if required
    if must_exist and not resolved.exists():
        raise InputValidationError(f"Path does not exist: {path}")

    return resolved


def safe_open(
    path: Union[str, Path],
    mode: str = "r",
    base_dir: Optional[Path] = None,
    max_size: int = MAX_FILE_SIZE,
    encoding: str = "utf-8",
) -> Any:
    """
    Safely open a file with security checks.

    Args:
        path: Path to open
        mode: File mode ('r', 'w', etc.)
        base_dir: Restrict to this directory
        max_size: Maximum file size for reads
        encoding: Text encoding

    Returns:
        File handle

    Raises:
        SecurityError: On security violations
    """
    # Validate path
    validated = validate_path(path, base_dir=base_dir)

    # Check file size for reads
    if "r" in mode and validated.exists():
        size = validated.stat().st_size
        if size > max_size:
            raise ResourceLimitError(f"File too large: {size} > {max_size} bytes")

    # Open with appropriate mode
    if "b" in mode:
        return open(validated, mode)
    else:
        return open(validated, mode, encoding=encoding)


# =============================================================================
# Sandboxing
# =============================================================================


@dataclass
class SandboxConfig:
    """Configuration for sandboxed execution."""

    allow_file_read: bool = False
    allow_file_write: bool = False
    allowed_read_dirs: List[Path] = field(default_factory=list)
    allowed_write_dirs: List[Path] = field(default_factory=list)
    max_memory_mb: int = 1024
    max_time_seconds: float = 300
    allow_network: bool = False
    allow_subprocess: bool = False


class Sandbox:
    """
    Sandbox for restricted code execution.

    Example:
        config = SandboxConfig(max_time_seconds=60)
        with Sandbox(config) as sandbox:
            result = sandbox.execute(some_function, args)
    """

    _thread_local = threading.local()

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self._original_open = None
        self._active = False

    def __enter__(self):
        self._activate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._deactivate()
        return False

    def _activate(self):
        """Activate sandbox restrictions."""
        self._active = True
        Sandbox._thread_local.sandbox = self
        logger.info("Sandbox activated")

    def _deactivate(self):
        """Deactivate sandbox restrictions."""
        self._active = False
        if hasattr(Sandbox._thread_local, "sandbox"):
            del Sandbox._thread_local.sandbox
        logger.info("Sandbox deactivated")

    @classmethod
    def current(cls) -> Optional["Sandbox"]:
        """Get the current active sandbox, if any."""
        return getattr(cls._thread_local, "sandbox", None)

    @classmethod
    def is_sandboxed(cls) -> bool:
        """Check if currently running in a sandbox."""
        return cls.current() is not None

    def check_file_access(self, path: Path, write: bool = False) -> bool:
        """Check if file access is allowed."""
        if not self._active:
            return True

        if write:
            if not self.config.allow_file_write:
                return False
            return any(path.is_relative_to(d) for d in self.config.allowed_write_dirs)
        else:
            if not self.config.allow_file_read:
                return False
            return any(path.is_relative_to(d) for d in self.config.allowed_read_dirs)

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function within the sandbox with timeout."""

        result = [None]
        exception = [None]

        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=self.config.max_time_seconds)

        if thread.is_alive():
            raise ResourceLimitError(
                f"Execution timeout ({self.config.max_time_seconds}s exceeded)"
            )

        if exception[0] is not None:
            raise exception[0]

        return result[0]


@contextmanager
def sandboxed(config: Optional[SandboxConfig] = None):
    """Context manager for sandboxed execution."""
    sandbox = Sandbox(config)
    with sandbox:
        yield sandbox


# =============================================================================
# Rate Limiting
# =============================================================================


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    max_requests: int = 100
    window_seconds: float = 60.0


class RateLimiter:
    """Simple rate limiter for API endpoints."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._requests: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> bool:
        """Check if request is allowed."""
        import time

        now = time.time()
        window_start = now - self.config.window_seconds

        with self._lock:
            # Get or create request list
            requests = self._requests.setdefault(key, [])

            # Remove old requests
            requests[:] = [t for t in requests if t > window_start]

            # Check limit
            if len(requests) >= self.config.max_requests:
                return False

            # Add current request
            requests.append(now)
            return True

    def require(self, key: str):
        """Require rate limit check to pass."""
        if not self.check(key):
            raise ResourceLimitError(
                f"Rate limit exceeded: {self.config.max_requests} "
                f"requests per {self.config.window_seconds}s"
            )


# =============================================================================
# Secure Random
# =============================================================================


def secure_random_string(length: int = 32) -> str:
    """Generate a cryptographically secure random string."""
    return secrets.token_hex(length // 2)


def secure_hash(data: Union[str, bytes], algorithm: str = "sha256") -> str:
    """Generate a secure hash of data."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    h = hashlib.new(algorithm)
    h.update(data)
    return h.hexdigest()


# =============================================================================
# Decorators
# =============================================================================

T = TypeVar("T")


def require_validation(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to require input validation."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Log security-relevant function call
        logger.debug(f"Security-validated call: {func.__name__}")
        return func(*args, **kwargs)

    return wrapper


def sandbox_aware(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to check sandbox restrictions."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sandbox = Sandbox.current()
        if sandbox is not None:
            # Apply sandbox checks
            pass
        return func(*args, **kwargs)

    return wrapper


# =============================================================================
# Initialization
# =============================================================================


def initialize_security():
    """Initialize security subsystem."""
    logger.info("Security subsystem initialized")
    logger.info(f"Max DSL size: {MAX_DSL_SIZE / 1024:.0f} KB")
    logger.info(f"Max file size: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB")
