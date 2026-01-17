"""
Secure path validation utilities.

Prevents path traversal attacks (CWE-22) by validating file paths.
"""

import os
import re
from typing import List, Optional

# Characters not allowed in filenames
UNSAFE_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')

# Common path traversal patterns
TRAVERSAL_PATTERNS = [
    "..",
    "..\\",
    "../",
    "..%2f",
    "..%5c",
    "%2e%2e",
]


class PathValidationError(ValueError):
    """Raised when a path fails validation."""


def is_safe_filename(filename: str) -> bool:
    """
    Check if a filename is safe (no path components).

    Args:
        filename: The filename to validate

    Returns:
        True if the filename is safe
    """
    if not filename:
        return False

    # Check for path separators
    if os.sep in filename or "/" in filename or "\\" in filename:
        return False

    # Check for unsafe characters
    if UNSAFE_CHARS.search(filename):
        return False

    # Check for traversal patterns
    for pattern in TRAVERSAL_PATTERNS:
        if pattern in filename.lower():
            return False

    # Check for reserved names on Windows
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
    if filename.upper().split(".")[0] in reserved:
        return False

    return True


def secure_filename(filename: str) -> str:
    """
    Sanitize a filename by removing unsafe characters.

    Similar to werkzeug.utils.secure_filename.

    Args:
        filename: The original filename

    Returns:
        A safe version of the filename
    """
    # Remove path components
    filename = os.path.basename(filename)

    # Replace unsafe characters with underscores
    filename = UNSAFE_CHARS.sub("_", filename)

    # Remove leading/trailing dots and spaces
    filename = filename.strip(". ")

    # Replace multiple underscores with single
    filename = re.sub(r"_+", "_", filename)

    if not filename:
        return "unnamed"

    return filename


def validate_path_within_base(user_path: str, base_path: str, must_exist: bool = False) -> str:
    """
    Validate that a user-provided path stays within a base directory.

    This is the recommended approach for path traversal prevention.

    Args:
        user_path: The user-provided path (potentially malicious)
        base_path: The base directory paths must stay within
        must_exist: If True, verify the path exists

    Returns:
        The validated absolute path

    Raises:
        PathValidationError: If the path escapes the base directory
    """
    # Normalize the base path
    base = os.path.normpath(os.path.abspath(base_path))

    # Join and normalize the full path
    full_path = os.path.normpath(os.path.abspath(os.path.join(base_path, user_path)))

    # Check that the full path starts with base
    # Use os.path.commonpath for more robust comparison
    try:
        common = os.path.commonpath([base, full_path])
        if common != base:
            raise PathValidationError(f"Path '{user_path}' escapes base directory")
    except ValueError:
        # commonpath raises ValueError if paths are on different drives (Windows)
        raise PathValidationError(f"Path '{user_path}' is on a different drive")

    # Additional check: ensure normalized path starts with base
    if not full_path.startswith(base + os.sep) and full_path != base:
        raise PathValidationError(f"Path '{user_path}' escapes base directory")

    # Check existence if required
    if must_exist and not os.path.exists(full_path):
        raise PathValidationError(f"Path does not exist: {full_path}")

    return full_path


def safe_open(
    file_path: str,
    mode: str = "r",
    base_path: Optional[str] = None,
    allowed_extensions: Optional[List[str]] = None,
    **kwargs,
):
    """
    Safely open a file with path validation.

    Args:
        file_path: Path to the file
        mode: File open mode
        base_path: If provided, validate path is within this directory
        allowed_extensions: If provided, restrict to these extensions
        **kwargs: Additional arguments for open()

    Returns:
        File handle

    Raises:
        PathValidationError: If path validation fails
    """
    # Validate within base path if specified
    if base_path:
        file_path = validate_path_within_base(file_path, base_path)
    else:
        file_path = os.path.normpath(os.path.abspath(file_path))

    # Check extension
    if allowed_extensions:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in [e.lower() for e in allowed_extensions]:
            raise PathValidationError(
                f"File extension '{ext}' not allowed. " f"Allowed: {allowed_extensions}"
            )

    return open(file_path, mode, **kwargs)


def safe_path_join(base: str, *parts: str) -> str:
    """
    Safely join path parts, preventing traversal.

    Args:
        base: The base directory
        *parts: Additional path components

    Returns:
        The validated joined path

    Raises:
        PathValidationError: If result escapes base
    """
    # Join parts first
    user_path = os.path.join(*parts) if parts else ""

    return validate_path_within_base(user_path, base)


__all__ = [
    "PathValidationError",
    "is_safe_filename",
    "secure_filename",
    "validate_path_within_base",
    "safe_open",
    "safe_path_join",
]
