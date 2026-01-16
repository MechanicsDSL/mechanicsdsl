"""
Test Helpers for Hardened Tests
================================

Common fixtures, utilities, and helper functions for testing.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
from pathlib import Path
from typing import Any, Dict, Generator, Optional
from contextlib import contextmanager
import functools


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_dsl_code() -> str:
    """Sample valid DSL code for testing."""
    return r"""
    \system{test_pendulum}
    
    \defvar{theta}{Angle}{rad}
    
    \parameter{m}{1.0}{kg}
    \parameter{l}{1.0}{m}
    \parameter{g}{9.81}{m/s^2}
    
    \lagrangian{0.5 * m * l^2 * \dot{theta}^2 - m * g * l * (1 - \cos{theta})}
    
    \initial{theta=0.5, theta_dot=0.0}
    """


@pytest.fixture
def malicious_dsl_codes() -> list:
    """Collection of malicious DSL codes for security testing."""
    return [
        "eval('__import__(\"os\").system(\"whoami\")')",
        "__import__('subprocess').call(['ls'])",
        "exec(open('/etc/passwd').read())",
        "os.system('rm -rf /')",
        "pickle.load(open('malicious.pkl', 'rb'))",
        "import subprocess; subprocess.Popen(['cat', '/etc/shadow'])",
    ]


@pytest.fixture
def valid_identifiers() -> list:
    """Valid Python identifiers."""
    return ['x', 'y1', 'theta', '_private', 'CamelCase', 'snake_case', 'x123']


@pytest.fixture
def invalid_identifiers() -> list:
    """Invalid Python identifiers."""
    return ['1x', 'my-var', 'my.var', 'import', 'class', 'def', '']


# =============================================================================
# Test Utilities
# =============================================================================

def assert_finite(value: Any, name: str = "value"):
    """Assert that a value is finite (not NaN or Inf)."""
    if isinstance(value, np.ndarray):
        assert np.all(np.isfinite(value)), f"{name} contains NaN or Inf"
    else:
        assert np.isfinite(value), f"{name} is NaN or Inf: {value}"


def assert_in_range(value: float, min_val: float, max_val: float, name: str = "value"):
    """Assert that a value is within a specified range."""
    assert min_val <= value <= max_val, \
        f"{name}={value} not in range [{min_val}, {max_val}]"


def assert_close(actual: float, expected: float, 
                 rtol: float = 1e-5, atol: float = 1e-8,
                 name: str = "value"):
    """Assert that two values are close within tolerance."""
    assert np.allclose(actual, expected, rtol=rtol, atol=atol), \
        f"{name}: {actual} not close to {expected} (rtol={rtol}, atol={atol})"


def assert_monotonic(sequence, increasing: bool = True, name: str = "sequence"):
    """Assert that a sequence is monotonically increasing or decreasing."""
    seq = list(sequence)
    if increasing:
        for i in range(1, len(seq)):
            assert seq[i] >= seq[i-1], f"{name} not monotonically increasing at index {i}"
    else:
        for i in range(1, len(seq)):
            assert seq[i] <= seq[i-1], f"{name} not monotonically decreasing at index {i}"


def assert_energy_conserved(energies: list, rtol: float = 0.01, name: str = "energy"):
    """Assert that energy is conserved (constant within tolerance)."""
    initial = energies[0]
    for i, e in enumerate(energies):
        rel_error = abs(e - initial) / abs(initial) if initial != 0 else abs(e)
        assert rel_error < rtol, \
            f"{name} not conserved: step {i}, error = {rel_error:.2%}"


# =============================================================================
# Context Managers
# =============================================================================

@contextmanager
def capture_logs():
    """Capture log output for testing."""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    
    logger = logging.getLogger('MechanicsDSL')
    original_handlers = logger.handlers[:]
    logger.handlers = [handler]
    
    try:
        yield log_capture
    finally:
        logger.handlers = original_handlers


@contextmanager
def timeout(seconds: float):
    """Context manager for timeout (Unix only)."""
    import signal
    
    def handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")
    
    if sys.platform != 'win32':
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(int(seconds))
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        yield  # No timeout on Windows


@contextmanager
def environment(**env_vars):
    """Temporarily set environment variables."""
    old_values = {}
    for key, value in env_vars.items():
        old_values[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


# =============================================================================
# Decorators
# =============================================================================

def skip_on_ci(reason: str = "Skipped on CI"):
    """Skip test when running on CI."""
    return pytest.mark.skipif(
        os.environ.get('CI') == 'true' or os.environ.get('GITHUB_ACTIONS') == 'true',
        reason=reason
    )


def require_gpu(reason: str = "GPU required"):
    """Skip test if GPU is not available."""
    try:
        import jax
        gpu_available = len(jax.devices('gpu')) > 0
    except:
        gpu_available = False
    
    return pytest.mark.skipif(not gpu_available, reason=reason)


def slow_test(func):
    """Mark a test as slow."""
    return pytest.mark.slow(func)


def repeat(count: int):
    """Repeat a test multiple times."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(count):
                func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# Data Generators
# =============================================================================

def random_state(dim: int, scale: float = 1.0) -> np.ndarray:
    """Generate a random state vector."""
    return np.random.randn(dim) * scale


def random_parameters(n: int = 5) -> Dict[str, float]:
    """Generate random parameters."""
    names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    return {names[i]: np.random.uniform(0.1, 10.0) for i in range(n)}


def random_initial_conditions(coords: list) -> Dict[str, float]:
    """Generate random initial conditions."""
    ics = {}
    for coord in coords:
        ics[coord] = np.random.uniform(-1.0, 1.0)
        ics[f"{coord}_dot"] = np.random.uniform(-1.0, 1.0)
    return ics
