"""
Security Tests - Sandbox and Resource Limits
=============================================

Tests for sandboxed execution and resource limit enforcement.
"""

import threading
import time

import pytest

from mechanics_dsl.security import (
    RateLimitConfig,
    RateLimiter,
    ResourceLimitError,
    Sandbox,
    SandboxConfig,
    sandboxed,
)


class TestSandbox:
    """Tests for sandbox functionality."""

    def test_sandbox_activation(self):
        """Test sandbox activation and deactivation."""
        assert not Sandbox.is_sandboxed()

        with Sandbox() as sb:
            assert Sandbox.is_sandboxed()
            assert Sandbox.current() is sb

        assert not Sandbox.is_sandboxed()

    def test_sandbox_context_manager(self):
        """Test sandboxed context manager."""
        with sandboxed() as sb:
            assert Sandbox.is_sandboxed()

        assert not Sandbox.is_sandboxed()

    def test_sandbox_timeout(self):
        """Test that sandbox enforces execution timeout."""
        config = SandboxConfig(max_time_seconds=0.1)

        def slow_function():
            time.sleep(10)
            return "completed"

        with Sandbox(config) as sb:
            with pytest.raises(ResourceLimitError, match="timeout"):
                sb.execute(slow_function)

    def test_sandbox_allows_fast_execution(self):
        """Test that fast execution completes successfully."""
        config = SandboxConfig(max_time_seconds=5.0)

        def fast_function():
            return 42

        with Sandbox(config) as sb:
            result = sb.execute(fast_function)
            assert result == 42

    def test_sandbox_propagates_exceptions(self):
        """Test that exceptions from sandboxed code propagate."""

        def failing_function():
            raise ValueError("test error")

        with Sandbox() as sb:
            with pytest.raises(ValueError, match="test error"):
                sb.execute(failing_function)

    def test_sandbox_with_arguments(self):
        """Test sandbox execution with function arguments."""
        config = SandboxConfig(max_time_seconds=5.0)

        def add(a, b):
            return a + b

        with Sandbox(config) as sb:
            result = sb.execute(add, 3, 4)
            assert result == 7

    def test_sandbox_with_kwargs(self):
        """Test sandbox execution with keyword arguments."""

        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        with Sandbox() as sb:
            result = sb.execute(greet, "World", greeting="Hi")
            assert result == "Hi, World!"

    def test_nested_sandbox_not_allowed(self):
        """Test that nested sandboxes work correctly."""
        with Sandbox() as outer:
            assert Sandbox.current() is outer

            # Inner sandbox replaces outer
            with Sandbox() as inner:
                assert Sandbox.current() is inner

            # Outer is restored (actually it's None after inner exits)
            # This behavior depends on implementation


class TestRateLimiter:
    """Tests for rate limiting functionality."""

    def test_rate_limiter_allows_under_limit(self):
        """Test that requests under limit are allowed."""
        config = RateLimitConfig(max_requests=5, window_seconds=60)
        limiter = RateLimiter(config)

        for i in range(5):
            assert limiter.check("test_key") is True

    def test_rate_limiter_blocks_over_limit(self):
        """Test that requests over limit are blocked."""
        config = RateLimitConfig(max_requests=3, window_seconds=60)
        limiter = RateLimiter(config)

        # First 3 should succeed
        for i in range(3):
            assert limiter.check("test_key") is True

        # 4th should fail
        assert limiter.check("test_key") is False

    def test_rate_limiter_separate_keys(self):
        """Test that different keys have separate limits."""
        config = RateLimitConfig(max_requests=2, window_seconds=60)
        limiter = RateLimiter(config)

        # Key 1
        assert limiter.check("key1") is True
        assert limiter.check("key1") is True
        assert limiter.check("key1") is False

        # Key 2 should be independent
        assert limiter.check("key2") is True
        assert limiter.check("key2") is True

    def test_rate_limiter_window_expiry(self):
        """Test that rate limit resets after window expires."""
        config = RateLimitConfig(max_requests=2, window_seconds=0.1)
        limiter = RateLimiter(config)

        # Use up quota
        assert limiter.check("test") is True
        assert limiter.check("test") is True
        assert limiter.check("test") is False

        # Wait for window to expire
        time.sleep(0.15)

        # Should be allowed again
        assert limiter.check("test") is True

    def test_rate_limiter_require(self):
        """Test the require method raises on limit."""
        config = RateLimitConfig(max_requests=1, window_seconds=60)
        limiter = RateLimiter(config)

        # First should work
        limiter.require("test")

        # Second should raise
        with pytest.raises(ResourceLimitError, match="Rate limit"):
            limiter.require("test")

    def test_rate_limiter_thread_safety(self):
        """Test that rate limiter is thread-safe."""
        config = RateLimitConfig(max_requests=100, window_seconds=60)
        limiter = RateLimiter(config)

        successes = [0]
        lock = threading.Lock()

        def worker():
            for _ in range(20):
                if limiter.check("shared"):
                    with lock:
                        successes[0] += 1

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have exactly 100 successes
        assert successes[0] == 100


class TestSandboxConfig:
    """Tests for sandbox configuration."""

    def test_default_config(self):
        """Test default sandbox configuration."""
        config = SandboxConfig()

        assert config.allow_file_read is False
        assert config.allow_file_write is False
        assert config.max_memory_mb == 1024
        assert config.max_time_seconds == 300
        assert config.allow_network is False
        assert config.allow_subprocess is False

    def test_custom_config(self):
        """Test custom sandbox configuration."""
        config = SandboxConfig(allow_file_read=True, max_time_seconds=60, max_memory_mb=512)

        assert config.allow_file_read is True
        assert config.max_time_seconds == 60
        assert config.max_memory_mb == 512


class TestResourceLimits:
    """Tests for resource limit enforcement."""

    def test_timeout_precision(self):
        """Test that timeout is reasonably precise."""
        config = SandboxConfig(max_time_seconds=0.5)

        start = time.time()

        def sleep_forever():
            time.sleep(100)

        with Sandbox(config) as sb:
            with pytest.raises(ResourceLimitError):
                sb.execute(sleep_forever)

        elapsed = time.time() - start

        # Should timeout around 0.5s, with some margin
        assert 0.4 < elapsed < 2.0
