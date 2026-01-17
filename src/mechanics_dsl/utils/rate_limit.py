"""
Rate limiting utilities for MechanicsDSL web services.

Provides thread-safe rate limiting for API endpoints and simulation requests.
Can be used with FastAPI, Flask, or any async web framework.

Example with FastAPI:
    >>> from fastapi import FastAPI, Request, HTTPException
    >>> from mechanics_dsl.utils.rate_limit import RateLimiter, RateLimitExceeded
    >>>
    >>> app = FastAPI()
    >>> limiter = RateLimiter(requests_per_minute=60, burst_limit=10)
    >>>
    >>> @app.middleware("http")
    >>> async def rate_limit_middleware(request: Request, call_next):
    ...     client_ip = request.client.host
    ...     if not limiter.allow(client_ip):
    ...         raise HTTPException(status_code=429, detail="Too many requests")
    ...     return await call_next(request)
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(self, key: str, retry_after: float):
        self.key = key
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded for {key}. Retry after {retry_after:.1f}s")


@dataclass
class TokenBucket:
    """Token bucket algorithm implementation for rate limiting."""

    capacity: float
    refill_rate: float  # tokens per second
    tokens: float = field(default=0.0)
    last_refill: float = field(default_factory=time.time)

    def __post_init__(self):
        self.tokens = self.capacity

    def consume(self, tokens: float = 1.0) -> bool:
        """
        Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self):
        """Add tokens based on time elapsed since last refill."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def time_until_available(self, tokens: float = 1.0) -> float:
        """
        Calculate time until the requested tokens will be available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Seconds until tokens are available (0 if already available)
        """
        self._refill()
        if self.tokens >= tokens:
            return 0.0
        needed = tokens - self.tokens
        return needed / self.refill_rate


class RateLimiter:
    """
    Thread-safe rate limiter using token bucket algorithm.

    Supports per-client rate limiting with configurable limits.

    Args:
        requests_per_minute: Maximum sustained requests per minute
        burst_limit: Maximum burst size (peak requests)
        cleanup_interval: Seconds between cleanup of stale buckets

    Example:
        >>> limiter = RateLimiter(requests_per_minute=60, burst_limit=10)
        >>> if limiter.allow("user_123"):
        ...     # Process request
        ...     pass
        >>> else:
        ...     # Return 429 Too Many Requests
        ...     pass
    """

    def __init__(
        self,
        requests_per_minute: float = 60.0,
        burst_limit: float = 10.0,
        cleanup_interval: float = 300.0,
    ):
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.cleanup_interval = cleanup_interval

        # Convert to tokens per second
        self._refill_rate = requests_per_minute / 60.0

        # Per-client buckets
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

    def allow(self, key: str, cost: float = 1.0) -> bool:
        """
        Check if a request is allowed under the rate limit.

        Args:
            key: Identifier for the client (e.g., IP address, user ID)
            cost: Cost of this request in tokens (default 1.0)

        Returns:
            True if request is allowed, False if rate limited
        """
        with self._lock:
            self._maybe_cleanup()
            bucket = self._get_or_create_bucket(key)
            return bucket.consume(cost)

    def allow_or_raise(self, key: str, cost: float = 1.0) -> None:
        """
        Check if request is allowed, raise RateLimitExceeded if not.

        Args:
            key: Identifier for the client
            cost: Cost of this request in tokens

        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        with self._lock:
            self._maybe_cleanup()
            bucket = self._get_or_create_bucket(key)
            if not bucket.consume(cost):
                retry_after = bucket.time_until_available(cost)
                raise RateLimitExceeded(key, retry_after)

    def get_remaining(self, key: str) -> float:
        """
        Get remaining tokens for a client.

        Args:
            key: Client identifier

        Returns:
            Number of remaining tokens
        """
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                return self.burst_limit
            bucket._refill()
            return bucket.tokens

    def reset(self, key: str) -> None:
        """Reset rate limit for a specific client."""
        with self._lock:
            if key in self._buckets:
                del self._buckets[key]

    def reset_all(self) -> None:
        """Reset all rate limits."""
        with self._lock:
            self._buckets.clear()

    def _get_or_create_bucket(self, key: str) -> TokenBucket:
        """Get existing bucket or create a new one."""
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(
                capacity=self.burst_limit, refill_rate=self._refill_rate
            )
        return self._buckets[key]

    def _maybe_cleanup(self) -> None:
        """Remove stale buckets to prevent memory leaks."""
        now = time.time()
        if now - self._last_cleanup < self.cleanup_interval:
            return

        # Remove buckets that are full (no recent activity)
        stale_keys = [
            key for key, bucket in self._buckets.items() if bucket.tokens >= bucket.capacity - 0.01
        ]
        for key in stale_keys:
            del self._buckets[key]

        self._last_cleanup = now


class SimulationRateLimiter(RateLimiter):
    """
    Specialized rate limiter for physics simulations.

    Applies higher costs for expensive operations like long simulations
    or high-resolution outputs.

    Args:
        simulations_per_minute: Base simulation rate
        max_points_free: Maximum points without extra cost
        points_per_token: Points that cost 1 token each
    """

    def __init__(
        self,
        simulations_per_minute: float = 30.0,
        burst_limit: float = 5.0,
        max_points_free: int = 1000,
        points_per_token: int = 1000,
    ):
        super().__init__(requests_per_minute=simulations_per_minute, burst_limit=burst_limit)
        self.max_points_free = max_points_free
        self.points_per_token = points_per_token

    def calculate_cost(
        self, num_points: int = 1000, time_span: float = 10.0, method: str = "RK45"
    ) -> float:
        """
        Calculate token cost for a simulation.

        Args:
            num_points: Number of output points
            time_span: Simulation time span
            method: Integration method

        Returns:
            Cost in tokens
        """
        base_cost = 1.0

        # Extra cost for high resolution
        if num_points > self.max_points_free:
            extra_points = num_points - self.max_points_free
            base_cost += extra_points / self.points_per_token

        # Extra cost for long simulations
        if time_span > 100:
            base_cost *= 1.5

        # Stiff solvers are more expensive
        if method in ("Radau", "BDF"):
            base_cost *= 1.2

        return base_cost

    def allow_simulation(
        self, key: str, num_points: int = 1000, time_span: float = 10.0, method: str = "RK45"
    ) -> bool:
        """
        Check if a simulation is allowed.

        Args:
            key: Client identifier
            num_points: Number of output points
            time_span: Simulation time span
            method: Integration method

        Returns:
            True if simulation is allowed
        """
        cost = self.calculate_cost(num_points, time_span, method)
        return self.allow(key, cost)


__all__ = [
    "RateLimiter",
    "SimulationRateLimiter",
    "RateLimitExceeded",
    "TokenBucket",
]
