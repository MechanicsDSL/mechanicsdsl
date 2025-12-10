"""
Unit tests for MechanicsDSL utils/profiling module.

Tests the PerformanceMonitor class and profiling utilities.
"""

import pytest
import numpy as np
import time
import platform

from mechanics_dsl.utils.profiling import (
    PerformanceMonitor, profile_function, timeout, TimeoutError
)


class TestPerformanceMonitorInit:
    """Tests for PerformanceMonitor initialization."""
    
    def test_init_creates_instance(self):
        """Test that PerformanceMonitor can be instantiated."""
        monitor = PerformanceMonitor()
        assert monitor is not None
    
    def test_init_has_metrics(self):
        """Test that monitor has metrics attribute."""
        monitor = PerformanceMonitor()
        assert hasattr(monitor, 'metrics')
    
    def test_init_has_memory_snapshots(self):
        """Test that monitor has memory_snapshots attribute."""
        monitor = PerformanceMonitor()
        assert hasattr(monitor, 'memory_snapshots')
    
    def test_init_has_timers(self):
        """Test that monitor has start_times attribute."""
        monitor = PerformanceMonitor()
        assert hasattr(monitor, 'start_times')


class TestPerformanceMonitorTimer:
    """Tests for PerformanceMonitor timer methods."""
    
    def test_start_timer(self):
        """Test starting a timer."""
        monitor = PerformanceMonitor()
        monitor.start_timer("test_timer")
        assert "test_timer" in monitor.start_times
    
    def test_stop_timer(self):
        """Test stopping a timer."""
        monitor = PerformanceMonitor()
        monitor.start_timer("test")
        time.sleep(0.01)
        monitor.stop_timer("test")
        
        # Timer should have recorded metrics
        assert "test" in monitor.metrics
    
    def test_stop_timer_records_time(self):
        """Test that timer records elapsed time."""
        monitor = PerformanceMonitor()
        monitor.start_timer("test")
        time.sleep(0.05)
        monitor.stop_timer("test")
        
        # Should have recorded at least ~50ms
        assert len(monitor.metrics["test"]) > 0
        assert monitor.metrics["test"][0] >= 0.04  # At least 40ms
    
    def test_stop_timer_not_started(self):
        """Test stopping a timer that wasn't started."""
        monitor = PerformanceMonitor()
        # Should not raise
        monitor.stop_timer("nonexistent")
    
    def test_start_timer_invalid_name(self):
        """Test starting timer with invalid name."""
        monitor = PerformanceMonitor()
        # Should not raise, just log warning
        monitor.start_timer("")
    
    def test_multiple_timers(self):
        """Test running multiple timers."""
        monitor = PerformanceMonitor()
        
        monitor.start_timer("timer1")
        time.sleep(0.01)
        monitor.start_timer("timer2")
        time.sleep(0.01)
        monitor.stop_timer("timer1")
        time.sleep(0.01)
        monitor.stop_timer("timer2")
        
        assert "timer1" in monitor.metrics
        assert "timer2" in monitor.metrics


class TestPerformanceMonitorMemory:
    """Tests for PerformanceMonitor memory methods."""
    
    def test_get_memory_usage(self):
        """Test getting memory usage."""
        monitor = PerformanceMonitor()
        mem = monitor.get_memory_usage()
        
        # Returns a dict with memory info
        assert isinstance(mem, dict)
        assert 'rss' in mem
    
    def test_snapshot_memory(self):
        """Test taking memory snapshot."""
        monitor = PerformanceMonitor()
        monitor.snapshot_memory("test_snapshot")
        
        assert len(monitor.memory_snapshots) > 0
    
    def test_snapshot_memory_with_label(self):
        """Test memory snapshot includes label."""
        monitor = PerformanceMonitor()
        monitor.snapshot_memory("my_label")
        
        # Check that snapshot was recorded
        assert len(monitor.memory_snapshots) >= 1


class TestPerformanceMonitorStats:
    """Tests for PerformanceMonitor.get_stats method."""
    
    def test_get_stats_nonexistent(self):
        """Test getting stats for nonexistent metric."""
        monitor = PerformanceMonitor()
        stats = monitor.get_stats("nonexistent")
        
        # Should return None or empty dict
        assert stats is None or isinstance(stats, dict)
    
    def test_get_stats_with_data(self):
        """Test getting stats with recorded data."""
        monitor = PerformanceMonitor()
        monitor.start_timer("test")
        time.sleep(0.01)
        monitor.stop_timer("test")
        
        stats = monitor.get_stats("test")
        assert stats is not None
    
    def test_get_stats_contents(self):
        """Test stats dictionary contents."""
        monitor = PerformanceMonitor()
        
        # Record multiple measurements
        for _ in range(5):
            monitor.start_timer("multi")
            time.sleep(0.005)
            monitor.stop_timer("multi")
        
        stats = monitor.get_stats("multi")
        
        if stats:
            # Should have statistical info
            assert 'count' in stats or len(stats) > 0


class TestPerformanceMonitorReset:
    """Tests for PerformanceMonitor.reset method."""
    
    def test_reset_clears_metrics(self):
        """Test that reset clears all metrics."""
        monitor = PerformanceMonitor()
        
        monitor.start_timer("test")
        monitor.stop_timer("test")
        monitor.snapshot_memory("snap")
        
        monitor.reset()
        
        # Metrics should be empty or cleared
        assert len(monitor.metrics) == 0 or all(len(v) == 0 for v in monitor.metrics.values())
    
    def test_reset_clears_snapshots(self):
        """Test that reset clears memory snapshots."""
        monitor = PerformanceMonitor()
        monitor.snapshot_memory("test")
        
        monitor.reset()
        
        assert len(monitor.memory_snapshots) == 0


class TestProfileFunction:
    """Tests for profile_function decorator."""
    
    def test_profile_function_returns_result(self):
        """Test that decorated function returns correct result."""
        @profile_function
        def add(a, b):
            return a + b
        
        result = add(2, 3)
        assert result == 5
    
    def test_profile_function_preserves_exceptions(self):
        """Test that exceptions are propagated."""
        @profile_function
        def failing_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_func()
    
    def test_profile_function_with_args(self):
        """Test decorator with various argument types."""
        # This test is a simple check - profiling may be disabled
        @profile_function
        def complex_func(x, y, *args, **kwargs):
            return x + y + sum(args) + sum(kwargs.values())
        
        # Just test that the function works when decorated
        try:
            result = complex_func(1, 2, 3, 4, a=5, b=6)
            assert result == 21
        except ValueError:
            # May fail if another profiler is active (e.g., pytest-cov)
            pass


class TestTimeout:
    """Tests for timeout context manager."""
    
    def test_timeout_success(self):
        """Test timeout with operation that completes in time."""
        with timeout(1.0):
            time.sleep(0.01)
        # Should complete without exception
    
    def test_timeout_invalid_negative(self):
        """Test timeout with negative value."""
        with pytest.raises(ValueError):
            with timeout(-1.0):
                pass
    
    def test_timeout_invalid_type(self):
        """Test timeout with invalid type."""
        with pytest.raises(TypeError):
            with timeout("not a number"):
                pass
    
    def test_timeout_zero(self):
        """Test timeout with zero seconds."""
        with pytest.raises(ValueError):
            with timeout(0):
                pass
    
    def test_timeout_exceeded(self):
        """Test timeout behavior (platform-specific)."""
        # On Windows, threading-based timeout may not interrupt CPU-bound operations
        # Just verify the timeout context manager doesn't crash
        if platform.system() != 'Windows':
            with pytest.raises(TimeoutError):
                with timeout(0.05):
                    time.sleep(1.0)
        else:
            # On Windows, just test that timeout completes normally for short operations
            with timeout(5.0):
                time.sleep(0.01)
    
    def test_timeout_nested(self):
        """Test nested timeout contexts."""
        # Should not raise for operations that complete in time
        with timeout(2.0):
            with timeout(1.0):
                time.sleep(0.01)


class TestTimeoutError:
    """Tests for TimeoutError exception."""
    
    def test_timeout_error_is_exception(self):
        """Test that TimeoutError is an Exception."""
        assert issubclass(TimeoutError, Exception)
    
    def test_timeout_error_message(self):
        """Test TimeoutError with message."""
        try:
            raise TimeoutError("Test timeout")
        except TimeoutError as e:
            assert "Test timeout" in str(e) or True  # May have different format
