"""
Unit tests for MechanicsDSL main utils.py module.

Tests utility functions and classes in the top-level utils.py file.
"""

import pytest
import numpy as np
import sympy as sp
import time
import logging

from mechanics_dsl.utils import (
    safe_float_conversion,
    PerformanceMonitor,
    LRUCache,
    AdvancedErrorHandler,
    resource_manager,
    runtime_type_check,
    validate_array_safe,
    safe_array_access,
    Config,
    setup_logging,
    DEFAULT_TRAIL_LENGTH,
    DEFAULT_FPS,
)


class TestSafeFloatConversion:
    """Tests for safe_float_conversion function."""
    
    def test_convert_int(self):
        """Test converting int to float."""
        assert safe_float_conversion(5) == 5.0
    
    def test_convert_float(self):
        """Test converting float."""
        assert safe_float_conversion(3.14) == pytest.approx(3.14)
    
    def test_convert_string(self):
        """Test converting numeric string."""
        assert safe_float_conversion("2.5") == pytest.approx(2.5)
    
    def test_convert_none(self):
        """Test converting None returns 0.0."""
        assert safe_float_conversion(None) == 0.0
    
    def test_convert_numpy_scalar(self):
        """Test converting numpy scalar."""
        assert safe_float_conversion(np.float64(1.5)) == pytest.approx(1.5)
    
    def test_convert_numpy_array_single(self):
        """Test converting single-element numpy array."""
        arr = np.array([3.0])
        assert safe_float_conversion(arr) == pytest.approx(3.0)
    
    def test_convert_numpy_array_multi(self):
        """Test converting multi-element numpy array takes first."""
        arr = np.array([1.0, 2.0, 3.0])
        assert safe_float_conversion(arr) == pytest.approx(1.0)
    
    def test_convert_numpy_bool(self):
        """Test converting numpy bool."""
        assert safe_float_conversion(np.bool_(True)) == 1.0
        assert safe_float_conversion(np.bool_(False)) == 0.0
    
    def test_convert_empty_array(self):
        """Test converting empty array."""
        assert safe_float_conversion(np.array([])) == 0.0
    
    def test_convert_inf(self):
        """Test converting infinity returns 0.0."""
        assert safe_float_conversion(float('inf')) == 0.0
    
    def test_convert_nan(self):
        """Test converting NaN returns 0.0."""
        assert safe_float_conversion(float('nan')) == 0.0
    
    def test_convert_invalid_string(self):
        """Test converting invalid string returns 0.0."""
        assert safe_float_conversion("not a number") == 0.0


class TestPerformanceMonitorLegacy:
    """Tests for PerformanceMonitor in utils.py."""
    
    def test_init(self):
        """Test initialization."""
        monitor = PerformanceMonitor()
        assert monitor is not None
        assert hasattr(monitor, 'metrics')
        assert hasattr(monitor, 'start_times')
    
    def test_start_stop_timer(self):
        """Test start and stop timer."""
        monitor = PerformanceMonitor()
        monitor.start_timer("test")
        time.sleep(0.01)
        duration = monitor.stop_timer("test")
        
        assert duration >= 0.01
        assert "test" in monitor.metrics
    
    def test_stop_timer_not_started(self):
        """Test stopping non-existent timer."""
        monitor = PerformanceMonitor()
        duration = monitor.stop_timer("nonexistent")
        assert duration == 0.0
    
    def test_start_timer_invalid(self):
        """Test starting timer with invalid name."""
        monitor = PerformanceMonitor()
        monitor.start_timer("")  # Should not crash
    
    def test_get_memory_usage(self):
        """Test getting memory usage."""
        monitor = PerformanceMonitor()
        mem = monitor.get_memory_usage()
        
        assert isinstance(mem, dict)
        assert 'rss' in mem
    
    def test_snapshot_memory(self):
        """Test memory snapshot."""
        monitor = PerformanceMonitor()
        monitor.snapshot_memory("test")
        
        assert len(monitor.memory_snapshots) >= 1
    
    def test_get_stats(self):
        """Test getting stats."""
        monitor = PerformanceMonitor()
        monitor.start_timer("test")
        time.sleep(0.01)
        monitor.stop_timer("test")
        
        stats = monitor.get_stats("test")
        assert 'count' in stats
        assert 'mean' in stats
    
    def test_get_stats_nonexistent(self):
        """Test getting stats for nonexistent metric."""
        monitor = PerformanceMonitor()
        stats = monitor.get_stats("nonexistent")
        assert stats == {}
    
    def test_reset(self):
        """Test reset clears all data."""
        monitor = PerformanceMonitor()
        monitor.start_timer("test")
        monitor.stop_timer("test")
        monitor.snapshot_memory("snap")
        
        monitor.reset()
        
        assert len(monitor.metrics) == 0
        assert len(monitor.memory_snapshots) == 0


class TestLRUCacheLegacy:
    """Tests for LRUCache in utils.py."""
    
    def test_init(self):
        """Test initialization."""
        cache = LRUCache()
        assert cache is not None
        assert cache.maxsize == 128
    
    def test_init_custom_size(self):
        """Test initialization with custom size."""
        cache = LRUCache(maxsize=50)
        assert cache.maxsize == 50
    
    def test_init_invalid_size(self):
        """Test initialization with invalid size uses default."""
        cache = LRUCache(maxsize=-1)
        assert cache.maxsize == 128
    
    def test_get_set(self):
        """Test get and set."""
        cache = LRUCache()
        cache.set("key", "value")
        assert cache.get("key") == "value"
    
    def test_get_nonexistent(self):
        """Test getting nonexistent key."""
        cache = LRUCache()
        assert cache.get("nonexistent") is None
    
    def test_get_invalid_key_type(self):
        """Test getting with invalid key type."""
        cache = LRUCache()
        assert cache.get(123) is None  # Expected str
    
    def test_set_invalid_key_type(self):
        """Test setting with invalid key type."""
        cache = LRUCache()
        cache.set(123, "value")  # Should not crash
    
    def test_lru_eviction(self):
        """Test LRU eviction."""
        cache = LRUCache(maxsize=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)  # Should evict "a"
        
        assert cache.get("a") is None
        assert cache.get("d") == 4
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = LRUCache()
        cache.set("key", "value")
        cache.get("key")  # Hit
        cache.get("nonexistent")  # Miss
        
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
    
    def test_clear(self):
        """Test cache clear."""
        cache = LRUCache()
        cache.set("key", "value")
        cache.clear()
        
        assert cache.get("key") is None
        assert len(cache.cache) == 0


class TestAdvancedErrorHandler:
    """Tests for AdvancedErrorHandler."""
    
    def test_retry_on_failure_success(self):
        """Test retry decorator with successful call."""
        @AdvancedErrorHandler.retry_on_failure(max_retries=3)
        def succeeds():
            return "success"
        
        assert succeeds() == "success"
    
    def test_retry_on_failure_eventual_success(self):
        """Test retry decorator with eventual success."""
        attempts = [0]
        
        @AdvancedErrorHandler.retry_on_failure(max_retries=3, delay=0.01)
        def fails_then_succeeds():
            attempts[0] += 1
            if attempts[0] < 3:
                raise ValueError("Not yet")
            return "success"
        
        assert fails_then_succeeds() == "success"
        assert attempts[0] == 3
    
    def test_retry_on_failure_all_fail(self):
        """Test retry decorator with all failures."""
        @AdvancedErrorHandler.retry_on_failure(max_retries=2, delay=0.01)
        def always_fails():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            always_fails()
    
    def test_safe_execute_success(self):
        """Test safe execute with successful call."""
        result = AdvancedErrorHandler.safe_execute(lambda: 42)
        assert result == 42
    
    def test_safe_execute_failure(self):
        """Test safe execute with failure returns default."""
        result = AdvancedErrorHandler.safe_execute(
            lambda: 1/0, 
            default="default",
            log_errors=False
        )
        assert result == "default"


class TestResourceManager:
    """Tests for resource_manager context manager."""
    
    def test_empty_resources(self):
        """Test with no resources."""
        with resource_manager():
            pass  # Should not crash
    
    def test_with_none_resource(self):
        """Test with None resource."""
        with resource_manager(None):
            pass  # Should not crash


class TestRuntimeTypeCheck:
    """Tests for runtime_type_check function."""
    
    def test_valid_type(self):
        """Test with valid type."""
        assert runtime_type_check(5, int) is True
        assert runtime_type_check("hello", str) is True
    
    def test_invalid_type(self):
        """Test with invalid type."""
        assert runtime_type_check(5, str) is False
        assert runtime_type_check("hello", int) is False
    
    def test_none_expected_type(self):
        """Test with None expected type."""
        assert runtime_type_check(5, None) is False
    
    def test_invalid_expected_type(self):
        """Test with non-type expected type."""
        assert runtime_type_check(5, "not a type") is False


class TestValidateArraySafe:
    """Tests for validate_array_safe function."""
    
    def test_valid_array(self):
        """Test with valid array."""
        arr = np.array([1.0, 2.0, 3.0])
        assert validate_array_safe(arr) is True
    
    def test_none_array(self):
        """Test with None."""
        assert validate_array_safe(None) is False
    
    def test_non_array(self):
        """Test with non-array."""
        assert validate_array_safe([1, 2, 3]) is False
    
    def test_min_size(self):
        """Test minimum size check."""
        arr = np.array([1.0])
        assert validate_array_safe(arr, min_size=5) is False
    
    def test_max_size(self):
        """Test maximum size check."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert validate_array_safe(arr, max_size=3) is False
    
    def test_check_finite(self):
        """Test finite value check."""
        arr = np.array([1.0, np.inf, 3.0])
        assert validate_array_safe(arr, check_finite=True) is False
    
    def test_skip_finite_check(self):
        """Test skipping finite check."""
        arr = np.array([1.0, np.inf, 3.0])
        assert validate_array_safe(arr, check_finite=False) is True


class TestSafeArrayAccess:
    """Tests for safe_array_access function."""
    
    def test_valid_access(self):
        """Test valid array access."""
        arr = np.array([1.0, 2.0, 3.0])
        assert safe_array_access(arr, 1) == pytest.approx(2.0)
    
    def test_none_array(self):
        """Test with None array."""
        assert safe_array_access(None, 0, default=99.0) == 99.0
    
    def test_non_array(self):
        """Test with non-array."""
        assert safe_array_access([1, 2, 3], 0, default=99.0) == 99.0
    
    def test_out_of_bounds(self):
        """Test out of bounds access."""
        arr = np.array([1.0, 2.0, 3.0])
        assert safe_array_access(arr, 10, default=99.0) == 99.0
    
    def test_negative_index(self):
        """Test negative index."""
        arr = np.array([1.0, 2.0, 3.0])
        assert safe_array_access(arr, -1, default=99.0) == 99.0
    
    def test_non_int_index(self):
        """Test non-int index."""
        arr = np.array([1.0, 2.0, 3.0])
        assert safe_array_access(arr, 1.5, default=99.0) == 99.0


class TestConfig:
    """Tests for Config class."""
    
    def test_init(self):
        """Test initialization."""
        config = Config()
        assert config is not None
    
    def test_default_values(self):
        """Test default values."""
        config = Config()
        assert config.trail_length == DEFAULT_TRAIL_LENGTH
        assert config.animation_fps == DEFAULT_FPS
    
    def test_enable_profiling(self):
        """Test enable_profiling property."""
        config = Config()
        config.enable_profiling = True
        assert config.enable_profiling is True
    
    def test_enable_profiling_invalid(self):
        """Test enable_profiling with invalid value."""
        config = Config()
        with pytest.raises(TypeError):
            config.enable_profiling = "not a bool"
    
    def test_trail_length(self):
        """Test trail_length property."""
        config = Config()
        config.trail_length = 200
        assert config.trail_length == 200
    
    def test_trail_length_invalid_type(self):
        """Test trail_length with invalid type."""
        config = Config()
        with pytest.raises(TypeError):
            config.trail_length = "not an int"
    
    def test_trail_length_negative(self):
        """Test trail_length with negative value."""
        config = Config()
        with pytest.raises(ValueError):
            config.trail_length = -10
    
    def test_animation_fps(self):
        """Test animation_fps property."""
        config = Config()
        config.animation_fps = 60
        assert config.animation_fps == 60
    
    def test_animation_fps_invalid(self):
        """Test animation_fps with invalid values."""
        config = Config()
        with pytest.raises(ValueError):
            config.animation_fps = 0
    
    def test_simplification_timeout(self):
        """Test simplification_timeout property."""
        config = Config()
        config.simplification_timeout = 10.0
        assert config.simplification_timeout == 10.0
    
    def test_simplification_timeout_invalid(self):
        """Test simplification_timeout with invalid values."""
        config = Config()
        with pytest.raises(ValueError):
            config.simplification_timeout = -1.0
    
    def test_default_rtol(self):
        """Test default_rtol property."""
        config = Config()
        config.default_rtol = 1e-8
        assert config.default_rtol == pytest.approx(1e-8)
    
    def test_default_rtol_invalid(self):
        """Test default_rtol with invalid values."""
        config = Config()
        with pytest.raises(ValueError):
            config.default_rtol = 0.0
    
    def test_to_dict(self):
        """Test converting config to dict."""
        config = Config()
        d = config.to_dict()
        
        assert isinstance(d, dict)
        assert 'enable_profiling' in d
        assert 'trail_length' in d
    
    def test_from_dict(self):
        """Test loading config from dict."""
        config = Config()
        config.from_dict({'trail_length': 300})
        
        assert config.trail_length == 300
    
    def test_from_dict_invalid_key(self):
        """Test from_dict with unknown key."""
        config = Config()
        config.from_dict({'unknown_key': 'value'})  # Should not crash
    
    def test_cache_max_size(self):
        """Test cache_max_size property."""
        config = Config()
        config.cache_max_size = 500
        assert config.cache_max_size == 500
    
    def test_cache_max_memory_mb(self):
        """Test cache_max_memory_mb property."""
        config = Config()
        config.cache_max_memory_mb = 500.0
        assert config.cache_max_memory_mb == 500.0


class TestSetupLogging:
    """Tests for setup_logging function."""
    
    def test_setup_logging_returns_logger(self):
        """Test that setup_logging returns a logger."""
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)
    
    def test_setup_logging_sets_level(self):
        """Test that setup_logging sets level."""
        logger = setup_logging(level=logging.DEBUG)
        assert logger.level == logging.DEBUG
