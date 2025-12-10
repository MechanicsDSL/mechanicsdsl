"""
Unit tests for the LEGACY utils.py module at top level.

Imports from mechanics_dsl.utils which uses the utils/ subpackage.
This tests the utils/ subpackage which covers most functionality.
The top-level utils.py is a legacy file with duplicate code.
"""

import pytest
import numpy as np
import time
import logging

# Import from the package as usual - pytest-cov should trace these
from mechanics_dsl.utils import (
    safe_float_conversion,
    PerformanceMonitor,
    LRUCache,
    validate_array_safe,
    safe_array_access,
    runtime_type_check,
    resource_manager,
    AdvancedErrorHandler,
    Config,
    config,
    setup_logging,
    logger,
    DEFAULT_TRAIL_LENGTH,
    DEFAULT_FPS,
)


class TestSafeFloatConversion:
    """Tests for safe_float_conversion function."""
    
    def test_convert_int(self):
        result = safe_float_conversion(5)
        assert result == 5.0
    
    def test_convert_float(self):
        result = safe_float_conversion(3.14)
        assert result == pytest.approx(3.14)
    
    def test_convert_string(self):
        result = safe_float_conversion("2.5")
        assert result == pytest.approx(2.5)
    
    def test_convert_none(self):
        result = safe_float_conversion(None)
        assert result == 0.0
    
    def test_convert_numpy_scalar(self):
        result = safe_float_conversion(np.float64(1.5))
        assert result == pytest.approx(1.5)
    
    def test_convert_numpy_int(self):
        result = safe_float_conversion(np.int32(10))
        assert result == 10.0
    
    def test_convert_numpy_array_single(self):
        arr = np.array([3.0])
        result = safe_float_conversion(arr)
        assert result == pytest.approx(3.0)
    
    def test_convert_numpy_array_multi(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = safe_float_conversion(arr)
        assert result == pytest.approx(1.0)
    
    def test_convert_numpy_bool(self):
        assert safe_float_conversion(np.bool_(True)) == 1.0
        assert safe_float_conversion(np.bool_(False)) == 0.0
    
    def test_convert_empty_array(self):
        result = safe_float_conversion(np.array([]))
        assert result == 0.0
    
    def test_convert_inf(self):
        result = safe_float_conversion(float('inf'))
        assert result == 0.0
    
    def test_convert_nan(self):
        result = safe_float_conversion(float('nan'))
        assert result == 0.0
    
    def test_convert_invalid_string(self):
        result = safe_float_conversion("not a number")
        assert result == 0.0


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor."""
    
    def test_init(self):
        monitor = PerformanceMonitor()
        assert monitor is not None
    
    def test_init_has_metrics(self):
        monitor = PerformanceMonitor()
        assert hasattr(monitor, 'metrics')
    
    def test_init_has_start_times(self):
        monitor = PerformanceMonitor()
        assert hasattr(monitor, 'start_times')
    
    def test_start_timer(self):
        monitor = PerformanceMonitor()
        monitor.start_timer("test")
        assert "test" in monitor.start_times
    
    def test_stop_timer(self):
        monitor = PerformanceMonitor()
        monitor.start_timer("test")
        time.sleep(0.01)
        duration = monitor.stop_timer("test")
        assert duration >= 0.01
        assert "test" in monitor.metrics
    
    def test_stop_timer_not_started(self):
        monitor = PerformanceMonitor()
        duration = monitor.stop_timer("nonexistent")
        assert duration == 0.0
    
    def test_get_memory_usage(self):
        monitor = PerformanceMonitor()
        mem = monitor.get_memory_usage()
        assert isinstance(mem, dict)
        assert 'rss' in mem
    
    def test_snapshot_memory(self):
        monitor = PerformanceMonitor()
        monitor.snapshot_memory("test_label")
        assert len(monitor.memory_snapshots) >= 1
    
    def test_get_stats(self):
        monitor = PerformanceMonitor()
        monitor.start_timer("test")
        time.sleep(0.01)
        monitor.stop_timer("test")
        stats = monitor.get_stats("test")
        assert 'count' in stats
        assert 'mean' in stats
    
    def test_get_stats_nonexistent(self):
        monitor = PerformanceMonitor()
        stats = monitor.get_stats("nonexistent")
        assert stats is None or stats == {}
    
    def test_reset(self):
        monitor = PerformanceMonitor()
        monitor.start_timer("test")
        monitor.stop_timer("test")
        monitor.snapshot_memory("snap")
        monitor.reset()
        assert len(monitor.metrics) == 0
        assert len(monitor.memory_snapshots) == 0


class TestLRUCache:
    """Tests for LRUCache."""
    
    def test_init(self):
        cache = LRUCache()
        assert cache is not None
    
    def test_init_custom_size(self):
        cache = LRUCache(maxsize=50)
        assert cache.maxsize == 50
    
    def test_get_set(self):
        cache = LRUCache()
        cache.set("key", "value")
        assert cache.get("key") == "value"
    
    def test_get_nonexistent(self):
        cache = LRUCache()
        assert cache.get("nonexistent") is None
    
    def test_lru_eviction(self):
        cache = LRUCache(maxsize=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)
        assert cache.get("a") is None
        assert cache.get("d") == 4
    
    def test_cache_stats(self):
        cache = LRUCache()
        cache.set("key", "value")
        cache.get("key")
        cache.get("nonexistent")
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
    
    def test_clear(self):
        cache = LRUCache()
        cache.set("key", "value")
        cache.clear()
        assert cache.get("key") is None
    
    def test_estimate_memory(self):
        cache = LRUCache()
        cache.set("key", np.array([1, 2, 3, 4, 5]))
        mem = cache._estimate_memory_mb()
        assert mem >= 0


class TestAdvancedErrorHandler:
    """Tests for AdvancedErrorHandler."""
    
    def test_retry_on_failure_success(self):
        @AdvancedErrorHandler.retry_on_failure(max_retries=3)
        def succeeds():
            return "success"
        assert succeeds() == "success"
    
    def test_retry_on_failure_eventual(self):
        attempts = [0]
        
        @AdvancedErrorHandler.retry_on_failure(max_retries=3, delay=0.01)
        def fails_then_succeeds():
            attempts[0] += 1
            if attempts[0] < 3:
                raise ValueError("Not yet")
            return "success"
        
        assert fails_then_succeeds() == "success"
    
    def test_retry_on_failure_all_fail(self):
        @AdvancedErrorHandler.retry_on_failure(max_retries=2, delay=0.01)
        def always_fails():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            always_fails()
    
    def test_safe_execute_success(self):
        result = AdvancedErrorHandler.safe_execute(lambda: 42)
        assert result == 42
    
    def test_safe_execute_failure(self):
        result = AdvancedErrorHandler.safe_execute(
            lambda: 1/0, default="default", log_errors=False
        )
        assert result == "default"


class TestResourceManager:
    """Tests for resource_manager."""
    
    def test_empty_resources(self):
        with resource_manager():
            pass
    
    def test_with_none_resource(self):
        with resource_manager(None):
            pass


class TestRuntimeTypeCheck:
    """Tests for runtime_type_check."""
    
    def test_valid_type(self):
        assert runtime_type_check(5, int) is True
        assert runtime_type_check("hello", str) is True
    
    def test_invalid_type(self):
        assert runtime_type_check(5, str) is False
        assert runtime_type_check("hello", int) is False


class TestValidateArraySafe:
    """Tests for validate_array_safe."""
    
    def test_valid_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert validate_array_safe(arr) is True
    
    def test_none_array(self):
        assert validate_array_safe(None) is False
    
    def test_non_array(self):
        assert validate_array_safe([1, 2, 3]) is False
    
    def test_min_size(self):
        arr = np.array([1.0])
        assert validate_array_safe(arr, min_size=5) is False
    
    def test_max_size(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert validate_array_safe(arr, max_size=3) is False
    
    def test_check_finite(self):
        arr = np.array([1.0, np.inf, 3.0])
        assert validate_array_safe(arr, check_finite=True) is False


class TestSafeArrayAccess:
    """Tests for safe_array_access."""
    
    def test_valid_access(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = safe_array_access(arr, 1)
        assert result == pytest.approx(2.0)
    
    def test_none_array(self):
        result = safe_array_access(None, 0, default=99.0)
        assert result == 99.0
    
    def test_out_of_bounds(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = safe_array_access(arr, 10, default=99.0)
        assert result == 99.0
    
    def test_negative_index(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = safe_array_access(arr, -1, default=99.0)
        assert result == 99.0


class TestConfig:
    """Tests for Config."""
    
    def test_init(self):
        cfg = Config()
        assert cfg is not None
    
    def test_enable_profiling(self):
        cfg = Config()
        cfg.enable_profiling = True
        assert cfg.enable_profiling is True
    
    def test_enable_profiling_invalid(self):
        cfg = Config()
        with pytest.raises(TypeError):
            cfg.enable_profiling = "not a bool"
    
    def test_trail_length(self):
        cfg = Config()
        cfg.trail_length = 200
        assert cfg.trail_length == 200
    
    def test_trail_length_invalid_type(self):
        cfg = Config()
        with pytest.raises(TypeError):
            cfg.trail_length = "not an int"
    
    def test_trail_length_negative(self):
        cfg = Config()
        with pytest.raises(ValueError):
            cfg.trail_length = -10
    
    def test_animation_fps(self):
        cfg = Config()
        cfg.animation_fps = 60
        assert cfg.animation_fps == 60
    
    def test_animation_fps_invalid(self):
        cfg = Config()
        with pytest.raises(ValueError):
            cfg.animation_fps = 0
    
    def test_simplification_timeout(self):
        cfg = Config()
        cfg.simplification_timeout = 10.0
        assert cfg.simplification_timeout == 10.0
    
    def test_simplification_timeout_invalid(self):
        cfg = Config()
        with pytest.raises(ValueError):
            cfg.simplification_timeout = -1.0
    
    def test_default_rtol(self):
        cfg = Config()
        cfg.default_rtol = 1e-8
        assert cfg.default_rtol == pytest.approx(1e-8)
    
    def test_default_rtol_invalid(self):
        cfg = Config()
        with pytest.raises(ValueError):
            cfg.default_rtol = 0.0
    
    def test_default_atol(self):
        cfg = Config()
        cfg.default_atol = 1e-10
        assert cfg.default_atol == pytest.approx(1e-10)
    
    def test_max_parser_errors(self):
        cfg = Config()
        cfg.max_parser_errors = 20
        assert cfg.max_parser_errors == 20
    
    def test_cache_max_size(self):
        cfg = Config()
        cfg.cache_max_size = 500
        assert cfg.cache_max_size == 500
    
    def test_cache_max_memory_mb(self):
        cfg = Config()
        cfg.cache_max_memory_mb = 500.0
        assert cfg.cache_max_memory_mb == 500.0
    
    def test_enable_adaptive_solver(self):
        cfg = Config()
        cfg.enable_adaptive_solver = False
        assert cfg.enable_adaptive_solver is False
    
    def test_enable_debug_logging(self):
        cfg = Config()
        cfg.enable_debug_logging = True
        assert cfg.enable_debug_logging is True
    
    def test_save_intermediate_results(self):
        cfg = Config()
        cfg.save_intermediate_results = True
        assert cfg.save_intermediate_results is True
    
    def test_cache_symbolic_results(self):
        cfg = Config()
        cfg.cache_symbolic_results = False
        assert cfg.cache_symbolic_results is False
    
    def test_enable_performance_monitoring(self):
        cfg = Config()
        cfg.enable_performance_monitoring = False
        assert cfg.enable_performance_monitoring is False
    
    def test_enable_memory_monitoring(self):
        cfg = Config()
        cfg.enable_memory_monitoring = False
        assert cfg.enable_memory_monitoring is False
    
    def test_error_recovery_enabled(self):
        cfg = Config()
        cfg.error_recovery_enabled = False
        assert cfg.error_recovery_enabled is False
    
    def test_to_dict(self):
        cfg = Config()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert 'enable_profiling' in d
    
    def test_from_dict(self):
        cfg = Config()
        cfg.from_dict({'trail_length': 300})
        assert cfg.trail_length == 300
    
    def test_from_dict_invalid_key(self):
        cfg = Config()
        cfg.from_dict({'unknown_key': 'value'})


class TestSetupLogging:
    """Tests for setup_logging."""
    
    def test_setup_logging_returns_logger(self):
        result = setup_logging()
        assert isinstance(result, logging.Logger)
    
    def test_setup_logging_sets_level(self):
        result = setup_logging(level=logging.DEBUG)
        assert result.level == logging.DEBUG


class TestConstants:
    """Tests for constants."""
    
    def test_default_trail_length(self):
        assert DEFAULT_TRAIL_LENGTH == 150
    
    def test_default_fps(self):
        assert DEFAULT_FPS == 30


class TestGlobalConfig:
    """Tests for global config instance."""
    
    def test_global_config_exists(self):
        assert config is not None
    
    def test_global_config_is_config(self):
        assert isinstance(config, Config)


class TestLogger:
    """Tests for logger."""
    
    def test_logger_exists(self):
        assert logger is not None
    
    def test_logger_is_logger(self):
        assert isinstance(logger, logging.Logger)
