"""
Comprehensive tests for utils.py
"""

import os
import tempfile
import time

import numpy as np
import pytest

from mechanics_dsl.utils import (
    AdvancedErrorHandler,
    Config,
    LRUCache,
    PerformanceMonitor,
    config,
    profile_function,
    resource_manager,
    runtime_type_check,
    safe_array_access,
    safe_float_conversion,
    timeout,
    validate_array_safe,
    validate_file_path,
    validate_finite,
    validate_non_negative,
    validate_positive,
    validate_solution_dict,
    validate_time_span,
)


class TestSafeFloatConversion:
    """Test safe_float_conversion function"""

    def test_none_value(self):
        """Test conversion of None"""
        result = safe_float_conversion(None)
        assert result == 0.0

    def test_numpy_array_empty(self):
        """Test conversion of empty numpy array"""
        arr = np.array([])
        result = safe_float_conversion(arr)
        assert result == 0.0

    def test_numpy_array_single_element(self):
        """Test conversion of single-element numpy array"""
        arr = np.array([5.5])
        result = safe_float_conversion(arr)
        assert result == 5.5

    def test_numpy_array_multiple_elements(self):
        """Test conversion of multi-element numpy array"""
        arr = np.array([1.0, 2.0, 3.0])
        result = safe_float_conversion(arr)
        assert result == 1.0  # Takes first element

    def test_numpy_array_non_finite(self):
        """Test conversion of non-finite numpy array"""
        arr = np.array([np.inf])
        result = safe_float_conversion(arr)
        assert result == 0.0

    def test_numpy_integer(self):
        """Test conversion of numpy integer"""
        val = np.int64(42)
        result = safe_float_conversion(val)
        assert result == 42.0

    def test_numpy_floating(self):
        """Test conversion of numpy floating"""
        val = np.float64(3.14)
        result = safe_float_conversion(val)
        assert result == 3.14

    def test_numpy_bool(self):
        """Test conversion of numpy bool"""
        val = np.bool_(True)
        result = safe_float_conversion(val)
        assert result == 1.0

    def test_python_int(self):
        """Test conversion of Python int"""
        result = safe_float_conversion(42)
        assert result == 42.0

    def test_python_float(self):
        """Test conversion of Python float"""
        result = safe_float_conversion(3.14)
        assert result == 3.14

    def test_python_float_non_finite(self):
        """Test conversion of non-finite Python float"""
        result = safe_float_conversion(float("inf"))
        assert result == 0.0

    def test_string_valid(self):
        """Test conversion of valid string"""
        result = safe_float_conversion("3.14")
        assert result == 3.14

    def test_string_invalid(self):
        """Test conversion of invalid string"""
        result = safe_float_conversion("not a number")
        assert result == 0.0

    def test_string_non_finite(self):
        """Test conversion of string representing non-finite value"""
        result = safe_float_conversion("inf")
        result2 = safe_float_conversion("nan")
        assert result == 0.0 or np.isinf(result)
        assert result2 == 0.0 or np.isnan(result2)


class TestPerformanceMonitor:
    """Test PerformanceMonitor class"""

    def test_monitor_initialization(self):
        """Test monitor initialization"""
        monitor = PerformanceMonitor()
        assert monitor is not None

    def test_start_stop_timer(self):
        """Test timer start/stop"""
        monitor = PerformanceMonitor()
        monitor.start_timer("test_timer")
        time.sleep(0.01)
        monitor.stop_timer("test_timer")
        stats = monitor.get_stats("test_timer")
        assert stats is not None
        assert "elapsed_time" in stats or "count" in stats

    def test_snapshot_memory(self):
        """Test memory snapshot"""
        monitor = PerformanceMonitor()
        monitor.snapshot_memory("test_snapshot")
        # Should not raise

    def test_get_stats_nonexistent(self):
        """Test getting stats for nonexistent timer"""
        monitor = PerformanceMonitor()
        stats = monitor.get_stats("nonexistent")
        assert stats is None or isinstance(stats, dict)


class TestLRUCache:
    """Test LRUCache class"""

    def test_cache_initialization(self):
        """Test cache initialization"""
        cache = LRUCache(maxsize=5)
        assert cache.maxsize == 5
        stats = cache.get_stats()
        assert stats["size"] == 0

    def test_cache_set_get(self):
        """Test cache set and get"""
        cache = LRUCache(maxsize=3)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_cache_eviction(self):
        """Test LRU eviction"""
        cache = LRUCache(maxsize=2)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_cache_get_nonexistent(self):
        """Test getting nonexistent key"""
        cache = LRUCache()
        assert cache.get("nonexistent") is None

    def test_cache_clear(self):
        """Test cache clearing"""
        cache = LRUCache()
        cache.set("key1", "value1")
        cache.clear()
        stats = cache.get_stats()
        assert stats["size"] == 0
        assert cache.get("key1") is None

    def test_cache_set_none_value(self):
        """Test setting None value"""
        cache = LRUCache()
        cache.set("key1", None)
        # Should handle None gracefully
        assert cache.get("key1") is None

    def test_cache_get_stats(self):
        """Test getting cache statistics"""
        cache = LRUCache(maxsize=5)
        cache.set("key1", "value1")
        cache.get("key1")
        stats = cache.get_stats()
        assert isinstance(stats, dict)
        assert "hits" in stats
        assert "size" in stats
        assert stats["hits"] >= 1


class TestAdvancedErrorHandler:
    """Test AdvancedErrorHandler class"""

    def test_retry_on_failure_success(self):
        """Test retry decorator with success"""
        handler = AdvancedErrorHandler()

        @handler.retry_on_failure(max_retries=3)
        def successful_func():
            return 42

        result = successful_func()
        assert result == 42

    def test_retry_on_failure_failure(self):
        """Test retry decorator with failure"""
        handler = AdvancedErrorHandler()

        @handler.retry_on_failure(max_retries=2, delay=0.01)
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_func()

    def test_safe_execute_success(self):
        """Test safe_execute with success"""
        handler = AdvancedErrorHandler()
        result = handler.safe_execute(lambda: 42)
        assert result == 42

    def test_safe_execute_failure(self):
        """Test safe_execute with failure"""
        handler = AdvancedErrorHandler()
        result = handler.safe_execute(lambda: 1 / 0, default=0)
        assert result == 0

    def test_safe_execute_no_log(self):
        """Test safe_execute without logging"""
        handler = AdvancedErrorHandler()
        result = handler.safe_execute(lambda: 1 / 0, default=0, log_errors=False)
        assert result == 0


class TestResourceManager:
    """Test resource_manager context manager"""

    def test_resource_manager_success(self):
        """Test resource manager with successful operations"""
        with resource_manager():
            pass  # Should not raise

    def test_resource_manager_with_resources(self):
        """Test resource manager with resources"""
        file1 = tempfile.NamedTemporaryFile(delete=False)
        file2 = tempfile.NamedTemporaryFile(delete=False)
        try:
            with resource_manager(file1, file2):
                pass
        finally:
            os.unlink(file1.name)
            os.unlink(file2.name)


class TestRuntimeTypeCheck:
    """Test runtime_type_check function"""

    def test_type_check_success(self):
        """Test successful type check"""
        assert runtime_type_check(42, int) is True
        assert runtime_type_check(3.14, float) is True
        assert runtime_type_check("hello", str) is True

    def test_type_check_failure(self):
        """Test failed type check"""
        assert runtime_type_check(42, str) is False
        assert runtime_type_check("hello", int) is False


class TestValidateArraySafe:
    """Test validate_array_safe function"""

    def test_validate_array_none(self):
        """Test validation of None"""
        assert validate_array_safe(None, "test") is False

    def test_validate_array_not_array(self):
        """Test validation of non-array"""
        assert validate_array_safe([1, 2, 3], "test") is False

    def test_validate_array_too_small(self):
        """Test validation of array that's too small"""
        arr = np.array([1])
        assert validate_array_safe(arr, "test", min_size=2) is False

    def test_validate_array_too_large(self):
        """Test validation of array that's too large"""
        arr = np.array([1, 2, 3])
        assert validate_array_safe(arr, "test", max_size=2) is False

    def test_validate_array_non_finite(self):
        """Test validation of array with non-finite values"""
        arr = np.array([1.0, np.inf, 3.0])
        assert validate_array_safe(arr, "test", check_finite=True) is False

    def test_validate_array_success(self):
        """Test successful validation"""
        arr = np.array([1.0, 2.0, 3.0])
        assert validate_array_safe(arr, "test") is True
        assert validate_array_safe(arr, "test", min_size=1, max_size=5) is True


class TestSafeArrayAccess:
    """Test safe_array_access function"""

    def test_safe_access_success(self):
        """Test successful array access"""
        arr = np.array([1.0, 2.0, 3.0])
        assert safe_array_access(arr, 0) == 1.0
        assert safe_array_access(arr, 1) == 2.0

    def test_safe_access_out_of_bounds(self):
        """Test out-of-bounds access"""
        arr = np.array([1.0, 2.0])
        assert safe_array_access(arr, 10, default=0.0) == 0.0

    def test_safe_access_negative_index(self):
        """Test negative index"""
        arr = np.array([1.0, 2.0])
        assert safe_array_access(arr, -1, default=0.0) == 0.0

    def test_safe_access_none_array(self):
        """Test access with None array"""
        assert safe_array_access(None, 0, default=0.0) == 0.0

    def test_safe_access_non_array(self):
        """Test access with non-array"""
        assert safe_array_access([1, 2, 3], 0, default=0.0) == 0.0

    def test_safe_access_non_int_index(self):
        """Test access with non-integer index"""
        arr = np.array([1.0, 2.0])
        assert safe_array_access(arr, "0", default=0.0) == 0.0

    def test_safe_access_non_finite_value(self):
        """Test access with non-finite value"""
        arr = np.array([1.0, np.inf, 3.0])
        assert safe_array_access(arr, 1, default=0.0) == 0.0


class TestConfig:
    """Test Config class"""

    def test_config_properties(self):
        """Test config property getters and setters"""
        cfg = Config()

        # Test enable_profiling
        cfg.enable_profiling = True
        assert cfg.enable_profiling is True
        with pytest.raises(TypeError):
            cfg.enable_profiling = "not bool"

        # Test enable_debug_logging
        cfg.enable_debug_logging = True
        assert cfg.enable_debug_logging is True

        # Test simplification_timeout
        cfg.simplification_timeout = 10.0
        assert cfg.simplification_timeout == 10.0
        with pytest.raises(ValueError):
            cfg.simplification_timeout = -1.0
        with pytest.raises(ValueError):
            cfg.simplification_timeout = 10000.0

        # Test max_parser_errors
        cfg.max_parser_errors = 20
        assert cfg.max_parser_errors == 20
        with pytest.raises(ValueError):
            cfg.max_parser_errors = 0
        with pytest.raises(ValueError):
            cfg.max_parser_errors = 2000

        # Test default_rtol
        cfg.default_rtol = 1e-5
        assert cfg.default_rtol == 1e-5
        with pytest.raises(ValueError):
            cfg.default_rtol = 0.0
        with pytest.raises(ValueError):
            cfg.default_rtol = 1.0

        # Test default_atol
        cfg.default_atol = 1e-7
        assert cfg.default_atol == 1e-7
        with pytest.raises(ValueError):
            cfg.default_atol = 0.0

        # Test trail_length
        cfg.trail_length = 100
        assert cfg.trail_length == 100
        with pytest.raises(ValueError):
            cfg.trail_length = -1
        with pytest.raises(ValueError):
            cfg.trail_length = 200000

        # Test animation_fps
        cfg.animation_fps = 60
        assert cfg.animation_fps == 60
        with pytest.raises(ValueError):
            cfg.animation_fps = 0
        with pytest.raises(ValueError):
            cfg.animation_fps = 200

    def test_config_to_dict(self):
        """Test config to_dict method"""
        cfg = Config()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "enable_profiling" in d

    def test_config_from_dict(self):
        """Test config from_dict method"""
        cfg = Config()
        data = {"enable_profiling": True, "trail_length": 200}
        cfg.from_dict(data)
        assert cfg.enable_profiling is True
        assert cfg.trail_length == 200

        with pytest.raises(TypeError):
            cfg.from_dict("not a dict")


class TestTimeout:
    """Test timeout context manager"""

    def test_timeout_success(self):
        """Test timeout with successful operation"""
        with timeout(1.0):
            time.sleep(0.01)  # Should complete successfully

    def test_timeout_invalid_seconds(self):
        """Test timeout with invalid seconds"""
        with pytest.raises(ValueError):
            with timeout(-1.0):
                pass

        with pytest.raises(TypeError):
            with timeout("1.0"):
                pass


class TestProfileFunction:
    """Test profile_function decorator"""

    def test_profile_function_disabled(self):
        """Test profiling when disabled"""
        config.enable_profiling = False

        @profile_function
        def test_func():
            return 42

        result = test_func()
        assert result == 42

    def test_profile_function_enabled(self):
        """Test profiling when enabled"""
        config.enable_profiling = True

        @profile_function
        def test_func():
            return 42

        result = test_func()
        assert result == 42
        # Should not raise


class TestValidateFinite:
    """Test validate_finite function"""

    def test_validate_finite_success(self):
        """Test validation of finite array"""
        arr = np.array([1.0, 2.0, 3.0])
        assert validate_finite(arr) is True

    def test_validate_finite_failure(self):
        """Test validation of non-finite array"""
        arr = np.array([1.0, np.inf, 3.0])
        assert validate_finite(arr) is False

    def test_validate_finite_type_error(self):
        """Test validation with non-array"""
        with pytest.raises(TypeError):
            validate_finite([1, 2, 3])


class TestValidatePositive:
    """Test validate_positive function"""

    def test_validate_positive_success(self):
        """Test validation of positive value"""
        validate_positive(1.0)
        validate_positive(0.1)

    def test_validate_positive_failure(self):
        """Test validation of non-positive value"""
        with pytest.raises(ValueError):
            validate_positive(0.0)
        with pytest.raises(ValueError):
            validate_positive(-1.0)

    def test_validate_positive_type_error(self):
        """Test validation with non-numeric"""
        with pytest.raises(TypeError):
            validate_positive("1.0")


class TestValidateNonNegative:
    """Test validate_non_negative function"""

    def test_validate_non_negative_success(self):
        """Test validation of non-negative value"""
        validate_non_negative(0.0)
        validate_non_negative(1.0)

    def test_validate_non_negative_failure(self):
        """Test validation of negative value"""
        with pytest.raises(ValueError):
            validate_non_negative(-1.0)

    def test_validate_non_negative_type_error(self):
        """Test validation with non-numeric"""
        with pytest.raises(TypeError):
            validate_non_negative("0.0")


class TestValidateTimeSpan:
    """Test validate_time_span function"""

    def test_validate_time_span_success(self):
        """Test validation of valid time span"""
        validate_time_span((0.0, 10.0))
        validate_time_span((1.0, 2.0))

    def test_validate_time_span_type_error(self):
        """Test validation with non-tuple"""
        with pytest.raises(TypeError):
            validate_time_span([0.0, 10.0])

    def test_validate_time_span_length_error(self):
        """Test validation with wrong length"""
        with pytest.raises(ValueError):
            validate_time_span((0.0,))
        with pytest.raises(ValueError):
            validate_time_span((0.0, 10.0, 20.0))

    def test_validate_time_span_negative(self):
        """Test validation with negative time"""
        with pytest.raises(ValueError):
            validate_time_span((-1.0, 10.0))
        with pytest.raises(ValueError):
            validate_time_span((0.0, -10.0))

    def test_validate_time_span_invalid_order(self):
        """Test validation with t_start >= t_end"""
        with pytest.raises(ValueError):
            validate_time_span((10.0, 0.0))
        with pytest.raises(ValueError):
            validate_time_span((10.0, 10.0))


class TestValidateSolutionDict:
    """Test validate_solution_dict function"""

    def test_validate_solution_dict_success(self):
        """Test validation of valid solution"""
        solution = {
            "success": True,
            "t": np.array([0.0, 1.0, 2.0]),
            "y": np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]),
            "coordinates": ["x"],
        }
        validate_solution_dict(solution)  # Should not raise

    def test_validate_solution_dict_type_error(self):
        """Test validation with non-dict"""
        with pytest.raises(TypeError):
            validate_solution_dict("not a dict")

    def test_validate_solution_dict_missing_success(self):
        """Test validation with missing success key"""
        with pytest.raises(ValueError):
            validate_solution_dict({})

    def test_validate_solution_dict_invalid_success_type(self):
        """Test validation with invalid success type"""
        with pytest.raises(TypeError):
            validate_solution_dict({"success": "True"})

    def test_validate_solution_dict_missing_keys(self):
        """Test validation with missing required keys"""
        with pytest.raises(ValueError):
            validate_solution_dict({"success": True})

    def test_validate_solution_dict_invalid_t(self):
        """Test validation with invalid t array"""
        solution = {
            "success": True,
            "t": [0.0, 1.0],  # Not numpy array
            "y": np.array([[1.0, 2.0]]),
            "coordinates": ["x"],
        }
        with pytest.raises(TypeError):
            validate_solution_dict(solution)

    def test_validate_solution_dict_empty_t(self):
        """Test validation with empty t array"""
        solution = {
            "success": True,
            "t": np.array([]),
            "y": np.array([[1.0]]),
            "coordinates": ["x"],
        }
        with pytest.raises(ValueError):
            validate_solution_dict(solution)

    def test_validate_solution_dict_shape_mismatch(self):
        """Test validation with shape mismatch"""
        solution = {
            "success": True,
            "t": np.array([0.0, 1.0]),
            "y": np.array([[1.0, 2.0, 3.0]]),  # Wrong length
            "coordinates": ["x"],
        }
        with pytest.raises(ValueError):
            validate_solution_dict(solution)

    def test_validate_solution_dict_state_vector_mismatch(self):
        """Test validation with state vector size mismatch"""
        solution = {
            "success": True,
            "t": np.array([0.0, 1.0]),
            "y": np.array([[1.0, 2.0]]),  # Should be 2*len(coords) = 2
            "coordinates": ["x", "y"],  # But we have 2 coords, so need 4 rows
        }
        with pytest.raises(ValueError):
            validate_solution_dict(solution)


class TestValidateFilePath:
    """Test validate_file_path function"""

    def test_validate_file_path_success(self):
        """Test validation of valid file path"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name

        try:
            validate_file_path(temp_path, must_exist=True)
            validate_file_path("test_file.txt", must_exist=False)
        finally:
            os.unlink(temp_path)

    def test_validate_file_path_type_error(self):
        """Test validation with non-string"""
        with pytest.raises(TypeError):
            validate_file_path(123)

    def test_validate_file_path_empty(self):
        """Test validation with empty string"""
        with pytest.raises(ValueError):
            validate_file_path("")

    def test_validate_file_path_path_traversal(self):
        """Test validation with path traversal"""
        with pytest.raises(ValueError):
            validate_file_path("../test.txt")

    def test_validate_file_path_not_exist(self):
        """Test validation with non-existent file"""
        with pytest.raises(FileNotFoundError):
            validate_file_path("nonexistent_file.txt", must_exist=True)

    def test_validate_file_path_not_file(self):
        """Test validation with directory instead of file"""
        temp_dir = tempfile.mkdtemp()
        try:
            with pytest.raises(ValueError):
                validate_file_path(temp_dir, must_exist=True)
        finally:
            os.rmdir(temp_dir)
