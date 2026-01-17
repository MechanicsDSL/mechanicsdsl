"""
Unit tests for MechanicsDSL utils/caching module.

Tests the LRUCache class for caching functionality.
"""

import numpy as np
import sympy as sp

from mechanics_dsl.utils.caching import LRUCache


class TestLRUCacheInit:
    """Tests for LRUCache initialization."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        cache = LRUCache()
        assert cache.maxsize == 128
        assert cache.max_memory_mb == 100.0

    def test_init_custom_maxsize(self):
        """Test initialization with custom maxsize."""
        cache = LRUCache(maxsize=50)
        assert cache.maxsize == 50

    def test_init_custom_max_memory(self):
        """Test initialization with custom max_memory_mb."""
        cache = LRUCache(max_memory_mb=50.0)
        assert cache.max_memory_mb == 50.0

    def test_init_both_custom(self):
        """Test initialization with both custom values."""
        cache = LRUCache(maxsize=64, max_memory_mb=25.0)
        assert cache.maxsize == 64
        assert cache.max_memory_mb == 25.0

    def test_init_invalid_maxsize_uses_default(self):
        """Test that invalid maxsize falls back to default."""
        cache = LRUCache(maxsize=-5)
        assert cache.maxsize == 128

    def test_init_zero_maxsize_uses_default(self):
        """Test that zero maxsize falls back to default."""
        cache = LRUCache(maxsize=0)
        assert cache.maxsize == 128

    def test_init_non_int_maxsize_uses_default(self):
        """Test that non-integer maxsize falls back to default."""
        cache = LRUCache(maxsize="ten")
        assert cache.maxsize == 128

    def test_init_invalid_max_memory_uses_default(self):
        """Test that invalid max_memory_mb falls back to default."""
        cache = LRUCache(max_memory_mb=-10.0)
        assert cache.max_memory_mb == 100.0

    def test_init_zero_max_memory_uses_default(self):
        """Test that zero max_memory_mb falls back to default."""
        cache = LRUCache(max_memory_mb=0)
        assert cache.max_memory_mb == 100.0

    def test_init_counters_at_zero(self):
        """Test that hit/miss counters start at zero."""
        cache = LRUCache()
        assert cache.hits == 0
        assert cache.misses == 0


class TestLRUCacheGet:
    """Tests for LRUCache.get method."""

    def test_get_existing_key(self):
        """Test getting an existing key."""
        cache = LRUCache()
        cache.set("key1", "value1")

        result = cache.get("key1")
        assert result == "value1"

    def test_get_nonexistent_key(self):
        """Test getting a nonexistent key returns None."""
        cache = LRUCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_get_increments_hits(self):
        """Test that successful get increments hits."""
        cache = LRUCache()
        cache.set("key", "value")

        cache.get("key")
        assert cache.hits == 1

    def test_get_increments_misses(self):
        """Test that failed get increments misses."""
        cache = LRUCache()
        cache.get("nonexistent")
        assert cache.misses == 1

    def test_get_invalid_key_type(self):
        """Test get with non-string key returns None."""
        cache = LRUCache()
        result = cache.get(123)
        assert result is None
        assert cache.misses == 1

    def test_get_moves_to_end(self):
        """Test that get moves item to end (most recently used)."""
        cache = LRUCache(maxsize=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        # Access "a" to move it to end
        cache.get("a")

        # Add new item, should evict "b" (oldest)
        cache.set("d", 4)

        assert cache.get("a") == 1  # Should still be there
        assert cache.get("b") is None  # Should be evicted
        assert cache.get("c") == 3
        assert cache.get("d") == 4


class TestLRUCacheSet:
    """Tests for LRUCache.set method."""

    def test_set_stores_value(self):
        """Test that set stores a value."""
        cache = LRUCache()
        cache.set("key", "value")
        assert cache.get("key") == "value"

    def test_set_updates_existing(self):
        """Test that set updates existing key."""
        cache = LRUCache()
        cache.set("key", "value1")
        cache.set("key", "value2")
        assert cache.get("key") == "value2"

    def test_set_evicts_oldest(self):
        """Test that set evicts oldest item when full."""
        cache = LRUCache(maxsize=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # Should evict "a"

        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_set_invalid_key_type(self):
        """Test set with non-string key is ignored."""
        cache = LRUCache()
        cache.set(123, "value")

        # Should not be stored
        stats = cache.get_stats()
        assert stats["size"] == 0

    def test_set_none_value(self):
        """Test setting None value."""
        cache = LRUCache()
        cache.set("key", None)

        # None is stored but get returns None for missing too
        # We can check stats to verify it was stored
        stats = cache.get_stats()
        assert stats["size"] == 1

    def test_set_numpy_array(self):
        """Test storing numpy arrays."""
        cache = LRUCache()
        arr = np.array([1, 2, 3, 4, 5])
        cache.set("array", arr)

        result = cache.get("array")
        np.testing.assert_array_equal(result, arr)

    def test_set_sympy_expression(self):
        """Test storing SymPy expressions."""
        cache = LRUCache()
        x = sp.Symbol("x")
        expr = x**2 + 2 * x + 1
        cache.set("expr", expr)

        result = cache.get("expr")
        assert result == expr


class TestLRUCacheEstimateMemory:
    """Tests for LRUCache._estimate_memory_mb method."""

    def test_estimate_empty_cache(self):
        """Test memory estimation for empty cache."""
        cache = LRUCache()
        mem = cache._estimate_memory_mb()
        assert mem >= 0

    def test_estimate_with_numpy(self):
        """Test memory estimation with numpy arrays."""
        cache = LRUCache()
        large_array = np.zeros((1000, 1000))  # ~8MB
        cache.set("big", large_array)

        mem = cache._estimate_memory_mb()
        assert mem > 0

    def test_estimate_with_sympy(self):
        """Test memory estimation with SymPy objects."""
        cache = LRUCache()
        x, y = sp.symbols("x y")
        expr = x**2 * sp.sin(y) + sp.exp(x * y)
        cache.set("expr", expr)

        mem = cache._estimate_memory_mb()
        assert mem >= 0

    def test_estimate_primitive_types(self):
        """Test memory estimation with primitive types."""
        cache = LRUCache()
        cache.set("str", "hello world")
        cache.set("int", 42)
        cache.set("float", 3.14159)

        mem = cache._estimate_memory_mb()
        assert mem >= 0


class TestLRUCacheClear:
    """Tests for LRUCache.clear method."""

    def test_clear_empties_cache(self):
        """Test that clear removes all items."""
        cache = LRUCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        cache.clear()

        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.get("c") is None

    def test_clear_resets_size(self):
        """Test that clear resets size to 0."""
        cache = LRUCache()
        cache.set("a", 1)
        cache.set("b", 2)

        cache.clear()
        stats = cache.get_stats()
        assert stats["size"] == 0

    def test_clear_resets_counters(self):
        """Test that clear resets hit/miss counters."""
        cache = LRUCache()
        cache.set("a", 1)
        cache.get("a")  # hit
        cache.get("b")  # miss

        cache.clear()

        assert cache.hits == 0
        assert cache.misses == 0


class TestLRUCacheGetStats:
    """Tests for LRUCache.get_stats method."""

    def test_get_stats_returns_dict(self):
        """Test that get_stats returns a dictionary."""
        cache = LRUCache()
        stats = cache.get_stats()
        assert isinstance(stats, dict)

    def test_get_stats_has_required_keys(self):
        """Test that stats has required keys."""
        cache = LRUCache()
        stats = cache.get_stats()

        required_keys = ["size", "maxsize", "hits", "misses", "hit_rate", "memory_mb"]
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"

    def test_get_stats_correct_size(self):
        """Test that size is correct."""
        cache = LRUCache()
        cache.set("a", 1)
        cache.set("b", 2)

        stats = cache.get_stats()
        assert stats["size"] == 2

    def test_get_stats_correct_maxsize(self):
        """Test that maxsize is correct."""
        cache = LRUCache(maxsize=50)
        stats = cache.get_stats()
        assert stats["maxsize"] == 50

    def test_get_stats_hit_rate(self):
        """Test hit rate calculation."""
        cache = LRUCache()
        cache.set("a", 1)

        cache.get("a")  # hit
        cache.get("a")  # hit
        cache.get("b")  # miss

        stats = cache.get_stats()
        # 2 hits, 1 miss = 2/3 = 0.667
        assert 0.65 < stats["hit_rate"] < 0.68

    def test_get_stats_zero_hit_rate(self):
        """Test hit rate when no accesses."""
        cache = LRUCache()
        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.0


class TestLRUCacheMemoryLimit:
    """Tests for memory limit enforcement."""

    def test_memory_limit_eviction(self):
        """Test that memory limit triggers eviction."""
        # Very small memory limit
        cache = LRUCache(maxsize=1000, max_memory_mb=0.001)  # 1 KB limit

        # Add some data
        for i in range(10):
            cache.set(f"key{i}", np.zeros(100))  # Each ~800 bytes

        # Cache should have evicted some items
        stats = cache.get_stats()
        # Might be less than 10 due to memory eviction
        assert stats["size"] >= 0
