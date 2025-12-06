"""
LRU Cache implementation for MechanicsDSL
"""
import sys
import numpy as np
import sympy as sp
from typing import Dict, Any, Optional
from collections import OrderedDict

from .logging import logger


class LRUCache:
    """Advanced LRU cache with size limits and memory awareness"""
    
    def __init__(self, maxsize: int = 128, max_memory_mb: float = 100.0):
        """Initialize LRU cache with validation"""
        if not isinstance(maxsize, int) or maxsize < 1:
            logger.warning(f"LRUCache: invalid maxsize {maxsize}, using 128")
            maxsize = 128
        if not isinstance(max_memory_mb, (int, float)) or max_memory_mb <= 0:
            logger.warning(f"LRUCache: invalid max_memory_mb {max_memory_mb}, using 100.0")
            max_memory_mb = 100.0
        self.cache: OrderedDict = OrderedDict()
        self.maxsize = maxsize
        self.max_memory_mb = float(max_memory_mb)
        self.hits = 0
        self.misses = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with validation"""
        if not isinstance(key, str):
            logger.warning(f"LRUCache.get: invalid key type {type(key).__name__}, expected str")
            self.misses += 1
            return None
        try:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
        except (KeyError, TypeError, AttributeError) as e:
            logger.error(f"LRUCache.get: error accessing key '{key}': {e}")
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache with eviction if needed and validation"""
        if not isinstance(key, str):
            logger.warning(f"LRUCache.set: invalid key type {type(key).__name__}, expected str")
            return
        if value is None:
            logger.debug(f"LRUCache.set: storing None value for key '{key}'")
        try:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.maxsize:
                    # Remove least recently used
                    try:
                        self.cache.popitem(last=False)
                    except KeyError:
                        pass  # Cache was empty
            self.cache[key] = value
        except (TypeError, AttributeError, MemoryError) as e:
            logger.error(f"LRUCache.set: error setting key '{key}': {e}")
            # Try to free space
            try:
                while len(self.cache) > self.maxsize * 0.5:
                    self.cache.popitem(last=False)
            except Exception:
                pass
        
        # Check memory usage
        try:
            current_mem = self._estimate_memory_mb()
            if current_mem > self.max_memory_mb:
                # Evict oldest items until under limit
                while current_mem > self.max_memory_mb * 0.8 and self.cache:
                    self.cache.popitem(last=False)
                    current_mem = self._estimate_memory_mb()
        except Exception:
            pass  # Memory estimation failed, continue
    
    def _estimate_memory_mb(self) -> float:
        """Estimate cache memory usage"""
        try:
            total = 0
            for value in self.cache.values():
                if isinstance(value, np.ndarray):
                    total += value.nbytes
                elif isinstance(value, (sp.Expr, sp.Matrix)):
                    # Rough estimate for SymPy objects
                    total += sys.getsizeof(str(value))
                else:
                    total += sys.getsizeof(value)
            return total / 1024 / 1024
        except Exception:
            return 0.0
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            'size': len(self.cache),
            'maxsize': self.maxsize,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'memory_mb': self._estimate_memory_mb()
        }
