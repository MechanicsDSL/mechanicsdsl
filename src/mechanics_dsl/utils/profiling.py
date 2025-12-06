"""
Profiling and performance monitoring for MechanicsDSL
"""
import time
import platform
import signal
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps

from .logging import logger
from .config import config

try:
    import psutil
except ImportError:
    psutil = None  # Optional dependency


class TimeoutError(Exception):
    """Raised when an operation times out"""
    pass


class PerformanceMonitor:
    """Advanced performance monitoring with memory and timing tracking"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.memory_snapshots: List[Dict[str, float]] = []
        self.start_times: Dict[str, float] = {}
        
    def start_timer(self, name: str) -> None:
        """Start timing an operation with validation"""
        if not isinstance(name, str) or not name:
            logger.warning(f"PerformanceMonitor.start_timer: invalid name '{name}', using 'unnamed'")
            name = 'unnamed'
        if name in self.start_times:
            logger.warning(f"PerformanceMonitor.start_timer: timer '{name}' already running, overwriting")
        self.start_times[name] = time.perf_counter()
        
    def stop_timer(self, name: str) -> float:
        """Stop timing and record duration with validation"""
        if not isinstance(name, str) or not name:
            logger.warning(f"PerformanceMonitor.stop_timer: invalid name '{name}'")
            return 0.0
        if name not in self.start_times:
            logger.warning(f"PerformanceMonitor.stop_timer: timer '{name}' was not started")
            return 0.0
        try:
            duration = time.perf_counter() - self.start_times[name]
            if duration < 0:
                logger.warning(f"PerformanceMonitor.stop_timer: negative duration for '{name}', clock issue?")
                duration = 0.0
            if duration > 86400:  # More than 24 hours seems wrong
                logger.warning(f"PerformanceMonitor.stop_timer: suspiciously long duration {duration}s for '{name}'")
            self.metrics[name].append(duration)
            del self.start_times[name]
            return duration
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"PerformanceMonitor.stop_timer: error stopping timer '{name}': {e}")
            return 0.0
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB"""
        if psutil is None:
            return {'rss': 0.0, 'vms': 0.0, 'percent': 0.0}
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            return {
                'rss': mem_info.rss / 1024 / 1024,  # Resident Set Size
                'vms': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
                'percent': process.memory_percent()
            }
        except (AttributeError, Exception):
            return {'rss': 0.0, 'vms': 0.0, 'percent': 0.0}
    
    def snapshot_memory(self, label: str = "") -> None:
        """Take a memory snapshot"""
        mem = self.get_memory_usage()
        mem['label'] = label
        mem['timestamp'] = time.time()
        self.memory_snapshots.append(mem)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric with validation"""
        if not isinstance(name, str) or not name:
            logger.warning(f"PerformanceMonitor.get_stats: invalid name '{name}'")
            return {}
        if name not in self.metrics or not self.metrics[name]:
            return {}
        try:
            values = self.metrics[name]
            if not values:
                return {}
            # Filter out invalid values
            valid_values = [v for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
            if not valid_values:
                logger.warning(f"PerformanceMonitor.get_stats: no valid values for '{name}'")
                return {}
            return {
                'count': len(valid_values),
                'total': sum(valid_values),
                'mean': float(np.mean(valid_values)),
                'std': float(np.std(valid_values)),
                'min': float(np.min(valid_values)),
                'max': float(np.max(valid_values))
            }
        except Exception as e:
            logger.error(f"PerformanceMonitor.get_stats: error computing stats for '{name}': {e}")
            return {}
    
    def reset(self) -> None:
        """Reset all metrics"""
        self.metrics.clear()
        self.memory_snapshots.clear()
        self.start_times.clear()


# Global performance monitor
_perf_monitor = PerformanceMonitor()


@contextmanager
def timeout(seconds: float):
    """
    Cross-platform timeout context manager for timing out operations.
    
    Uses signal.SIGALRM on Unix systems and threading.Timer on Windows.
    Note: Threading-based timeout on Windows cannot interrupt CPU-bound operations.
    
    Args:
        seconds: Maximum time allowed (must be positive)
        
    Raises:
        TimeoutError: If operation exceeds time limit
        ValueError: If seconds is not positive
    """
    if not isinstance(seconds, (int, float)):
        raise TypeError(f"seconds must be numeric, got {type(seconds).__name__}")
    if seconds <= 0:
        raise ValueError(f"seconds must be positive, got {seconds}")
    
    if platform.system() == 'Windows':
        # Windows: Use threading.Timer (cannot interrupt CPU-bound operations)
        timer: Optional[threading.Timer] = None
        timeout_occurred = threading.Event()
        
        def timeout_handler() -> None:
            timeout_occurred.set()
            raise TimeoutError(f"Operation timed out after {seconds} seconds")
        
        timer = threading.Timer(seconds, timeout_handler)
        timer.daemon = True
        timer.start()
        
        try:
            yield
        finally:
            if timer is not None:
                timer.cancel()
                timer.join(timeout=0.1)
    else:
        # Unix: Use signal.SIGALRM (can interrupt operations)
        def timeout_handler(signum: int, frame: Any) -> None:
            raise TimeoutError(f"Operation timed out after {seconds} seconds")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(seconds))
        
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def profile_function(func: Callable) -> Callable:
    """Decorator to profile function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if config.enable_profiling:
            import cProfile
            import pstats
            from io import StringIO
            
            profiler = cProfile.Profile()
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
            
            s = StringIO()
            stats = pstats.Stats(profiler, stream=s)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions
            
            logger.debug(f"\n{'='*70}\nProfile for {func.__name__}:\n{s.getvalue()}\n{'='*70}")
            return result
        else:
            return func(*args, **kwargs)
    return wrapper
