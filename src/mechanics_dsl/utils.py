import logging
import time
import numpy as np
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import platform
import threading
import signal
from typing import Any, Dict, List, Optional, Tuple, Union

# Configuration Constants
DEFAULT_TRAIL_LENGTH = 150
DEFAULT_FPS = 30
SIMPLIFICATION_TIMEOUT = 5.0
MAX_PARSER_ERRORS = 10

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def setup_logging(level: int = logging.INFO):
    logger = logging.getLogger('MechanicsDSL')
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    return logger

logger = setup_logging()

class Config:
    def __init__(self):
        self.enable_profiling = False
        self.simplification_timeout = SIMPLIFICATION_TIMEOUT
        self.max_parser_errors = MAX_PARSER_ERRORS
        self.default_rtol = 1e-6
        self.default_atol = 1e-8
        self.enable_adaptive_solver = True
        self.cache_max_size = 256
        self.animation_fps = DEFAULT_FPS
        self.trail_length = DEFAULT_TRAIL_LENGTH

config = Config()

class LRUCache:
    def __init__(self, maxsize: int = 128):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)
        self.cache[key] = value
        
    def clear(self):
        self.cache.clear()

class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds: float):
    if platform.system() == 'Windows':
        timer = threading.Timer(seconds, lambda: (_ for _ in ()).throw(TimeoutError))
        timer.start()
        try:
            yield
        finally:
            timer.cancel()
    else:
        def handler(signum, frame):
            raise TimeoutError()
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(int(seconds))
        try:
            yield
        finally:
            signal.alarm(0)

def validate_array_safe(arr: Any, name: str = "array", min_size: int = 0) -> bool:
    if arr is None: return False
    if not isinstance(arr, np.ndarray): return False
    if arr.size < min_size: return False
    return True

def safe_float_conversion(value: Any) -> float:
    try:
        return float(value)
    except:
        return 0.0
