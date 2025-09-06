import time
from functools import wraps

from slm_lab.lib import logger
from slm_lab.lib.profiler import get_profiler

logger = logger.get_logger(__name__)


def profile(func):
    """Non-invasive profiler: CPU, RAM, GPU, VRAM, timing. Activate with --profile flag."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if profiler := get_profiler():
            with profiler.profile_function(func.__name__):
                return func(*args, **kwargs)
        return func(*args, **kwargs)

    return wrapper


def lab_api(func):
    """SLM-Lab API decorator for core algorithm methods (act, update, save, etc.)."""
    return profile(func)


def timeit(fn):
    """Simple timing decorator for debug logging."""

    @wraps(fn)
    def time_fn(*args, **kwargs):
        start = time.time()
        output = fn(*args, **kwargs)
        end = time.time()
        logger.debug(f"Timed: {fn.__name__} {round((end - start) * 1000, 4)}ms")
        return output

    return time_fn
