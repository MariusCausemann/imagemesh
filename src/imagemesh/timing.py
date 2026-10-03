"""Timing helpers."""

import time
from contextlib import contextmanager
from functools import wraps


@contextmanager
def timer(name):
    """Print the wall time of the with-block."""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"[{name}] finished in {end - start:.4f} seconds")


def time_func(func):
    """Decorator printing the wall time of each call."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with timer(func.__name__):
            return func(*args, **kwargs)

    return wrapper
