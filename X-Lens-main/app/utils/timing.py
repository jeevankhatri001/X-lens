from contextlib import contextmanager
from time import perf_counter

@contextmanager
def timer(target: dict, key: str):
    start = perf_counter()
    try: yield
    finally: target[key] = round((perf_counter() - start) * 1000, 3)
