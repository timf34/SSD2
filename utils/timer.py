import time

class Timer:
    def __init__(self):
        self.start = 0

    def begin(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start
