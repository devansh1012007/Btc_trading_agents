"""
Walk-forward train/test split generator with purge + embargo gaps.
"""
from typing import Iterator, Tuple
import numpy as np


class WalkForwardSplitter:
    """
    Rolling train/test window generator with purge + embargo gaps, so that
    a label whose outcome window extends into the test set doesn't leak
    information across the split boundary.
    """

    def __init__(self, train_bars: int, test_bars: int, step_bars: int, embargo_bars: int):
        if train_bars <= 0 or test_bars <= 0 or step_bars <= 0:
            raise ValueError("train_bars, test_bars, and step_bars must be positive")
        if embargo_bars < 0:
            raise ValueError("embargo_bars must be non-negative")

        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars
        self.embargo_bars = embargo_bars

    def split(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        start = 0
        while True:
            train_start = start
            train_end = train_start + self.train_bars
            test_start = train_end + self.embargo_bars
            test_end = test_start + self.test_bars

            if test_end > n_samples:
                break

            yield np.arange(train_start, train_end), np.arange(test_start, test_end)
            start += self.step_bars

    def n_splits(self, n_samples: int) -> int:
        return sum(1 for _ in self.split(n_samples))
