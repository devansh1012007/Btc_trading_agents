"""
Walk-forward train/test split generator with purge + embargo gaps.

Both the zone-survival model (labels look forward up to
zone_label_horizon_bars) and the regime classifier (trained on smoothed,
whole-sequence HMM decodes) produce autocorrelated, forward-looking labels.
A naive chronological split leaks information across the train/test
boundary unless a gap ("embargo") separates the two windows, so that no
label in the train set has an outcome window overlapping the test set.
"""
from typing import Iterator, Tuple
import numpy as np


class WalkForwardSplitter:
    """
    Rolling train/test window generator with purge + embargo gaps, so that
    a label whose outcome window extends into the test set doesn't leak
    information across the split boundary.
    """

    def __init__(
        self,
        train_bars: int,
        test_bars: int,
        step_bars: int,
        embargo_bars: int,
    ):
        if train_bars <= 0 or test_bars <= 0 or step_bars <= 0:
            raise ValueError("train_bars, test_bars, and step_bars must be positive")
        if embargo_bars < 0:
            raise ValueError("embargo_bars must be non-negative")

        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars
        self.embargo_bars = embargo_bars

    def split(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield (train_idx, test_idx) index arrays, walking forward through
        n_samples. Each fold's test window starts embargo_bars after its
        train window ends; the next fold's train window starts
        step_bars after the current one's.
        """
        start = 0
        while True:
            train_start = start
            train_end = train_start + self.train_bars  # exclusive
            test_start = train_end + self.embargo_bars
            test_end = test_start + self.test_bars  # exclusive

            if test_end > n_samples:
                break

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx

            start += self.step_bars

    def n_splits(self, n_samples: int) -> int:
        """Count how many folds split() would yield, without materializing them."""
        return sum(1 for _ in self.split(n_samples))
