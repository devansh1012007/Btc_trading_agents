"""
Walk-forward train/test split generator with purge + embargo gaps.
"""
from typing import Iterator, Optional, Tuple

import numpy as np


class WalkForwardSplitter:
    """
    Rolling train/test window generator with purge + embargo gaps, so that
    a label whose outcome window extends into the test set doesn't leak
    information across the split boundary.
    """

    def __init__(
        self, train_bars: int, test_bars: int, step_bars: int, embargo_bars: int,
        label_horizon_bars: int = 0,
    ):
        if train_bars <= 0 or test_bars <= 0 or step_bars <= 0:
            raise ValueError("train_bars, test_bars, and step_bars must be positive")
        if embargo_bars < 0:
            raise ValueError("embargo_bars must be non-negative")
        if label_horizon_bars < 0:
            raise ValueError("label_horizon_bars must be non-negative")
        if label_horizon_bars > embargo_bars:
            raise ValueError(
                f"embargo_bars={embargo_bars} is shorter than the label horizon "
                f"({label_horizon_bars} bars: labels look that far ahead). The last training "
                f"labels would be decided by bars inside the test window -- raise "
                f"backtest.embargo_bars to at least {label_horizon_bars}."
            )
        self.label_horizon_bars = label_horizon_bars

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

    def split_by_bar_index(
        self, bar_index: np.ndarray, label_end: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Walk-forward windows measured in BARS over sparse samples.

        ``split()`` walks consecutive row numbers, which is only a time
        window when there is exactly one row per bar. Training sets here
        have one row per CANDIDATE (many bars have none), so a "5000-bar
        train window" in row units is an unknown amount of time. This
        yields row indices for windows defined on each sample's
        ``bar_index``:

            train [t, t + train_bars)   embargo   test [t + train_bars + embargo, + test_bars)

        ``label_end`` (``bar_index + bars_held`` per row) lets the splitter
        PURGE any training row whose label was still being decided at the
        first test bar; with ``embargo_bars >= label_horizon_bars`` that
        never fires, it is the safety net for a hand-built horizon.
        """
        bars = np.asarray(bar_index, dtype="int64")
        if bars.size == 0:
            return
        ends = np.asarray(label_end, dtype="int64") if label_end is not None else None
        if ends is not None and ends.shape != bars.shape:
            raise ValueError("label_end must align with bar_index")

        last_bar = int(bars.max()) + 1
        start = int(bars.min())
        while True:
            train_end = start + self.train_bars
            test_start = train_end + self.embargo_bars
            test_end = test_start + self.test_bars
            if test_end > last_bar:
                break

            train_mask = (bars >= start) & (bars < train_end)
            if ends is not None:
                train_mask &= ends < test_start
            test_mask = (bars >= test_start) & (bars < test_end)
            if train_mask.any() and test_mask.any():
                yield np.flatnonzero(train_mask), np.flatnonzero(test_mask)
            start += self.step_bars

    def n_splits_by_bar_index(self, bar_index: np.ndarray, label_end: Optional[np.ndarray] = None) -> int:
        return sum(1 for _ in self.split_by_bar_index(bar_index, label_end))
