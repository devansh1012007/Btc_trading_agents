"""
Rolling bar buffer: shared utility for maintaining a fixed-size rolling
window of bar data across the perception layers (FeatureEngine,
DynamicStructureEngine, RegimeDetector).

This consolidates the append/pop-to-lookback pattern that was previously
duplicated in feature_engine.py and structure_engine.py (Phase 2 review,
finding #7) -- all three perception-layer buffers now share one
implementation, so they can't drift out of sync as independently managed
lists.
"""
from typing import Dict, List, Optional
import numpy as np


class RollingBarBuffer:
    """
    Fixed-capacity rolling buffer of bar dicts, FIFO eviction.
    """

    def __init__(self, maxlen: int):
        if maxlen <= 0:
            raise ValueError("maxlen must be positive")
        self.maxlen = maxlen
        self._buffer: List[Dict] = []

    def append(self, bar: Dict) -> None:
        """Add a bar dict to the buffer; evict the oldest if over capacity."""
        self._buffer.append(bar)
        if len(self._buffer) > self.maxlen:
            self._buffer.pop(0)

    def __len__(self) -> int:
        return len(self._buffer)

    def __getitem__(self, idx):
        return self._buffer[idx]

    def __iter__(self):
        return iter(self._buffer)

    def __bool__(self) -> bool:
        return len(self._buffer) > 0

    def as_list(self) -> List[Dict]:
        """Return a shallow copy of the buffer contents, oldest first."""
        return list(self._buffer)

    def field_array(self, key: str) -> np.ndarray:
        """Extract one field across all bars as a numpy array, insertion order."""
        return np.array([b[key] for b in self._buffer])

    def latest(self) -> Optional[Dict]:
        """Return the most recently appended bar, or None if empty."""
        return self._buffer[-1] if self._buffer else None

    def clear(self) -> None:
        self._buffer.clear()

    def is_full(self) -> bool:
        return len(self._buffer) >= self.maxlen
