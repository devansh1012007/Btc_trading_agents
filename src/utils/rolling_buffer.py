"""
Rolling bar buffer: shared utility for maintaining a fixed-size rolling
window of bar data across the perception layers (FeatureEngine,
DynamicStructureEngine, RegimeDetector, BreakoutExpert).
"""
from typing import Dict, List, Optional
import numpy as np


class RollingBarBuffer:
    """Fixed-capacity rolling buffer of bar dicts, FIFO eviction."""

    def __init__(self, maxlen: int):
        if maxlen <= 0:
            raise ValueError("maxlen must be positive")
        self.maxlen = maxlen
        self._buffer: List[Dict] = []

    def append(self, bar: Dict) -> None:
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
        return list(self._buffer)

    def field_array(self, key: str) -> np.ndarray:
        return np.array([b[key] for b in self._buffer])

    def latest(self) -> Optional[Dict]:
        return self._buffer[-1] if self._buffer else None

    def clear(self) -> None:
        self._buffer.clear()

    def is_full(self) -> bool:
        return len(self._buffer) >= self.maxlen
