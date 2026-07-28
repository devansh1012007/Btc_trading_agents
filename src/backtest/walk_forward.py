


class WalkForwardSplitter:
    """
    Rolling train/test window generator with purge + embargo gaps,
    so that a label whose outcome window extends into the test set
    doesn't leak information across the split boundary.
    """
    def __init__(self, train_bars: int, test_bars: int, step_bars: int,
                 embargo_bars: int):
        ...
    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        ...