import contextlib
import numpy as np
import random

@contextlib.contextmanager
def set_seed(seed):
    """
    Context manager that temporarily sets numpy and Python random seeds
    and restores the previous state afterward.

    Args:
        seed: Integer seed to set. If None, no seed is set.

    Yields:
        None
    """
    if seed is None:
        yield
        return

    # Save the current random states
    np_state = np.random.get_state()
    py_state = random.getstate()

    try:
        # Set the new seeds
        np.random.seed(seed)
        random.seed(seed)
        yield
    finally:
        # Restore the previous random states
        np.random.set_state(np_state)
        random.setstate(py_state)
