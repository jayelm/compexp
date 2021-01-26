import numpy as np


def onehot(arr, minlength=None):
    """
    Expands an array of integers in one-hot encoding by adding a new last
    dimension, leaving zeros everywhere except for the nth dimension, where
    the original array contained the integer n.  The minlength parameter is
    used to indcate the minimum size of the new dimension.
    """
    length = np.amax(arr) + 1
    if minlength is not None:
        length = max(minlength, length)
    result = np.zeros(arr.shape + (length,), dtype=np.float32)
    result[tuple(list(np.indices(arr.shape)) + [arr])] = 1
    return result
