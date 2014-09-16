from __future__ import division

import numpy as np


def simple_draw(num, weights, index):
    """
    Choose among indexes based on weights using a simple random draw.

    Parameters
    ----------
    num : int
        Number of items to draw from `index`.
    weights : array
        Array of weights corresponding to each value in `index`.
        Must be the same length as `index`.
    index : array
        Array of values from which to draw. Must be the same
        length as `weights`.

    Returns
    -------
    draw : array
        Array of indexes drawn based on weights.

    """
    p = weights / weights.sum()
    return np.random.choice(index, size=num, p=p, replace=True)
