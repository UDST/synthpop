import numpy as np
import pandas as pd


def draw(frame, num, weights=None):
    """
    Draw rows from a DataFrame according to some weights.

    Parameters
    ----------
    frame : pandas.DataFrame
        Table from which to draw.
    num : int
        Number of rows to return.
    weights : pandas.Series, optional
        Weight of each row to use when drawing.
        Should have the same index as `households`.
        If not given each row has equal weight.

    Returns
    -------
    draws : pandas.DataFrame
        New DataFrame of rows drawn. Index is reset to range(len(draws)).
    drawn_indexes : ndarray
        The array of index values used to create `draws`.

    """
    if weights is None:
        weights = pd.Series(np.ones(len(frame)), index=frame.index)

    weights = weights / weights.sum()
    idx = np.random.choice(weights.index, size=num, p=weights.values)

    draws = frame.loc[idx]
    draws.index = range(len(draws))

    return draws, idx
