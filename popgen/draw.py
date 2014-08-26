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

    """
    if weights is None:
        weights = pd.Series(np.ones(len(frame)), index=frame.index)

    weights = weights / weights.sum()
    idx = np.random.choice(weights.index, size=num, p=weights.values)

    draw = frame.loc[idx]
    draw.index = range(len(draw))

    return draw
