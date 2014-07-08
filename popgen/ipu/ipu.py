from __future__ import division


def calculate_fit_quality(column, weights, constraint):
    """
    Calculate quality of fit metric for a column of the frequency table.

    Parameters
    ----------
    column : pandas.Series
        Column from frequency table.
    weights : pandas.Series
        Weights for each household. Index must match `column`.
    constraint : float
        Target marginal constraint for this column.

    Returns
    -------
    quality : float

    """
    return abs((column * weights).sum() - constraint) / constraint
