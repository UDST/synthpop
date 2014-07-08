# encoding: utf-8

from __future__ import division


def calculate_fit_quality(column, weights, constraint):
    """
    Calculate quality of fit metric for a column of the frequency table.
    (The 𝛿 parameter described in the IPU paper.)

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


def update_weights(column, weights, constraint):
    """
    Update household weights based on a single column.

    The update will be applied to all weights, so make sure only the
    non-zero elements of `column` and the corresponding weights are given.

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
    new_weights : pandas.Series

    """
    adj = constraint / (column * weights).sum()
    return weights * adj
