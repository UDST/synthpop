# encoding: utf-8

from __future__ import division


def fit_quality(column, weights, constraint):
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


def average_fit_quality(frequency_table, weights, constraints):
    """
    Parameters
    ----------
    frequency_table : FrequencyTable
    weights : pandas.Series
        Weights for each household. Index must match `frequency_table`.
    constraints : pandas.Series
        Target marginal constraints. Must be indexable in the same
        way as `frequency_table`.

    """
    return sum(
        fit_quality(col, weights, constraints[hp][name])
        for hp, name, col in frequency_table.itercols()
        ) / frequency_table.ncols


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
