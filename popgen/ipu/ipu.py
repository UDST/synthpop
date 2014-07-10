# encoding: utf-8

from __future__ import division

import numpy as np
import pandas as pd


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


def household_weights(frequency_table, constraints, convergence=1e-4):
    """
    Calculate the household weights that best match household and
    person level attributes.

    Parameters
    ----------
    frequency_table : FrequencyTable
    constraints : pandas.Series
        Target marginal constraints. Must be indexable in the same
        way as `frequency_table`.
    convergence : float, optional
        When the average fit quality metric changes by less than this value
        after an iteration we declare done and send back the weights
        from the best fit.

    Returns
    -------
    weights : pandas.Series
    fit_qual : float
        The final average fit quality metric.
    iterations : int
        Number of iterations made.

    """
    weights = pd.Series(
        np.ones(len(frequency_table)),
        index=frequency_table.index, dtype=np.float)
    best_weights = weights
    fit_qual = average_fit_quality(frequency_table, weights, constraints)
    fit_change = np.inf
    iterations = 0

    while fit_change > convergence:
        for hp, name, col in frequency_table.itercols():
            weights.loc[col.index] = update_weights(
                col, weights.loc[col.index], constraints[hp][name])

        new_fit_qual = average_fit_quality(
            frequency_table, weights, constraints)
        fit_change = abs(new_fit_qual - fit_qual)

        if new_fit_qual < fit_qual:
            fit_qual = new_fit_qual
            best_weights = weights

        iterations += 1

    return best_weights, fit_qual, iterations
