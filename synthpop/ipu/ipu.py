# encoding: utf-8

from __future__ import division

import numpy as np
import pandas as pd


def _drop_zeros(df):
    """
    Drop zeros from a DataFrame, returning a dictionary that has the same
    keys as the DataFrame columns and values that are Series objects
    of the nonzero elements.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    nonzero_dict : dict
        Will be indexable just like `df`.

    """
    # this is a bit awkward you can't iloc an empty list apparently
    return {
        col_idx: col.iloc[col.nonzero()[0]] if len(col.nonzero()[0]) else
        pd.Series() for col_idx, col in df.iteritems()}


class _FrequencyAndConstraints(object):
    """
    Wrap frequency tables and constraints for both household and person
    classes for easy iteration over all of them.

    Parameters
    ----------
    household_freq : pandas.DataFrame
        Frequency table for household attributes. Columns should be
        a MultiIndex matching the index of `household_constraints` and
        index should be household IDs matching the index of
        `person_freq`.
    person_Freq : pandas.DataFrame
        Frequency table for household person. Columns should be
        a MultiIndex matching the index of `person_constraints` and
        index should be household IDs matching the index of
        `household_freq`.
    household_constraints : pandas.Series
        Target marginal constraints for household classes.
        Index must be the same as the columns of `household_freq`.
    person_constraints : pandas.Series
        Target marginal constraints for person classes.
        Index must be the same as the columns of `person_freq`.

    Attributes
    ----------
    ncols : int
        Total number of columns across household and person classes.

    """
    def __init__(self, household_freq, person_freq, household_constraints,
                 person_constraints):
        self.household_freq = _drop_zeros(household_freq)
        self.person_freq = _drop_zeros(person_freq)
        self.household_constraints = household_constraints
        self.person_constraints = person_constraints

        self.ncols = len(household_constraints) + len(person_constraints)

    def iter_columns(self):
        """
        Iterate over columns of both household and frequency tables AND
        the corresponding constraints for each column.
        Yields tuples of (column, constraint). The returned column
        contains only the non-zero elements.

        """
        for col_idx in self.household_freq:
            yield (
                self.household_freq[col_idx],
                self.household_constraints[col_idx])

        for col_idx in self.person_freq:
            yield self.person_freq[col_idx], self.person_constraints[col_idx]


def _fit_quality(column, weights, constraint):
    """
    Calculate quality of fit metric for a column of the frequency table.
    (The 𝛿 parameter described in the IPU paper.)

    Parameters
    ----------
    column : pandas.Series
        Column from frequency table. Index must match `weights`.
    weights : pandas.Series
        Weights for each household. Index must match `column`.
    constraint : float
        Target marginal constraint for this column.

    Returns
    -------
    quality : float

    """
    return abs((column * weights).sum() - constraint) / constraint


def _average_fit_quality(freq_wrap, weights):
    """
    Parameters
    ----------
    freq_wrap : `_FrequencyAndConstraints`
    weights : pandas.Series
        Weights for each household, keyed by household ID like
        the frequency tables.

    """
    return sum(
        _fit_quality(col, weights[col.index], constraint)
        for col, constraint in freq_wrap.iter_columns()
        ) / freq_wrap.ncols


def _update_weights(column, weights, constraint):
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


def household_weights(
        household_freq, person_freq, household_constraints, person_constraints,
        convergence=1e-4):
    """
    Calculate the household weights that best match household and
    person level attributes.

    Parameters
    ----------
    household_freq : pandas.DataFrame
        Frequency table for household attributes. Columns should be
        a MultiIndex matching the index of `household_constraints` and
        index should be household IDs matching the index of
        `person_freq`.
    person_Freq : pandas.DataFrame
        Frequency table for household person. Columns should be
        a MultiIndex matching the index of `person_constraints` and
        index should be household IDs matching the index of
        `household_freq`.
    household_constraints : pandas.Series
        Target marginal constraints for household classes.
        Index must be the same as the columns of `household_freq`.
    person_constraints : pandas.Series
        Target marginal constraints for person classes.
        Index must be the same as the columns of `person_freq`.
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
        np.ones(len(household_freq)),
        index=household_freq.index, dtype=np.float)
    best_weights = weights.copy()

    freq_wrap = _FrequencyAndConstraints(
        household_freq, person_freq, household_constraints, person_constraints)

    fit_qual = _average_fit_quality(freq_wrap, weights)
    fit_change = np.inf
    iterations = 0

    while fit_change > convergence:
        for col, constraint in freq_wrap.iter_columns():
            weights[col.index] = _update_weights(
                col, weights[col.index], constraint)

        new_fit_qual = _average_fit_quality(freq_wrap, weights)
        fit_change = abs(new_fit_qual - fit_qual)

        if new_fit_qual < fit_qual:
            fit_qual = new_fit_qual
            best_weights = weights.copy()

        iterations += 1

    return best_weights, fit_qual, iterations
