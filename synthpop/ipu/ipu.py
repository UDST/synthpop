# encoding: utf-8

from __future__ import division

import itertools

import numpy as np
import pandas as pd


def _drop_zeros(df):
    """
    Drop zeros from a DataFrame, returning an iterator over the columns
    in the DataFrame.

    Yields tuples of (column name, non-zero column values, non-zero indexes).

    Parameters
    ----------
    df : pandas.DataFrame

    """
    for col_idx, col in df.iteritems():
        nz = col.nonzero()[0]
        yield (col_idx, col.values[nz], nz)


class _FrequencyAndConstraints(object):
    """
    Wrap frequency tables and constraints for both household and
    person classes for easy iteration over all of them.

    Also tracks the locations of non-zero elements in each column
    of the frequency tables.

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
        self._everything = tuple(itertools.chain(
            ((col, household_constraints[key], nz)
             for key, col, nz in _drop_zeros(household_freq)),
            ((col, person_constraints[key], nz)
             for key, col, nz in _drop_zeros(person_freq))))

        self.ncols = len(self._everything)

    def iter_columns(self):
        """
        Iterate over columns of both household and frequency tables AND
        the corresponding constraints for each column AND non-zero indexes
        applicable to each column.
        Yields tuples of (column, constraint, nonzero). The returned column
        contains only the non-zero elements.

        """
        return iter(self._everything)


def _fit_quality(column, weights, constraint):
    """
    Calculate quality of fit metric for a column of the frequency table.
    (The 𝛿 parameter described in the IPU paper.)

    Parameters
    ----------
    column : 1D array
        Non-zero elements of a column of a frequency table.
        Must have the same length as `weights`.
    weights : 1D array
        Weights corresponding to the household rows in `column`.
        Must have the same length as `column`.
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
    weights : ndarray
        Array of weights for all households.

    """
    return sum(
        _fit_quality(col, weights[nz], constraint)
        for col, constraint, nz in freq_wrap.iter_columns()
        ) / freq_wrap.ncols


def _update_weights(column, weights, constraint):
    """
    Update household weights based on a single column.

    The update will be applied to all weights, so make sure only the
    non-zero elements of `column` and the corresponding weights are given.

    Parameters
    ----------
    column : 1D array
        Non-zero elements of a column of a frequency table.
        Must have the same length as `weights`.
    weights : 1D array
        Weights corresponding to the household rows in `column`.
        Must have the same length as `column`.
    constraint : float
        Target marginal constraint for this column.

    Returns
    -------
    new_weights : ndarray

    """
    adj = constraint / (column * weights).sum()
    return weights * adj


def household_weights(
        household_freq, person_freq, household_constraints, person_constraints,
        convergence=1e-4, max_iterations=20000):
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
    max_iterations, int, optional
        Maximum number of iterations to do before stopping and raising
        an exception.

    Returns
    -------
    weights : pandas.Series
    fit_qual : float
        The final average fit quality metric.
    iterations : int
        Number of iterations made.

    """
    weights = np.ones(len(household_freq), dtype='float')
    best_weights = weights.copy()

    freq_wrap = _FrequencyAndConstraints(
        household_freq, person_freq, household_constraints, person_constraints)

    fit_qual = _average_fit_quality(freq_wrap, weights)
    best_fit_qual = fit_qual
    fit_change = np.inf
    iterations = 0

    while fit_change > convergence:
        for col, constraint, nz in freq_wrap.iter_columns():
            weights[nz] = _update_weights(col, weights[nz], constraint)

        new_fit_qual = _average_fit_quality(freq_wrap, weights)
        fit_change = abs(new_fit_qual - fit_qual)

        if new_fit_qual < fit_qual:
            best_fit_qual = new_fit_qual
            best_weights = weights.copy()

        fit_qual = new_fit_qual
        iterations += 1

        if iterations > max_iterations:
            raise RuntimeError(
                'Maximum number of iterations reached during IPU: {}'.format(
                    max_iterations))

    return (
        pd.Series(best_weights, index=household_freq.index),
        best_fit_qual, iterations)
