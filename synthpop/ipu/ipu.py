# encoding: utf-8

from __future__ import division

import itertools
from collections import OrderedDict
import warnings

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
    def for_each_col(col):
        nz = col.values.nonzero()[0]
        return col.iloc[nz], nz

    for (col_idx, (col, nz)) in df.apply(for_each_col, axis=0, raw=False).items():
        yield (col_idx, col.values, nz)


class _FrequencyAndConstraints(object):
    """
    Wrap frequency tables and constraints for both household and
    person classes for easy iteration over all of them.

    Also tracks the locations of non-zero elements in each column
    of the frequency tables. If including person classes, both
    `person_freq` and `person_constraints` are required.

    Parameters
    ----------
    household_freq : pandas.DataFrame
        Frequency table for household attributes. Columns should be
        a MultiIndex matching the index of `household_constraints` and
        index should be household IDs matching the index of
        `person_freq`.
    household_constraints : pandas.Series
        Target marginal constraints for household classes.
        Index must be the same as the columns of `household_freq`.
    person_freq : pandas.DataFrame, optional
        Frequency table for household person. Columns should be
        a MultiIndex matching the index of `person_constraints` and
        index should be household IDs matching the index of
        `household_freq`.
    person_constraints : pandas.Series, optional
        Target marginal constraints for person classes.
        Index must be the same as the columns of `person_freq`.

    Attributes
    ----------
    ncols : int
        Total number household_wof columns across household and person classes.

    """

    def __init__(self, household_freq, household_constraints, person_freq=None,
                 person_constraints=None):

        hh_cols = ((key, col, household_constraints[key], nz)
                   for key, col, nz in _drop_zeros(household_freq))

        has_pers = person_freq is not None and person_constraints is not None
        if has_pers:
            p_cols = ((key, col, person_constraints[key], nz)
                      for key, col, nz in _drop_zeros(person_freq))
        else:
            p_cols = []

        self._everything = OrderedDict(
            (t[0], t) for t in itertools.chain(hh_cols, p_cols))
        self.ncols = len(self._everything)

        """
        Check for problems in the resulting keys.
        These typically arise when column names are shared accross
        households and persons.
        """
        keys = set([c[0] for c in self.iter_columns()])
        assert len(set(household_freq.columns) - keys) == 0
        if has_pers:
            assert len(set(person_freq.columns) - keys) == 0
            assert self.ncols == len(household_freq.columns) + len(person_freq.columns)

    def iter_columns(self):
        """
        Iterate over columns of both household and frequency tables AND
        the corresponding constraints for each column AND non-zero indexes
        applicable to each column.
        Yields tuples of (column name, column, constraint, nonzero).
        The returned column contains only the non-zero elements.

        """
        return list(self._everything.values())

    def get_column(self, key):
        """
        Return a specific column's info by its name.

        Parameters
        ----------
        key : object
            Column name or tuple required to index a MultiIndex column.

        Returns
        -------
        col_name : object
            Same as `key`.
        column : pandas.Series
            Has only the non-zero elements.
        constraint : float
            The target constraint for this type.
        nonzero : array
            The location of the non-zero items in the column.

        """
        return self._everything[key]


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
        for _, col, constraint, nz in freq_wrap.iter_columns()
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
    adj = constraint / float((column * weights).sum())
    return weights * adj


def household_weights(
        household_freq, person_freq, household_constraints,
        person_constraints, geography, ignore_max_iters,
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
        household_freq, household_constraints, person_freq, person_constraints)

    fit_qual = _average_fit_quality(freq_wrap, weights)
    best_fit_qual = fit_qual
    fit_change = np.inf
    iterations = 0

    while fit_change > convergence:
        for _, col, constraint, nz in freq_wrap.iter_columns():
            weights[nz] = _update_weights(col, weights[nz], constraint)

        new_fit_qual = _average_fit_quality(freq_wrap, weights)
        fit_change = abs(new_fit_qual - fit_qual)

        if new_fit_qual < fit_qual:
            best_fit_qual = new_fit_qual
            best_weights = weights.copy()

        fit_qual = new_fit_qual
        iterations += 1

        if iterations > max_iterations:
            if ignore_max_iters:
                fitting_tolerance = fit_change - convergence
                print('Fitting tolerance before 20000 iterations: %s' % str(fitting_tolerance))
                ipu_dict = {'best_fit_qual': best_fit_qual,
                            'fit_change': fit_change,
                            'fitting_tolerance': fitting_tolerance,
                            'geog_id': geography}
                if isinstance(geography, pd.Series):
                    state, county = geography['state'], geography['county']
                    tract, bgroup = geography['tract'], geography['block group']
                    np.save('max_iter_{}_{}_{}_{}.npy'.format(state, county,
                                                              tract, bgroup), ipu_dict)
                elif isinstance(geography, list):
                    np.save('max_iter_{}_{}.npy'.format(geography[0], geography[1]), ipu_dict)
                else:
                    np.save('max_iter_{}.npy'.format(str(geography)), ipu_dict)

                warnings.warn(
                    'Maximum number of iterations reached '
                    'during IPU: {}'.format(max_iterations), UserWarning)
                return (
                    pd.Series(best_weights, index=household_freq.index),
                    best_fit_qual, iterations)
            else:
                raise RuntimeError(
                    'Maximum number of iterations reached '
                    'during IPU: {}'.format(max_iterations))

    return (
        pd.Series(best_weights, index=household_freq.index),
        best_fit_qual, iterations)
