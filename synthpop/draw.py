from __future__ import division

import numpy as np
import pandas as pd
from scipy.stats import chisquare

from .ipu.ipu import _FrequencyAndConstraints


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


def _draw_indexes(num, fac, weights):
    """
    Construct a set of indexes that can be used to index a complete
    set of synthetic households.

    Parameters
    ----------
    num : int
        The total number of households to draw.
    fac : _FrequencyAndConstraints
    weights : pandas.Series

    Returns
    -------
    idx : pandas.Index
        Will be drawn from the index of `weights`.

    """
    idx = []
    constraint_diffs = []

    for col_name, _, constraint, nz in fac.iter_columns():
        if len(nz) == 0:
            continue

        flr_constraint = int(np.floor(constraint))
        constraint_diffs.append((col_name, constraint - flr_constraint))

        if flr_constraint > 0:
            wts = weights.values[nz]
            idx.extend(
                simple_draw(flr_constraint, wts, weights.index.values[nz]))

    if len(idx) < num:
        num_to_add = num - len(idx)

        if num_to_add > len(weights):
            raise RuntimeError(
                'There is a mismatch between the constraints and the total '
                'number of households to draw. The total to draw appears '
                'to be higher than indicated by the constraints.')

        constraint_diffs = sorted(
            constraint_diffs, key=lambda x: x[1], reverse=True)[:num_to_add]

        for col_name, _ in constraint_diffs:
            _, _, _, nz = fac.get_column(col_name)
            wts = weights.values[nz]
            idx.extend(simple_draw(1, wts, weights.index.values[nz]))

    return pd.Index(idx)


def execute_draw(indexes, h_pums, p_pums, hh_index_start=0):
    """
    Take new household indexes and create new household and persons tables
    with updated indexes and relations.

    Parameters
    ----------
    indexes : array
        Will be used to index `h_pums` into a new table.
    h_pums : pandas.DataFrame
        Table of household data. Expected to have a "serialno" column
        that matches `p_pums`.
    p_pums : pandas.DataFrame
        Table of person data. Expected to have a "serialno" columns
        that matches `h_pums`.
    hh_index_start : int, optional
        The starting point for new indexes on the synthesized
        households table.

    Returns
    -------
    synth_hh : pandas.DataFrame
        Index will match the ``hh_id`` column in `synth_people`.
    synth_people : pandas.DataFrame
        Will be related to `synth_hh` by the ``hh_id`` column.

    """
    synth_hh = h_pums.loc[indexes].reset_index(drop=True)
    synth_hh.index += hh_index_start

    mrg_tbl = pd.DataFrame(
        {'serialno': synth_hh.serialno.values,
         'hh_id': synth_hh.index.values})
    synth_people = pd.merge(
        p_pums, mrg_tbl, left_on='serialno', right_on='serialno')

    return synth_hh, synth_people


def compare_to_constraints(synth, constraints):
    """
    Compare the results of a synthesis draw to the target constraints.

    This comparison performs chi square test between the synthesized
    category counts and the target constraints used as inputs for the IPU.

    Parameters
    ----------
    synth : pandas.Series
        Series of category IDs from synthesized table.
    constraints : pandas.Series
        Target constraints used in IPU step.

    Returns
    -------
    chisq : float
        The chi squared test statistic.
    p : float
        The p-value of the test.

    See Also
    --------
    scipy.stats.chisquare : Calculates a one-way chi square test.

    """
    counts = synth.value_counts()

    # need to add zeros to counts for any categories that are
    # in the constraints but not in the counts
    diff = constraints.index.difference(counts.index)
    counts = counts.combine_first(
        pd.Series(np.zeros(len(diff), dtype='int'), index=diff))

    counts, constraints = counts.align(constraints)

    # remove any items that are zero in the constraints
    w = constraints >= 1
    counts, constraints = counts[w], constraints[w]

    return chisquare(counts.values, constraints.values)


def draw_households(
        num, h_pums, p_pums, household_freq, household_constraints,
        person_constraints, weights, hh_index_start=0):
    """
    Draw households and persons according to weights from the IPU.

    Parameters
    ----------
    num : int
        The total number of households to draw.
    h_pums : pandas.DataFrame
        Table of household data. Expected to have a "serialno" column
        that matches `p_pums`.
    p_pums : pandas.DataFrame
        Table of person data. Expected to have a "serialno" columns
        that matches `h_pums`.
    household_freq : pandas.DataFrame
        Frequency table for household attributes. Columns should be
        a MultiIndex matching the index of `household_constraints` and
        index should be household IDs matching the index `h_pums`
        and `weights`.
    household_constraints : pandas.Series
        Target marginal constraints for household classes.
        Index must be the same as the columns of `household_freq`.
    person_constraints : pandas.Series
        Target marginal constraints for person classes.
        Index must be the same as the columns of `person_freq`.
    weights : pandas.Series
        Weights from IPU. Index should match `h_pums` and `household_freq`.
    hh_index_start : int, optional
        Index at which to start the indexing of returned households.

    Returns
    -------
    best_households : pandas.DataFrame
        Index will match the ``hh_id`` column in `synth_people`.
    best_people : pandas.DataFrame
        Will be related to `best_households` by the ``hh_id`` column.
    people_chisq : float
    people_p : float

    """
    if num == 0:
        return (
            pd.DataFrame(columns=h_pums.columns),
            pd.DataFrame(columns=p_pums.columns.append(pd.Index(['hh_id']))),
            0, 1)

    fac = _FrequencyAndConstraints(household_freq, household_constraints)

    best_chisq = np.inf

    for _ in range(20):
        indexes = _draw_indexes(num, fac, weights)
        synth_hh, synth_people = execute_draw(
            indexes, h_pums, p_pums, hh_index_start=hh_index_start)
        people_chisq, people_p = compare_to_constraints(
            synth_people.cat_id, person_constraints)

        if people_chisq < best_chisq:
            best_chisq = people_chisq
            best_p = people_p
            best_households, best_people = synth_hh, synth_people

    return best_households, best_people, best_chisq, best_p
