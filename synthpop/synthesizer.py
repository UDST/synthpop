import logging
import sys

import numpy as np
import pandas as pd
from scipy.stats import chisquare

from . import categorizer as cat
from . import draw
from .ipf.ipf import calculate_constraints
from .ipu.ipu import household_weights

logger = logging.getLogger("synthpop")


def enable_logging():
    handler = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def execute_draw(indexes, h_pums, p_pums):
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

    Returns
    -------
    synth_hh : pandas.DataFrame
        Index will match the ``hh_id`` column in `synth_people`.
    synth_people : pandas.DataFrame
        Will be related to `synth_hh` by the ``hh_id`` column.

    """
    synth_hh = h_pums.loc[indexes].reset_index(drop=True)

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
    diff = constraints.index.diff(counts.index)
    counts = counts.combine_first(
        pd.Series(np.zeros(len(diff), dtype='int'), index=diff))

    counts, constraints = counts.align(constraints)

    return chisquare(counts.values, constraints.values)


def synthesize(h_marg, p_marg, h_jd, p_jd, h_pums, p_pums,
               marginal_zero_sub=.01, jd_zero_sub=.001):

    # this is the zero marginal problem
    h_marg = h_marg.replace(0, marginal_zero_sub)
    p_marg = p_marg.replace(0, marginal_zero_sub)

    # zero cell problem
    h_jd.frequency = h_jd.frequency.replace(0, jd_zero_sub)
    p_jd.frequency = p_jd.frequency.replace(0, jd_zero_sub)

    # ipf for households
    logger.info("Running ipf for households")
    h_constraint, _ = calculate_constraints(h_marg, h_jd.frequency)
    h_constraint.index = h_jd.cat_id

    logger.debug("Household constraint")
    logger.debug(h_constraint)

    # ipf for persons
    logger.info("Running ipf for persons")
    p_constraint, _ = calculate_constraints(p_marg, p_jd.frequency)
    p_constraint.index = p_jd.cat_id

    logger.debug("Person constraint")
    logger.debug(p_constraint)

    # make frequency tables that the ipu expects
    household_freq, person_freq = cat.frequency_tables(p_pums, h_pums,
                                                       p_jd.cat_id,
                                                       h_jd.cat_id)

    # do the ipu to match person marginals
    logger.info("Running ipu")
    import time
    t1 = time.time()
    best_weights, fit_quality, iterations = household_weights(household_freq,
                                                              person_freq,
                                                              h_constraint,
                                                              p_constraint)
    logger.info("Time to run ipu: %.3fs" % (time.time()-t1))

    logger.debug("IPU weights:")
    logger.debug(best_weights.describe())
    logger.debug("Fit quality:")
    logger.debug(fit_quality)
    logger.debug("Number of iterations:")
    logger.debug(iterations)

    num_households = int(h_marg.groupby(level=0).sum().mean())
    print "Drawing %d households" % num_households

    indexes = draw.simple_draw(
        num_households, best_weights.values, best_weights.index.values)

    best_chisq = np.inf

    for _ in range(20):
        synth_households, synth_people = execute_draw(indexes, h_pums, p_pums)
        household_chisq, household_p = compare_to_constraints(
            synth_households.cat_id, h_constraint)
        people_chisq, people_p = compare_to_constraints(
            synth_people.cat_id, p_constraint)

        if household_chisq + people_chisq < best_chisq:
            best_chisq = household_chisq + people_chisq
            best_hh_chisq, best_people_chisq = household_chisq, people_chisq
            best_hh_p, best_people_p = household_p, people_p
            best_households, best_people = synth_households, synth_people

    return (
        best_households, best_people, best_hh_chisq, best_hh_p,
        best_people_chisq, best_people_p)


def synthesize_all(recipe, num_geogs=None, indexes=None,
                   marginal_zero_sub=.01, jd_zero_sub=.001):

    print "Synthesizing at geog level: '{}' (number of geographies is {})".\
        format(recipe.get_geography_name(), recipe.get_num_geographies())

    if indexes is None:
        indexes = recipe.get_available_geography_ids()

    hhs = []
    cnt = 0
    # TODO will parallelization work here?
    for geog_id in indexes:
        print "Synthesizing geog id:\n", geog_id

        h_marg = recipe.get_household_marginal_for_geography(geog_id)
        logger.debug("Household marginal")
        logger.debug(h_marg)

        p_marg = recipe.get_person_marginal_for_geography(geog_id)
        logger.debug("Person marginal")
        logger.debug(p_marg)

        h_pums, h_jd = recipe.\
            get_household_joint_dist_for_geography(geog_id)
        logger.debug("Household joint distribution")
        logger.debug(h_jd)

        p_pums, p_jd = recipe.get_person_joint_dist_for_geography(geog_id)
        logger.debug("Person joint distribution")
        logger.debug(p_jd)

        households, people, hh_chisq, hh_p, people_chisq, people_p = \
            synthesize(
                h_marg, p_marg, h_jd, p_jd, h_pums, p_pums,
                marginal_zero_sub=marginal_zero_sub, jd_zero_sub=jd_zero_sub)
        hhs.append(households)

        cnt += 1
        if num_geogs is not None and cnt >= num_geogs:
            break

    # TODO might want to write this to disk as we go?
    return pd.concat(hhs, verify_integrity=True, ignore_index=True)
