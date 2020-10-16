import logging
import sys
from collections import namedtuple

from functools import partial
import multiprocessing

import numpy as np
import pandas as pd
from scipy.stats import chisquare

from . import categorizer as cat
from . import draw
from .ipf.ipf import calculate_constraints
from .ipu.ipu import unit_weights

logger = logging.getLogger("synthpop")
FitQuality = namedtuple(
    'FitQuality',
    ('units_chisq', 'units_p'))
BlockGroupID = namedtuple(
    'BlockGroupID', ('state', 'county', 'tract', 'block_group'))


def enable_logging():
    handler = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def synthesize_units(units_marg, units_jd, units_pums, geography,
               marginal_zero_sub=.01, jd_zero_sub=.001, index_start=0):

    # this is the zero marginal problem
    units_marg = units_marg.replace(0, marginal_zero_sub)

    # zero cell problem
    units_jd.frequency = units_jd.frequency.replace(0, jd_zero_sub)

    # ipf for households
    logger.info("Running ipf for units")
    units_constraint, _ = calculate_constraints(units_marg, units_jd.frequency)
    units_constraint.index = units_jd.cat_id

    logger.debug("Units constraint")
    logger.debug(units_constraint)

    # make frequency tables that the ipu expects
    units_sample = units_pums.copy()
    units_sample.index.name = "unit_id"
    units_sample = units_sample.reset_index().set_index("serialno")
    units_freq = cat._frequency_table(units_sample, units_constraint.index, "unit_id")
    units_freq = units_freq.sort_index(axis=1)

    # Calculate best weights
    logger.info("Running ipu")
    import time
    t1 = time.time()
    best_weights, fit_quality, iterations = unit_weights(units_freq,
                                                        units_constraint,
                                                        geography)
    num_units = int(units_marg.groupby(level=0).sum().mean())
    print("Drawing %d units" % num_units)
    return draw.draw_units(
        num_units, units_pums, units_freq, units_constraint,
        best_weights, index_start=index_start)


def synthesize_units_all_recipe(recipe, num_geogs=None, indexes=None,
                   marginal_zero_sub=.01, jd_zero_sub=.001):
    """
    Returns
    -------
    units : pandas.DataFrame
    fit_quality : dict of FitQuality
        Keys are geographic IDs, values are namedtuples with attributes
        ``.units_chisq`` and ``units_p``

    """
    print("Synthesizing at geog level: '{}' (number of geographies is {})"
          .format(recipe.get_geography_name(), recipe.get_num_geographies()))

    if indexes is None:
        indexes = recipe.get_available_geography_ids()

    units_list = []
    cnt = 0
    fit_quality = {}
    index_start = 0

    for geog_id in indexes:
        print("Synthesizing geog id:\n", geog_id)

        units_marg = recipe.get_units_marginal_for_geography(geog_id) #TODO
        logger.debug("Units marginal")
        logger.debug(units_marg)

        units_pums, units_jd = recipe.\
            get_units_joint_dist_for_geography(geog_id) #TODO
        logger.debug("Units joint distribution")
        logger.debug(units_jd)

        units, units_chisq, units_p = \
            synthesize_units(
                units_marg, units_jd, units_pums, geog_id,
                marginal_zero_sub=marginal_zero_sub, jd_zero_sub=jd_zero_sub,
                index_start=index_start)

        # Append location identifiers to the synthesized households
        for geog_cat in geog_id.keys():
            units[geog_cat] = geog_id[geog_cat]

        units_list.append(units)
        key = BlockGroupID(
            geog_id['state'], geog_id['county'], geog_id['tract'],
            geog_id['block group'])
        fit_quality[key] = FitQuality(units_chisq, units_p)

        cnt += 1
        if len(units) > 0:
            index_start = units.index.values[-1] + 1

        if num_geogs is not None and cnt >= num_geogs:
            break

    # TODO might want to write this to disk as we go?
    all_units = pd.concat(units_list)

    return (all_units, fit_quality)


def synthesize_units_zone(units_marg, units_sample, xwalk):
    """
    Synthesize residential units a single zone (Used within multiprocessing synthesis)

    Parameters
    ----------
    units_marg : pandas.DataFrame
        processed and properly indexed residential units marginals table
    units_sample : pandas.DataFrame
        residential units sample table
    xwalk : tuple
        tuple of marginal-to-sample geography crosswalk
    Returns
    -------
    units : pandas.DataFrame
        synthesized residential units records
    stats : pandas.DataFrame
        chi-square and p-score values for marginal geography drawn
    """
    unit, units_jd = cat.joint_distribution(
            units_sample[units_sample.sample_geog == xwalk[1]],
            cat.category_combinations(units_marg.columns))
    units, units_chisq, units_p = synthesize_units(
            units_marg.loc[xwalk[0]], units_jd, unit, xwalk[0],
            index_start=1)
    units['geog'] = xwalk[0]
    stats = {'geog': xwalk[0], 'chi-square': units_chisq, 'p-score': units_p}
    return units, stats


def multiprocess_synthesize_units_zone(units_marg, units_sample, xwalk, cores=False):
    """
    Synthesize units for a set of marginal geographies via multiprocessing

    Parameters
    ----------
    units_marg : pandas.DataFrame
        processed and properly indexed residential units marginals table
    units_sample : pandas.DataFrame
        residential units sample table
    xwalk : tuple
        tuple of marginal-to-sample geography crosswalk
    cores : integer, optional
        number of cores to use in the multiprocessing pool. defaults to
        multiprocessing.cpu_count() - 1
    Returns
    -------
    all_units : pandas.DataFrame
        synthesized residential units records
    all_stats : pandas.DataFrame
        chi-square and p-score values for each marginal geography drawn
    """
    cores = cores if cores else (multiprocessing.cpu_count()-1)
    part = partial(synthesize_units_zone, units_marg, units_sample)
    p = multiprocessing.Pool(cores)
    results = p.map(part, list(xwalk))
    p.close()
    p.join()

    units_list = [result[0] for result in results]
    all_stats = pd.DataFrame([result[1] for result in results])
    all_units = pd.concat(units_list)
    return all_units, all_stats