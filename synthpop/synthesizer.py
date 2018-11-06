import logging
import sys
from collections import namedtuple

import numpy as np
import pandas as pd
from datetime import datetime as dt
from time import sleep, time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, TimeoutError
from itertools import repeat

from . import categorizer as cat
from . import draw
from .ipf.ipf import calculate_constraints
from .ipu.ipu import household_weights

logger = logging.getLogger("synthpop")
FitQuality = namedtuple(
    'FitQuality',
    ('people_chisq', 'people_p'))
BlockGroupID = namedtuple(
    'BlockGroupID', ('state', 'county', 'tract', 'block_group'))


def enable_logging():
    handler = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def synthesize(h_marg, p_marg, h_jd, p_jd, h_pums, p_pums,
               marginal_zero_sub=.01, jd_zero_sub=.001, hh_index_start=0):

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
    # p_constraint.index = p_jd.cat_id

    logger.debug("Person constraint")
    logger.debug(p_constraint)

    # modify person cat ids so they are unique when combined with households
    p_starting_cat_id = h_jd['cat_id'].max() + 1
    p_jd['cat_id'] += p_starting_cat_id
    p_pums['cat_id'] += p_starting_cat_id
    p_constraint.index = p_jd.cat_id

    # make frequency tables that the ipu expects
    household_freq, person_freq = cat.frequency_tables(p_pums, h_pums,
                                                       p_jd.cat_id,
                                                       h_jd.cat_id)

    # do the ipu to match person marginals
    logger.info("Running ipu")
    import time
    t1 = time.time()
    max_iterations = 20000
    best_weights, fit_quality, iterations = household_weights(
        household_freq, person_freq, h_constraint, p_constraint,
        max_iterations=max_iterations)

    logger.info("Time to run ipu: %.3fs" % (time.time() - t1))
    logger.debug("IPU weights:")
    logger.debug(best_weights.describe())
    logger.debug("Fit quality: {0}".format(fit_quality))
    if iterations == 20000:
        logger.warn("Number of iterations: {0}".format(str(iterations)))
    else:
        logger.debug("Number of iterations: {0}".format(str(iterations)))
    num_households = int(h_marg.groupby(level=0).sum().mean())

    # print("Drawing %d households" % num_households)

    return draw.draw_households(
        num_households, h_pums, p_pums, household_freq, h_constraint,
        p_constraint, best_weights, hh_index_start=hh_index_start)


def synthesize_all(recipe, num_geogs=None, indexes=None,
                   marginal_zero_sub=.01, jd_zero_sub=.001):
    """
    Returns
    -------
    households, people : pandas.DataFrame
    fit_quality : dict of FitQuality
        Keys are geographic IDs, values are namedtuples with attributes
        ``.household_chisq``, ``household_p``, ``people_chisq``,
        and ``people_p``.

    """
    if indexes is None:
        indexes = recipe.get_available_geography_ids()

    hh_list = []
    people_list = []
    cnt = 0
    fit_quality = {}
    hh_index_start = 0

    for geog_id in tqdm(indexes, total=num_geogs):
        # print("Synthesizing geog id:\n", geog_id)

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

        households, people, people_chisq, people_p = \
            synthesize(
                h_marg, p_marg, h_jd, p_jd, h_pums, p_pums,
                marginal_zero_sub=marginal_zero_sub, jd_zero_sub=jd_zero_sub,
                hh_index_start=hh_index_start)

        # Append location identifiers to the synthesized households
        for geog_cat in geog_id.keys():
            households[geog_cat] = geog_id[geog_cat]

        hh_list.append(households)
        people_list.append(people)
        key = BlockGroupID(
            geog_id['state'], geog_id['county'], geog_id['tract'],
            geog_id['block group'])
        fit_quality[key] = FitQuality(people_chisq, people_p)

        cnt += 1
        if len(households) > 0:
            hh_index_start = households.index.values[-1] + 1

        if num_geogs is not None and cnt >= num_geogs:
            break

    # TODO might want to write this to disk as we go?
    all_households = pd.concat(hh_list)
    all_persons = pd.concat(people_list, ignore_index=True)

    return (all_households, all_persons, fit_quality)


def geog_preprocessing(geog_id, recipe, marginal_zero_sub, jd_zero_sub):
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

    return h_marg, p_marg, h_jd, p_jd, h_pums, p_pums, marginal_zero_sub,\
        jd_zero_sub


def synth_worker(geog_id, recipe, marginal_zero_sub, jd_zero_sub):
    # geog_id, recipe, marginal_zero_sub, jd_zero_sub = arg_tuple

    synth_args = geog_preprocessing(
        geog_id, recipe, marginal_zero_sub, jd_zero_sub)
    households, people, people_chisq, people_p = synthesize(*synth_args)

    for geog_cat in geog_id.keys():
        households[geog_cat] = geog_id[geog_cat]

    key = BlockGroupID(
        geog_id['state'], geog_id['county'], geog_id['tract'],
        geog_id['block group'])

    return households, people, key, people_chisq, people_p


def synthesize_all_in_parallel(
        recipe, num_geogs=None, indexes=None, marginal_zero_sub=.01,
        jd_zero_sub=.001, max_workers=None):
    """
    Returns
    -------
    households, people : pandas.DataFrame
    fit_quality : dict of FitQuality
        Keys are geographic IDs, values are namedtuples with attributes
        ``.household_chisq``, ``household_p``, ``people_chisq``,
        and ``people_p``.

    """
    # cluster = LocalCluster()
    # client = Client(cluster)
    with ProcessPoolExecutor(max_workers=5) as ex:

        if indexes is None:
            indexes = recipe.get_available_geography_ids()

        hh_list = []
        people_list = []
        cnt = 0
        fit_quality = {}
        hh_index_start = 0
        geog_synth_args = []
        finished_args = []
        geog_ids = []
        futures = []

        print('Submitting function args for parallel processing:')
        for i, geog_id in enumerate(indexes):
            geog_synth_args.append(ex.submit(
                geog_preprocessing, geog_id, recipe, marginal_zero_sub,
                jd_zero_sub))
            geog_ids.append(geog_id)
            cnt += 1
            if num_geogs is not None and cnt >= num_geogs:
                break

        print('Processing function args in parallel:')
        for finished_arg in tqdm(
                as_completed(geog_synth_args), total=len(geog_synth_args)):
            finished_args.append(finished_arg.result())

        print('Submitting {0} geographies for parallel processing.'.format(
            len(finished_args)))
        futures = [
            ex.submit(synthesize, *geog_args) for geog_args in finished_args]

        print('Beginning population synthesis in parallel:')
        for f in tqdm(as_completed(futures), total=len(futures)):
            pass

        print('Processing results:')
        for i, future in tqdm(enumerate(futures), total=len(futures)):
            try:
                households, people, people_chisq, people_p = future.result()
            except Exception as e:
                print('Generated an exception: {0}'.format(e))
            else:
                geog_id = geog_ids[i]

                # Append location identifiers to the synthesized households
                for geog_cat in geog_id.keys():
                    households[geog_cat] = geog_id[geog_cat]

                # update the household_ids since we can't do it in the call to
                # synthesize when we execute in parallel
                households.index += hh_index_start
                people.hh_id += hh_index_start

                hh_list.append(households)
                people_list.append(people)
                key = BlockGroupID(
                    geog_id['state'], geog_id['county'], geog_id['tract'],
                    geog_id['block group'])
                fit_quality[key] = FitQuality(people_chisq, people_p)

                if len(households) > 0:
                    hh_index_start = households.index.values[-1] + 1

        all_households = pd.concat(hh_list)
        all_persons = pd.concat(people_list, ignore_index=True)

        return (all_households, all_persons, fit_quality)


def synthesize_all_in_parallel_mp(
        recipe, num_geogs=None, indexes=None, marginal_zero_sub=.01,
        jd_zero_sub=.001, max_workers=None):
    """
    Returns
    -------
    households, people : pandas.DataFrame
    fit_quality : dict of FitQuality
        Keys are geographic IDs, values are namedtuples with attributes
        ``.household_chisq``, ``household_p``, ``people_chisq``,
        and ``people_p``.

    """
    if indexes is None:
        indexes = recipe.get_available_geography_ids()

    hh_list = []
    people_list = []
    cnt = 0
    fit_quality = {}
    hh_index_start = 0
    geog_ids = []

    for i, geog_id in enumerate(indexes):
        geog_ids.append(geog_id)
        cnt += 1
        if num_geogs is not None and cnt >= num_geogs:
            break

    with Pool(processes=max_workers) as pool:
        print('{0} - Generating function args for parallel processing.'.format(
            str(dt.now())))
        geog_synth_args = pool.starmap(
            geog_preprocessing, zip(
                geog_ids, repeat(recipe), repeat(marginal_zero_sub),
                repeat(jd_zero_sub)))
        pool.close()
        pool.join()
        print('{0} - Finished.'.format(str(dt.now())))

    with Pool(processes=max_workers) as pool:
        print(
            '{0} - Submitting funtion args to synthesizers '
            'in parallel.'.format(str(dt.now())))
        results = pool.starmap(synthesize, geog_synth_args)
        pool.close()
        print('{0} - Waiting for parallel synthesizer to finish.'.format(
            str(dt.now())))
        pool.join()
        print('{0} - Finished synthesizing geographies in parallel.'.format(
            str(dt.now())))

    print('Processing results:')
    for i, result in tqdm(enumerate(results), total=len(results)):
        households, people, people_chisq, people_p = result
        geog_id = geog_ids[i]

        # Append location identifiers to the synthesized households
        for geog_cat in geog_id.keys():
            households[geog_cat] = geog_id[geog_cat]

        # update the household_ids since we can't do it in the call to
        # synthesize when we execute in parallel
        households.index += hh_index_start
        people.hh_id += hh_index_start

        hh_list.append(households)
        people_list.append(people)
        key = BlockGroupID(
            geog_id['state'], geog_id['county'], geog_id['tract'],
            geog_id['block group'])
        fit_quality[key] = FitQuality(people_chisq, people_p)

        if len(households) > 0:
            hh_index_start = households.index.values[-1] + 1

    all_households = pd.concat(hh_list)
    all_persons = pd.concat(people_list, ignore_index=True)

    return (all_households, all_persons, fit_quality)


def synthesize_all_in_parallel_full(
        recipe, num_geogs=None, indexes=None, marginal_zero_sub=.01,
        jd_zero_sub=.001, max_workers=None):
    """
    Returns
    -------
    households, people : pandas.DataFrame
    fit_quality : dict of FitQuality
        Keys are geographic IDs, values are namedtuples with attributes
        ``.household_chisq``, ``household_p``, ``people_chisq``,
        and ``people_p``.

    """
    if indexes is None:
        indexes = recipe.get_available_geography_ids()

    cnt = 0
    geog_ids = []

    for i, geog_id in enumerate(indexes):
        geog_ids.append(geog_id)
        cnt += 1
        if num_geogs is not None and cnt >= num_geogs:
            break

    results = []
    timeouts = []
    print('Synthesizing geographies in parallel.')
    pool = Pool()
    # arg_tuple = ((geog_id, recipe, marginal_zero_sub, jd_zero_sub))
    procs = [pool.apply_async(
        synth_worker, (geog_id, recipe, marginal_zero_sub, jd_zero_sub)
    ) for geog_id in geog_ids]
    print('Initialized all processes. Now recovering results:')
    # for proc in tqdm(procs, total=len(procs)):
    for proc in procs:
        try:
            result = proc.get(120)
            results.append(result)
            print('{0} results completed'.format(str(len(results))))
        except TimeoutError:
            timeouts.append(geog_id)
            print('{0} timeouts'.format(str(len(timeouts))))

    # arg_tuples = zip(
    #     geog_ids, repeat(recipe), repeat(marginal_zero_sub),
    #     repeat(jd_zero_sub))
    # for result in tqdm(
    #         pool.imap_unordered(
    #             synth_worker, arg_tuples),
    #         total=len(geog_ids), ncols=80):
    #     results.append(result)
    print('Shutting down the worker pool.')
    pool.close()
    pool.join()
    print('Pool closed.')

    return results, timeouts
    hh_index_start = 0
    hh_list = []
    people_list = []
    fit_quality = {}
    print('Processing results:')
    for i, result in tqdm(enumerate(results), total=len(results)):
        households, people, key, people_chisq, people_p = result

        # update the household_ids since we can't do it in the call to
        # synthesize when we execute in parallel
        households.index += hh_index_start
        people.hh_id += hh_index_start

        hh_list.append(households)
        people_list.append(people)
        fit_quality[key] = FitQuality(people_chisq, people_p)

        if len(households) > 0:
            hh_index_start = households.index.values[-1] + 1

    all_households = pd.concat(hh_list)
    all_persons = pd.concat(people_list, ignore_index=True)

    return (all_households, all_persons, fit_quality)
