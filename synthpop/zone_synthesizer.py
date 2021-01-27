from functools import partial
import multiprocessing

import pandas as pd

from .synthesizer import synthesize, enable_logging
from . import categorizer as cat


def load_data(hh_marginal_file, person_marginal_file,
              hh_sample_file, person_sample_file):
    """
    Load and process data inputs from .csv files on disk

    Parameters
    ----------
    hh_marginal_file : string
        path to a csv file of household marginals
    person_marginal_file : string
        path to a csv file of person marginals
    hh_sample_file : string
        path to a csv file of sample household records to be drawn from
    person_sample_file : string
        path to a csv file of sample person records
    Returns
    -------
    hh_marg : pandas.DataFrame
        processed and properly indexed household marginals table
    p_marg : pandas.DataFrame
        processed and properly indexed person marginals table
    hh_sample : pandas.DataFrame
        household sample table
    p_sample : pandas.DataFrame
        person sample table
    xwalk : list of tuples
        list of marginal-to-sample geography crosswalks to iterate over
    """
    hh_sample = pd.read_csv(hh_sample_file)
    p_sample = pd.read_csv(person_sample_file)

    hh_marg = pd.read_csv(hh_marginal_file, header=[0, 1], index_col=0)
    hh_marg.columns.levels[0].set_names('cat_name', inplace=True)
    hh_marg.columns.levels[1].set_names('cat_values', inplace=True)

    xwalk = list(zip(hh_marg.index, hh_marg.sample_geog.unstack().values))
    hh_marg = hh_marg.drop('sample_geog', axis=1, level=0)

    p_marg = pd.read_csv(person_marginal_file, header=[0, 1], index_col=0)
    p_marg.columns.levels[0].set_names('cat_name', inplace=True)
    p_marg.columns.levels[1].set_names('cat_values', inplace=True)

    return hh_marg, p_marg, hh_sample, p_sample, xwalk


def synthesize_all_zones(hh_marg, p_marg, hh_sample, p_sample, xwalk):
    """
    Iterate over a geography crosswalk list and synthesize in-line

    Parameters
    ----------
    hh_marg : pandas.DataFrame
        processed and properly indexed household marginals table
    p_marg : pandas.DataFrame
        processed and properly indexed person marginals table
    hh_sample : pandas.DataFrame
        household sample table
    p_sample : pandas.DataFrame
        person sample table
    xwalk : list of tuples
        list of marginal-to-sample geography crosswalks to iterate over
    Returns
    -------
    all_households : pandas.DataFrame
        synthesized household records
    all_persons : pandas.DataFrame
        synthesized person records
    all_stats : pandas.DataFrame
        chi-square and p-score values for each marginal geography drawn
    """
    hh_list = []
    people_list = []
    stats_list = []
    hh_index_start = 1
    for geogs in xwalk:
        households, people, stats = synthesize_zone(hh_marg, p_marg,
                                                    hh_sample, p_sample, geogs)
        stats_list.append(stats)
        hh_list.append(households)
        people_list.append(people)

        if len(households) > 0:
            hh_index_start = households.index.values[-1] + 1
    all_households = pd.concat(hh_list)
    all_persons = pd.concat(people_list)
    all_households, all_persons = synch_hhids(all_households, all_persons)
    all_stats = pd.DataFrame(stats_list)
    return all_households, all_persons, all_stats


def synch_hhids(households, persons):
    """
    Synchronize household ids with corresponding person records

    Parameters
    ----------
    households : pandas.DataFrame
        full households table with id values sequential by geog
    persons : pandas.DataFrame
        full persons table with id values sequential by geog
    Returns
    -------
    households : pandas.DataFrame
        households table with reindexed sequential household ids
    persons : pandas.DataFrame
        persons table synchronized with updated household ids
    """
    households['hh_id'] = households.index
    households['household_id'] = range(1, len(households.index)+1)
    persons = pd.merge(
            persons, households[['household_id', 'geog', 'hh_id']],
            how='left', left_on=['geog', 'hh_id'], right_on=['geog', 'hh_id'],
            suffixes=('', '_x')).drop('hh_id', axis=1)
    households.set_index('household_id', inplace=True)
    households.drop('hh_id', axis=1, inplace=True)
    return households, persons


def synthesize_zone(hh_marg, p_marg, hh_sample, p_sample, xwalk):
    """
    Synthesize a single zone (Used within multiprocessing synthesis)

    Parameters
    ----------
    hh_marg : pandas.DataFrame
        processed and properly indexed household marginals table
    p_marg : pandas.DataFrame
        processed and properly indexed person marginals table
    hh_sample : pandas.DataFrame
        household sample table
    p_sample : pandas.DataFrame
        person sample table
    xwalk : tuple
        tuple of marginal-to-sample geography crosswalk
    Returns
    -------
    households : pandas.DataFrame
        synthesized household records
    people : pandas.DataFrame
        synthesized person records
    stats : pandas.DataFrame
        chi-square and p-score values for marginal geography drawn
    """
    hhs, hh_jd = cat.joint_distribution(
            hh_sample[hh_sample.sample_geog == xwalk[1]],
            cat.category_combinations(hh_marg.columns))
    ps, p_jd = cat.joint_distribution(
            p_sample[p_sample.sample_geog == xwalk[1]],
            cat.category_combinations(p_marg.columns))
    households, people, people_chisq, people_p = synthesize(
            hh_marg.loc[xwalk[0]], p_marg.loc[xwalk[0]], hh_jd, p_jd, hhs, ps, xwalk[0],
            ignore_max_iters=False, hh_index_start=1)
    households['geog'] = xwalk[0]
    people['geog'] = xwalk[0]
    stats = {'geog': xwalk[0], 'chi-square': people_chisq, 'p-score': people_p}
    return households, people, stats


def multiprocess_synthesize(hh_marg, p_marg, hh_sample,
                            p_sample, xwalk, cores=False):
    """
    Synthesize for a set of marginal geographies via multiprocessing

    Parameters
    ----------
    hh_marg : pandas.DataFrame
        processed and properly indexed household marginals table
    p_marg : pandas.DataFrame
        processed and properly indexed person marginals table
    hh_sample : pandas.DataFrame
        household sample table
    p_sample : pandas.DataFrame
        person sample table
    xwalk : list of tuples
        list of marginal-to-sample geography crosswalks to iterate over
    cores : integer, optional
        number of cores to use in the multiprocessing pool. defaults to
        multiprocessing.cpu_count() - 1
    Returns
    -------
    all_households : pandas.DataFrame
        synthesized household records
    all_persons : pandas.DataFrame
        synthesized person records
    all_stats : pandas.DataFrame
        chi-square and p-score values for each marginal geography drawn
    """
    cores = cores if cores else (multiprocessing.cpu_count()-1)
    part = partial(synthesize_zone, hh_marg, p_marg, hh_sample, p_sample)
    p = multiprocessing.Pool(cores)
    results = p.map(part, list(xwalk))
    p.close()
    p.join()

    hh_list = [result[0] for result in results]
    people_list = [result[1] for result in results]
    all_stats = pd.DataFrame([result[2] for result in results])
    all_households = pd.concat(hh_list)
    all_persons = pd.concat(people_list)
    all_households, all_persons = synch_hhids(all_households, all_persons)
    return all_households, all_persons,  all_stats
