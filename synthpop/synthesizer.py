from ipf.ipf import calculate_constraints
from ipu.ipu import household_weights
import categorizer as cat
import numpy as np
import pandas as pd
import logging
import sys
logger = logging.getLogger("synthpop")


def enable_logging():
    handler = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def synthesize(h_marg, p_marg, h_jd, p_jd, h_pums, p_pums,
               marginal_zero_sub=.01, constraint_zero_sub=.01):

    # this is the zero marginal problem
    h_marg = h_marg.replace(0, marginal_zero_sub)
    p_marg = p_marg.replace(0, marginal_zero_sub)

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

    # is this the zero cell problem?
    h_constraint = h_constraint.replace(0, constraint_zero_sub)
    p_constraint = p_constraint.replace(0, constraint_zero_sub)

    # make frequency tables that the ipu expects
    household_freq, person_freq = cat.frequency_tables(p_pums, h_pums,
                                                       p_jd.cat_id,
                                                       h_jd.cat_id)

    # TODO this is still a problem right?
    '''
    # for some reason there are households with no people
    l1 = len(household_freq)
    household_freq = household_freq[person_freq.sum(axis=1) > 0]
    person_freq = person_freq[person_freq.sum(axis=1) > 0]
    l2 = len(household_freq)
    if l2 - l1 > 0:
        print "Dropped %d households because they have no people in them" %\
            (l2-l1)
    '''
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

    # TODO this isn't the only way to draw?
    indexes = np.random.choice(h_pums.index.values,
                               size=num_households,
                               replace=True,
                               p=(best_weights/best_weights.sum()).values)
    # TODO deal with p_pums too
    return h_pums.loc[indexes]


def synthesize_all(recipe, num_geogs=None, indexes=None,
                   marginal_zero_sub=.01, constraint_zero_sub=.01):

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

        hh = synthesize(h_marg, p_marg, h_jd, p_jd, h_pums, p_pums,
                        marginal_zero_sub=marginal_zero_sub,
                        constraint_zero_sub=constraint_zero_sub)
        hhs.append(hh)

        cnt += 1
        if num_geogs is not None and cnt >= num_geogs:
            break

    # TODO might want to write this to disk as we go?
    return pd.concat(hhs, verify_integrity=True, ignore_index=True)
