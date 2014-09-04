from ipf.ipf import calculate_constraints
from ipu.ipu import household_weights
import categorizer as cat
import numpy as np
import pandas as pd


def synthesize(h_marg, p_marg, h_jd, p_jd, h_pums, p_pums,
               marginal_zero_sub=.01, constraint_zero_sub=.01, debug=False):

    # this is the zero marginal problem
    h_marg[h_marg == 0] = marginal_zero_sub
    p_marg[p_marg == 0] = marginal_zero_sub

    # ipf for households
    print "Running ipf for households"
    h_constraint, _ = calculate_constraints(h_marg, h_jd.frequency)
    h_constraint.index = h_jd.cat_id
    # TODO convert all these prints to logging messages
    print "\nHousehold constraint"
    print h_constraint

    # ipf for persons
    print "Running ipf for persons"
    p_constraint, _ = calculate_constraints(p_marg, p_jd.frequency)
    p_constraint.index = p_jd.cat_id
    print "\nPerson constraint"
    print p_constraint

    # is this the zero cell problem?
    h_constraint[h_constraint == 0] = constraint_zero_sub
    p_constraint[p_constraint == 0] = constraint_zero_sub

    # make frequency tables
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
    print "Running ipu"
    import time
    t1 = time.time()
    best_weights, fit_quality, iterations = household_weights(household_freq,
                                                              person_freq,
                                                              h_constraint,
                                                              p_constraint)
    print "Time to run ipu: %.3fs" % (time.time()-t1)

    if debug:
        print "Weights from ipu"
        print "\nIPU weights:\n", best_weights.describe()
        print "\nFit quality:", fit_quality
        print "\nNumber of iterations:", iterations

    num_households = int(h_marg.groupby(level=0).sum().mean())
    print "Drawing %d households" % num_households
    indexes = np.random.choice(h_pums.index.values,
                               size=num_households,
                               replace=True,
                               p=(best_weights/best_weights.sum()).values)
    # TODO deal with p_pums too
    return h_pums.loc[indexes]


def synthesize_all(recipe, debug=False):

    print "\nSynthesizing at geog level: '%s'" % recipe.get_geography_name()

    hhs = []
    # TODO will parallelization work here?
    for geog_id in recipe.get_available_geography_ids():
        print "\nSynthesizing geog id:\n", geog_id

        h_marg = recipe.get_household_marginal_for_geography(geog_id)
        if debug:
            print "\nHousehold marginal"
            print h_marg

        p_marg = recipe.get_person_marginal_for_geography(geog_id)
        if debug:
            print "\nPerson marginal"
            print p_marg

        h_pums, h_jd = recipe.\
            get_household_joint_dist_for_geography(geog_id)
        if debug:
            print "\nHousehold joint distribution"
            print h_jd
            # print "\nHousehold pums"
            # print h_pums.describe()

        p_pums, p_jd = recipe.get_person_joint_dist_for_geography(geog_id)
        if debug:
            print "\nPerson joint distribution"
            print p_jd
            # print "\nPerson pums"
            # print p_pums.describe()

        hh = synthesize(h_marg, p_marg, h_jd, p_jd, h_pums, p_pums,
                        debug=debug)

        hhs.append(hh)

        if debug:
            break

    # TODO might want to write this to disk?
    return pd.concat(hhs, axis=1)
