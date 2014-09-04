from ipf.ipf import calculate_constraints
from ipu.ipu import household_weights
import categorizer as cat


class SynthPop:

    def __init__(self, recipe):
        self.recipe = recipe

    @staticmethod
    def synthesize(h_marg, p_marg, h_jd, p_jd, h_pums, p_pums, debug=False):

        # this is the zero marginal problem
        h_marg[h_marg == 0] = .01
        p_marg[p_marg == 0] = .01

        # ipf for households
        print "Running ipf for households"
        h_constraint, _ = calculate_constraints(h_marg, h_jd.frequency)

        # ipf for persons
        print "Running ipf for persons"
        p_constraint, _ = calculate_constraints(p_marg, p_jd.frequency)

        # is this the zero cell problem?
        h_constraint[h_constraint == 0] = .00001
        p_constraint[p_constraint == 0] = .00001

        # make frequency tables
        household_freq, person_freq = cat.frequency_tables(p_pums, h_pums,
                                                           p_jd.cat_id,
                                                           h_jd.cat_id)

        print h_constraint
        h_constraint = h_constraint.reset_index(drop=True)
        p_constraint = p_constraint.reset_index(drop=True)

        # for some reason there are households with no people
        l1 = len(household_freq)
        household_freq = household_freq[person_freq.sum(axis=1) > 0]
        person_freq = person_freq[person_freq.sum(axis=1) > 0]
        l2 = len(household_freq)
        if l2 - l1 > 0:
            print "Dropped %d households because they have no people in them" %\
                (l2-l1)

        # do the ipu to match person marginals
        print "Running ipu"
        import time
        t1 = time.time()
        best_weights, fit_quality, iterations = household_weights(
            household_freq,
            person_freq,
            h_constraint,
            p_constraint)
        print "%.3f" % (time.time()-t1)

        if debug:
            print "Weights from ipu"
            print best_weights.describe()
            print fit_quality
            print iterations

    def synthesize_all(self, debug=False):

        recipe = self.recipe
        print "\nSynthesizing at geog level: '%s'" % recipe.get_geography_name()

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
                print "\nHousehold pums"
                print h_pums.describe()

            p_pums, p_jd = recipe.get_person_joint_dist_for_geography(geog_id)
            if debug:
                print "\nPerson joint distribution"
                print p_jd
                print "\nPerson pums"
                print p_pums.describe()

            self.synthesize(h_marg, p_marg, h_jd, p_jd, h_pums, p_pums,
                            debug=debug)

            if debug:
                break
