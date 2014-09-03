from ipf.ipf import calculate_constraints
from ipu.ipu import household_weights
import categorizer as cat


class SynthPop:

    def __init__(self, c):
        self.c = c

    @staticmethod
    def synthesize(h_marg, p_marg, h_jd, p_jd, h_pums, p_pums, h_cat_ids,
                   p_cat_ids):

        #print h_marg
        #print p_marg
        #print h_jd
        #print p_jd
        # this is the zero marginal problem
        h_marg[h_marg == 0] = .01
        p_marg[p_marg == 0] = .01

        # ipf for households
        h_constraint, _ = calculate_constraints(h_marg, h_jd)

        # ipf for persons
        p_constraint, _ = calculate_constraints(p_marg, p_jd)

        # is this the zero cell problem?
        h_constraint[h_constraint == 0] = .00001
        p_constraint[p_constraint == 0] = .00001

        # make frequency tables
        household_freq, person_freq = cat.frequency_tables(p_pums, h_pums,
                                                           p_cat_ids, h_cat_ids)

        h_constraint = h_constraint.reset_index(drop=True)
        p_constraint = p_constraint.reset_index(drop=True)

        # for some reason there are households with no people
        household_freq = household_freq[person_freq.sum(axis=1) > 0]
        person_freq = person_freq[person_freq.sum(axis=1) > 0]
        #print household_freq
        #print person_freq
        #print h_constraint
        #print p_constraint

        # do the ipu to match person marginals
        best_weights, fit_qual, iterations = household_weights(household_freq,
                                                               person_freq,
                                                               h_constraint,
                                                               p_constraint)
        print best_weights.describe()
        print fit_qual
        print iterations
        import time
        print time.ctime()

    def synthesize_many(self, h_marg_df, p_marg_df, h_jd_df, p_jd_df,
                        p_pums_d, h_pums_d, p_cat_ids, h_cat_ids):

        tract_ind = h_marg_df.index.names.index("tract")
        state_ind = h_marg_df.index.names.index("state")
        county_ind = h_marg_df.index.names.index("county")
        assert list(h_marg_df.index.names) == list(p_marg_df.index.names)

        # iterate over households, should be the same index in the person
        # marginals
        print len(h_marg_df)
        for index in h_marg_df.index:
            puma = self.c.tract_to_puma(index[state_ind],
                                        index[county_ind],
                                        index[tract_ind])
            print index[tract_ind]
            print puma

            h_pums = h_pums_d[puma]
            p_pums = p_pums_d[puma]

            self.synthesize(h_marg_df.loc[index],
                            p_marg_df.loc[index],
                            h_jd_df.loc[puma],
                            p_jd_df.loc[puma],
                            h_pums, p_pums,
                            h_cat_ids, p_cat_ids)
