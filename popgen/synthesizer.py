from popgen.ipf import ipf


class PopSynth:

    def __init__(self, c):
        self.c = c

    def synthesize(self, h_marg, p_marg, h_jd, p_jd):

        tract_ind = h_marg.index.names.index("tract")
        state_ind = h_marg.index.names.index("state")
        county_ind = h_marg.index.names.index("county")
        assert list(h_marg.index.names) == list(p_marg.index.names)

        # running ipf for households
        h_weights = []
        for index, row in h_marg.iterrows():
            puma = self.c.tract_to_puma(index[state_ind],
                                        index[county_ind],
                                        index[tract_ind])
            s, _ = ipf.calculate_constraints(row, h_jd.loc[puma])
            h_weights.append(s)
        print len(h_weights)

        # running ipf for population
        p_weights = []
        for index, row in p_marg.iterrows():
            puma = self.c.tract_to_puma(index[state_ind],
                                        index[county_ind],
                                        index[tract_ind])
            s, _ = ipf.calculate_constraints(row, p_jd.loc[puma])
            p_weights.append(s)
        print len(p_weights)
