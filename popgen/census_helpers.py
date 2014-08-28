import census
import pandas as pd
import numpy as np
import us


class Census:

    def __init__(self, key):
        self.c = census.Census(key)
        self.pums_relationship_file_url = "https://www.census.gov/geo/" \
                                          "maps-data/data/docs/rel/2010_"\
                                          "Census_Tract_to_2010_PUMA.txt"
        self.pums_relationship_df = None
        self.base_url = "http://paris.urbansim.org/data/pums/"
        self.pums_population_base_url = \
            self.base_url + "puma_p_%s_%s.csv"
        self.pums_household_base_url = \
            self.base_url + "puma_h_%s_%s.csv"
        self.pums_population_state_base_url = \
            self.base_url + "puma_p_%s.csv"
        self.pums_household_state_base_url = \
            self.base_url + "puma_h_%s.csv"
        self.fips_url = "https://www.census.gov/geo/reference/codes/files/" \
                        "national_county.txt"
        self.fips_df = None

    # df1 is the disaggregate data frame (e.g. block groups)
    # df2 is the aggregate data frame (e.g. tracts)
    # need to scale down df2 variables to df1 level totals
    def _scale_and_merge(self, df1, tot1, df2, tot2, columns_to_scale,
                         merge_columns, suffixes):
        df = pd.merge(df1, df2, left_on=merge_columns, right_on=merge_columns,
                      suffixes=suffixes)

        # going to scale these too so store current values
        tot2, tot1 = df[tot2], df[tot1]
        # if agg number if 0, disaggregate should be 0
        # note this is filled by fillna below
        assert np.all(tot1[tot2 == 0] == 0)

        for col in columns_to_scale:
            df[col] = df[col] / tot2 * tot1
            # round?
            df[col] = df[col].fillna(0).astype('int')
        return df

    def block_group_query(self, census_columns, state, county, tract=None,
                          year=None, id=None):
        if id is None:
            id = "*"
        return self._query(census_columns, state, county,
                           forstr="block group:%s" % id,
                           tract=tract, year=year)

    def tract_query(self, census_columns, state, county, tract=None,
                    year=None, id=None):
        if id is None:
            id = "*"
        return self._query(census_columns, state, county,
                           forstr="tract:%s" % id,
                           tract=None, year=year)

    def _query(self, census_columns, state, county, forstr,
               tract=None, year=None):
        c = self.c

        state, county = self.try_fips_lookup(state, county)

        if tract is None:
            in_str = 'state:%s county:%s' % (state, county)
        else:
            in_str = 'state:%s county:%s tract:%s' % (state, county, tract)

        dfs = []

        # unfortunately the api only queries 50 columns at a time
        # leave room for a few extra id columns
        def chunks(l, n):
            """ Yield successive n-sized chunks from l.
            """
            for i in xrange(0, len(l), n):
                yield l[i:i+n]

        for census_column_batch in chunks(census_columns, 45):
            census_column_batch = list(census_column_batch)
            d = c.acs.get(['NAME'] + census_column_batch,
                          geo={'for': forstr,
                               'in': in_str},
                          year=year)
            df = pd.DataFrame(d)
            df[census_column_batch] = df[census_column_batch].astype('int')
            dfs.append(df)

        assert len(dfs) >= 1
        df = dfs[0]
        for mdf in dfs[1:]:
            df = pd.merge(df, mdf, on="NAME", suffixes=("", "_ignore"))
            drop_cols = filter(lambda x: "_ignore" in x, df.columns)
            df = df.drop(drop_cols, axis=1)

        return df

    def block_group_and_tract_query(self, block_group_columns,
                                    tract_columns, state, county,
                                    merge_columns, block_group_size_attr,
                                    tract_size_attr, tract=None, year=None):
        df2 = self.tract_query(tract_columns, state, county, tract=tract,
                               year=year)
        df1 = self.block_group_query(block_group_columns, state, county,
                                     tract=tract, year=year)

        df = self._scale_and_merge(df1, block_group_size_attr, df2,
                                   tract_size_attr, tract_columns,
                                   merge_columns, suffixes=("", "_ignore"))
        drop_cols = filter(lambda x: "_ignore" in x, df.columns)
        df = df.drop(drop_cols, axis=1)

        return df

    def _get_pums_relationship(self):
        if self.pums_relationship_df is None:
            self.pums_relationship_df = \
                pd.read_csv(self.pums_relationship_file_url, dtype={
                    "STATEFP": "object",
                    "COUNTYFP": "object",
                    "TRACTCE": "object",
                    "PUMA5CE": "object"
                })
        return self.pums_relationship_df

    def _get_fips_lookup(self):
        if self.fips_df is None:
            self.fips_df = pd.read_csv(
                self.fips_url,
                dtype={
                    "State ANSI": "object",
                    "County ANSI": "object"
                },
                index_col=["State",
                           "County Name"]
            )
            del self.fips_df["ANSI Cl"]
        return self.fips_df

    def tract_to_puma(self, state, county, tract):

        state, county = self.try_fips_lookup(state, county)

        df = self._get_pums_relationship()
        q = "STATEFP == '%s' and COUNTYFP == '%s' and TRACTCE == '%s'" % \
            (state, county, tract)
        r = df.query(q)
        return r["PUMA5CE"].values[0]

    def tracts_to_pumas(self, state, county, tracts):

        state, county = self.try_fips_lookup(state, county)

        df = self._get_pums_relationship()
        q = "STATEFP == '%s' and COUNTYFP == '%s'" % (state, county)
        r = df.query(q)
        r = r[r["TRACTCE"].isin(tracts)]
        return list(r["PUMA5CE"].unique())

    def _read_csv(self, loc):
        return pd.read_csv(loc, dtype={
            "PUMA10": "object",
            "ST": "object"
        })

    def download_population_pums(self, state, puma=None):
        state = self.try_fips_lookup(state)
        if puma is None:
            return self._read_csv(self.pums_population_state_base_url % (state))
        return self._read_csv(self.pums_population_base_url % (state, puma))

    def download_household_pums(self, state, puma=None):
        state = self.try_fips_lookup(state)
        if puma is None:
            return self._read_csv(self.pums_household_state_base_url % (state))
        return self._read_csv(self.pums_household_base_url % (state, puma))

    def try_fips_lookup(self, state, county=None):
        df = self._get_fips_lookup()

        if county is None:
            try:
                return getattr(us.states, state).fips
            except:
                pass
            return state

        try:
            return df.loc[(state, county)]
        except:
            pass
        return state, county