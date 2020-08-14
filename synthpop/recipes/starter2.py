import numpy as np
import pandas as pd

from .. import categorizer as cat
from ..census_helpers import Census


# TODO DOCSTRINGS!!
class Starter:
    """
    This is a recipe for getting the marginals and joint distributions to use
    to pass to the synthesizer using simple categories - population, age,
    race, and sex for people, and children, income, cars, and workers for
    households.  This module is responsible for

    Parameters
    ----------
    c : object
        census_helpers.Census object
    state : string
        FIPS code the state
    county : string
        FIPS code for the county
    tract : string, optional
        FIPS code for a specific track or None for all tracts in the county
    acsyear : integer, optional
        Final year in the 5-year estimates ACS dataset.
        Default: 2016, which corresponds to 2011-2016 ACS dataset

    Returns
    -------
    household_marginals : DataFrame
        Marginals per block group for the household data (from ACS 5-year estimates)
    person_marginals : DataFrame
        Marginals per block group for the person data (from ACS 5-year estimates)
    household_jointdist : DataFrame
        joint distributions for the households (from PUMS 2010-2000), one joint
        distribution for each PUMA (one row per PUMA)
    person_jointdist : DataFrame
        joint distributions for the persons (from PUMS 2010-2000), one joint
        distribution for each PUMA (one row per PUMA)
    tract_to_puma_map : dictionary
        keys are tract ids and pumas are puma ids
    """

    def __init__(self, key, state, county, tract=None, acsyear=2016):
        self.c = c = Census(key, acsyear)
        self.state = state
        self.county = county
        self.tract = tract
        self.acsyear = acsyear

        structure_size_columns = ['B25032_0%02dE' % i for i in range(1, 24)]
        age_of_head_columns = ['B25007_0%02dE' % i for i in range(1, 22)]
        race_of_head_columns = ['B25006_0%02dE' % i for i in range(1, 11)]
        hispanic_head_columns = ['B25003I_0%02dE' % i for i in range(1, 4)]
        hh_size_columns = ['B25009_0%02dE' % i for i in range(1, 18)]
        income_columns = ['B19001_0%02dE' % i for i in range(1, 18)]
        vehicle_columns = ['B08201_0%02dE' % i for i in range(1, 7)]
        workers_columns = ['B08202_0%02dE' % i for i in range(1, 6)]
        presence_of_children_columns = ['B11005_001E', 'B11005_002E', 'B11005_011E']
        presence_of_seniors_columns = ['B11007_002E', 'B11007_007E']
        tenure_mover_columns = ['B25038_0%02dE' % i for i in range(1, 16)]
        block_group_columns = (
            income_columns + presence_of_children_columns +
            presence_of_seniors_columns + tenure_mover_columns +
            hh_size_columns + age_of_head_columns + structure_size_columns +
            race_of_head_columns + hispanic_head_columns)
        tract_columns = vehicle_columns + workers_columns
        h_acs = c.block_group_and_tract_query(
            block_group_columns,
            tract_columns, state, county,
            merge_columns=['tract', 'county', 'state'],
            block_group_size_attr="B11005_001E",
            tract_size_attr="B08201_001E",
            tract=tract, year=acsyear)
        self.h_acs = h_acs

        self.h_acs_cat = cat.categorize(h_acs, {
            ("sf_detached", "yes"): "B25032_003E + B25032_014E",
            ("sf_detached", "no"): "B25032_001E - B25032_003E - B25032_014E",
            ("hh_age_of_head", "lt35"):
                "B25007_003E + B25007_004E + B25007_013E + B25007_014E",
            ("hh_age_of_head", "gt35-lt65"):
                "B25007_005E + B25007_006E + B25007_007E + B25007_008E + "
                "B25007_015E + B25007_016E + B25007_017E + B25007_018E",
            ("hh_age_of_head", "gt65"):
                "B25007_009E + B25007_010E + B25007_011E + "
                "B25007_019E + B25007_020E + B25007_021E",
            ("hh_race_of_head", "black"): "B25006_003E",
            ("hh_race_of_head", "white"): "B25006_002E",
            ("hh_race_of_head", "asian"): "B25006_005E",
            ("hh_race_of_head", "other"):
                "B25006_004E + B25006_006E + B25006_007E + B25006_008E ",
            ("hispanic_head", "yes"): "B25003I_001E",
            ("hispanic_head", "no"): "B11005_001E - B25003I_001E",
            ("hh_children", "yes"): "B11005_002E",
            ("hh_children", "no"): "B11005_011E",
            ("seniors", "yes"): "B11007_002E",
            ("seniors", "no"): "B11007_007E",
            ("hh_income", "lt30"):
                "B19001_002E + B19001_003E + B19001_004E + "
                "B19001_005E + B19001_006E",
            ("hh_income", "gt30-lt60"):
                "B19001_007E + B19001_008E + B19001_009E + "
                "B19001_010E + B19001_011E",
            ("hh_income", "gt60-lt100"): "B19001_012E + B19001_013E",
            ("hh_income", "gt100-lt150"): "B19001_014E + B19001_015E",
            ("hh_income", "gt150"): "B19001_016E + B19001_017E",
            ("hh_cars", "none"): "B08201_002E",
            ("hh_cars", "one"): "B08201_003E",
            ("hh_cars", "two or more"):
                "B08201_004E + B08201_005E + B08201_006E",
            ("hh_workers", "none"): "B08202_002E",
            ("hh_workers", "one"): "B08202_003E",
            ("hh_workers", "two or more"): "B08202_004E + B08202_005E",
            ("tenure_mover", "own recent"): "B25038_003E",
            ("tenure_mover", "own not recent"): "B25038_002E - B25038_003E",
            ("tenure_mover", "rent recent"): "B25038_010E",
            ("tenure_mover", "rent not recent"): "B25038_009E - B25038_010E",
            ("hh_size", "one"): "B25009_003E + B25009_011E",
            ("hh_size", "two"): "B25009_004E + B25009_012E",
            ("hh_size", "three"): "B25009_005E + B25009_013E",
            ("hh_size", "four or more"): "B25009_006E + B25009_014E + "
                                         "B25009_007E + B25009_015E + "
                                         "B25009_008E + B25009_016E + "
                                         "B25009_009E + B25009_017E",
        }, index_cols=['state', 'county', 'tract', 'block group'])

        # gq_population = ['B26001_001E']
        # HH population, for the hhpop/totalpop adjustment
        hh_population = ['B11002_001E']
        population = ['B01001_001E']  # This includes GQ
        hispanic = ['B03003_002E', 'B03003_003E']
        sex = ['B01001_002E', 'B01001_026E']
        race = ['B02001_0%02dE' % i for i in range(1, 11)]
        male_age_columns = ['B01001_0%02dE' % i for i in range(3, 26)]
        female_age_columns = ['B01001_0%02dE' % i for i in range(27, 50)]
        all_columns = population + sex + race + male_age_columns + \
            female_age_columns + hh_population + hispanic
        p_acs = c.block_group_query(all_columns, state, county, tract=tract, year=acsyear)
        self.p_acs = p_acs
        self.p_acs_cat = cat.categorize(p_acs, {
            ("person_age", "19 and under"):
                "(B01001_003E + B01001_004E + B01001_005E + "
                "B01001_006E + B01001_007E + B01001_027E + "
                "B01001_028E + B01001_029E + B01001_030E + "
                "B01001_031E) * B11002_001E*1.0/B01001_001E",
            ("person_age", "20 to 35"):
                "(B01001_008E + B01001_009E + B01001_010E + "
                "B01001_011E + B01001_012E + B01001_032E + "
                "B01001_033E + B01001_034E + B01001_035E + "
                "B01001_036E) * B11002_001E*1.0/B01001_001E",
            ("person_age", "35 to 60"):
                "(B01001_013E + B01001_014E + B01001_015E + "
                "B01001_016E + B01001_017E + B01001_037E + "
                "B01001_038E + B01001_039E + B01001_040E + "
                "B01001_041E) * B11002_001E*1.0/B01001_001E",
            ("person_age", "above 60"):
                "(B01001_018E + B01001_019E + B01001_020E + "
                "B01001_021E + B01001_022E + B01001_023E + "
                "B01001_024E + B01001_025E + B01001_042E + "
                "B01001_043E + B01001_044E + B01001_045E + "
                "B01001_046E + B01001_047E + B01001_048E + "
                "B01001_049E) * B11002_001E*1.0/B01001_001E",
            ("race", "white"):   "(B02001_002E) * B11002_001E*1.0/B01001_001E",
            ("race", "black"):   "(B02001_003E) * B11002_001E*1.0/B01001_001E",
            ("race", "asian"):   "(B02001_005E) * B11002_001E*1.0/B01001_001E",
            ("race", "other"):   "(B02001_004E + B02001_006E + B02001_007E + "
                                 "B02001_008E) * B11002_001E*1.0/B01001_001E",
            ("person_sex", "male"):
                "(B01001_002E) * B11002_001E*1.0/B01001_001E",
            ("person_sex", "female"):
                "(B01001_026E) * B11002_001E*1.0/B01001_001E",
            ("hispanic", "yes"):
                "(B03003_003E) * B11002_001E*1.0/B01001_001E",
            ("hispanic", "no"):
                "(B03003_002E) * B11002_001E*1.0/B01001_001E",
        }, index_cols=['state', 'county', 'tract', 'block group'])

        # Put the needed PUMS variables here.  These are also the PUMS variables
        # that will be in the outputted synthetic population
        self.h_pums_cols = ('serialno', 'PUMA10', 'RT', 'NP', 'TYPE',
                            'R65', 'HINCP', 'VEH', 'MV', 'TEN', 'BLD', 'R18')
        self.p_pums_cols = ('serialno', 'PUMA10', 'RELP', 'AGEP',
                            'ESR', 'RAC1P', 'HISP', 'SEX', 'SPORDER',
                            'PERNP', 'SCHL', 'WKHP', 'JWTR', 'SCH')
        if self.acsyear < 2018:
            self.h_pums_cols = list(self.h_pums_cols)
            self.h_pums_cols.insert(1, 'PUMA00')
            self.h_pums_cols = tuple(self.h_pums_cols)
            self.p_pums_cols = list(self.p_pums_cols)
            self.p_pums_cols.insert(1, 'PUMA00')
            self.p_pums_cols = tuple(self.p_pums_cols)

    def get_geography_name(self):
        # this synthesis is at the block group level for most variables
        return "block_group"

    def get_num_geographies(self):
        return len(self.p_acs_cat)

    def get_available_geography_ids(self):
        # return the ids of the geographies, in this case a state, county,
        # tract, block_group id tuple
        for tup in self.p_acs_cat.index:
            yield pd.Series(tup, index=self.p_acs_cat.index.names)

    def get_household_marginal_for_geography(self, ind):
        return self.h_acs_cat.loc[tuple(ind.values)]

    def get_person_marginal_for_geography(self, ind):
        return self.p_acs_cat.loc[tuple(ind.values)]

    def get_household_joint_dist_for_geography(self, ind):
        c = self.c

        puma10, puma00 = c.tract_to_puma(ind.state, ind.county, ind.tract)

        # this is cached so won't download more than once
        if type(puma00) == str:
            h_pums = self.c.download_household_pums(ind.state, puma10, puma00,
                                                    usecols=self.h_pums_cols)
            p_pums = self.c.download_population_pums(ind.state, puma10, puma00,
                                                     usecols=self.p_pums_cols)
        elif np.isnan(puma00):  # only puma10 available
            h_pums = self.c.download_household_pums(ind.state, puma10, None,
                                                    usecols=self.h_pums_cols)
            p_pums = self.c.download_population_pums(ind.state, puma10, None,
                                                     usecols=self.p_pums_cols)

        h_pums = h_pums.set_index('serialno')

        # join persons to households,
        # calculate needed household-level variables
        age_of_head = p_pums[p_pums.RELP == 0].groupby('serialno').AGEP.max()
        num_workers = p_pums[p_pums.ESR.isin([1, 2, 4, 5])].groupby(
            'serialno').size()
        h_pums['race_of_head'] = p_pums[p_pums.RELP == 0].groupby(
            'serialno').RAC1P.max()
        h_pums['hispanic_head'] = p_pums[p_pums.RELP == 0].groupby(
            'serialno').HISP.max()
        h_pums['age_of_head'] = age_of_head
        h_pums['workers'] = num_workers
        h_pums.workers = h_pums.workers.fillna(0)
        h_pums = h_pums.reset_index()

        def sf_detached_cat(r):
            if r.BLD == 2:
                return "yes"
            return "no"

        def age_of_head_cat(r):
            if r.age_of_head < 35:
                return "lt35"
            elif r.age_of_head >= 65:
                return "gt65"
            return "gt35-lt65"

        def race_of_head_cat(r):
            if r.race_of_head == 1:
                return "white"
            elif r.race_of_head == 2:
                return "black"
            elif r.race_of_head == 6:
                return "asian"
            return "other"

        def hispanic_head_cat(r):
            if r.hispanic_head == 1:
                return "no"
            return "yes"

        def hh_size_cat(r):
            if r.NP == 1:
                return "one"
            elif r.NP == 2:
                return "two"
            elif r.NP == 3:
                return "three"
            return "four or more"

        def cars_cat(r):
            if r.VEH == 0:
                return "none"
            elif r.VEH == 1:
                return "one"
            return "two or more"

        def children_cat(r):
            if r.R18 == 1:
                return "yes"
            return "no"

        def seniors_cat(r):
            if r.R65 > 0:
                return "yes"
            return "no"

        def income_cat(r):
            if r.HINCP >= 150000:
                return "gt150"
            elif (r.HINCP >= 100000) & (r.HINCP < 150000):
                return "gt100-lt150"
            elif (r.HINCP >= 60000) & (r.HINCP < 100000):
                return "gt60-lt100"
            elif (r.HINCP >= 30000) & (r.HINCP < 60000):
                return "gt30-lt60"
            return "lt30"

        def workers_cat(r):
            if r.workers >= 2:
                return "two or more"
            elif r.workers == 1:
                return "one"
            return "none"

        def tenure_mover_cat(r):
            if (r.MV < 4) & (r.TEN < 3):
                return "own recent"
            elif (r.MV >= 4) & (r.TEN < 3):
                return "own not recent"
            elif (r.MV < 4) & (r.TEN >= 3):
                return "rent recent"
            return "rent not recent"

        h_pums, jd_households = cat.joint_distribution(
            h_pums,
            cat.category_combinations(self.h_acs_cat.columns),
            {"hh_cars": cars_cat,
             "hh_children": children_cat,
             "hh_income": income_cat,
             "hh_workers": workers_cat,
             "tenure_mover": tenure_mover_cat,
             "seniors": seniors_cat,
             "hh_size": hh_size_cat,
             "hh_age_of_head": age_of_head_cat,
             "sf_detached": sf_detached_cat,
             "hh_race_of_head": race_of_head_cat,
             "hispanic_head": hispanic_head_cat}
        )
        return h_pums, jd_households

    def get_person_joint_dist_for_geography(self, ind):
        c = self.c

        puma10, puma00 = c.tract_to_puma(ind.state, ind.county, ind.tract)
        # this is cached so won't download more than once
        if type(puma00) == str:
            p_pums = self.c.download_population_pums(ind.state, puma10, puma00,
                                                     usecols=self.p_pums_cols)
        elif np.isnan(puma00):  # only puma10 available
            p_pums = self.c.download_population_pums(ind.state, puma10, None,
                                                     usecols=self.p_pums_cols)

        def age_cat(r):
            if r.AGEP <= 19:
                return "19 and under"
            elif r.AGEP <= 35:
                return "20 to 35"
            elif r.AGEP <= 60:
                return "35 to 60"
            return "above 60"

        def race_cat(r):
            if r.RAC1P == 1:
                return "white"
            elif r.RAC1P == 2:
                return "black"
            elif r.RAC1P == 6:
                return "asian"
            return "other"

        def sex_cat(r):
            if r.SEX == 1:
                return "male"
            return "female"

        def hispanic_cat(r):
            if r.HISP == 1:
                return "no"
            return "yes"

        p_pums, jd_persons = cat.joint_distribution(
            p_pums,
            cat.category_combinations(self.p_acs_cat.columns),
            {"person_age": age_cat, "race": race_cat, "person_sex": sex_cat,
             "hispanic": hispanic_cat}
        )
        return p_pums, jd_persons
