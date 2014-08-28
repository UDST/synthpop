from .. import categorizer as cat


def marginals_and_joint_distribution(c, state, county, tract=None):
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

    Returns
    -------
    household_marginals : DataFrame
        Marginals per block group for the household data (from ACS)
    person_marginals : DataFrame
        Marginals per block group for the person data (from ACS)
    household_jointdist : DataFrame
        Joint distribution for the households (from PUMS)
    person_jointdist : DataFrame
        Joint distribution for the person records (from PUMS)
    """

    income_columns = ['B19001_0%02dE' % i for i in range(1, 18)]
    vehicle_columns = ['B08201_0%02dE' % i for i in range(1, 7)]
    workers_columns = ['B08202_0%02dE' % i for i in range(1, 6)]
    families_columns = ['B11001_001E', 'B11001_002E']
    block_group_columns = income_columns + families_columns
    tract_columns = vehicle_columns + workers_columns
    h_acs = c.block_group_and_tract_query(block_group_columns,
                                          tract_columns, state, county,
                                          merge_columns=['tract', 'county',
                                                         'state'],
                                          block_group_size_attr="B11001_001E",
                                          tract_size_attr="B08201_001E",
                                          tract=tract)

    population = ['B01001_001E']
    sex = ['B01001_002E', 'B01001_026E']
    race = ['B02001_0%02dE' % i for i in range(1, 11)]
    male_age_columns = ['B01001_0%02dE' % i for i in range(3, 26)]
    female_age_columns = ['B01001_0%02dE' % i for i in range(27, 50)]
    all_columns = population + sex + race + male_age_columns + \
        female_age_columns
    p_acs = c.block_group_query(all_columns, state, county, tract=tract)

    puma = c.tract_to_pums(state, county, tract)
    p_pums = c.download_population_pums(state, puma)
    h_pums = c.download_household_pums(state, puma)

    h_acs_cat = cat.categorize(h_acs, {
        ("households", "total"): "B11001_001E",
        ("children", "yes"): "B11001_002E",
        ("children", "no"): "B11001_001E - B11001_002E",
        ("income", "lt35"): "B19001_002E + B19001_003E + B19001_004E + "
                            "B19001_005E + B19001_006E + B19001_007E",
        ("income", "gt35-lt100"): "B19001_008E + B19001_009E + "
                                  "B19001_010E + B19001_011E + B19001_012E"
                                  "+ B19001_013E",
        ("income", "gt100"): "B19001_014E + B19001_015E + B19001_016E"
                             "+ B19001_017E",
        ("cars", "none"): "B08201_002E",
        ("cars", "one"): "B08201_003E",
        ("cars", "two or more"): "B08201_004E + B08201_005E + B08201_006E",
        ("workers", "none"): "B08202_002E",
        ("workers", "one"): "B08202_003E",
        ("workers", "two or more"): "B08202_004E + B08202_005E"
    }, index_cols=['NAME'])

    p_acs_cat = cat.categorize(p_acs, {
        ("population", "total"): "B01001_001E",
        ("age", "19 and under"): "B01001_003E + B01001_004E + B01001_005E + "
                                 "B01001_006E + B01001_007E + B01001_027E + "
                                 "B01001_028E + B01001_029E + B01001_030E + "
                                 "B01001_031E",
        ("age", "20 to 35"): "B01001_008E + B01001_009E + B01001_010E + "
                             "B01001_011E + B01001_012E + B01001_032E + "
                             "B01001_033E + B01001_034E + B01001_035E + "
                             "B01001_036E",
        ("age", "35 to 60"): "B01001_013E + B01001_014E + B01001_015E + "
                             "B01001_016E + B01001_017E + B01001_037E + "
                             "B01001_038E + B01001_039E + B01001_040E + "
                             "B01001_041E",
        ("age", "above 60"): "B01001_018E + B01001_019E + B01001_020E + "
                             "B01001_021E + B01001_022E + B01001_023E + "
                             "B01001_024E + B01001_025E + B01001_042E + "
                             "B01001_043E + B01001_044E + B01001_045E + "
                             "B01001_046E + B01001_047E + B01001_048E + "
                             "B01001_049E",
        ("race", "white"):   "B02001_002E",
        ("race", "black"):   "B02001_003E",
        ("race", "asian"):   "B02001_005E",
        ("race", "other"):   "B02001_004E + B02001_006E + B02001_007E + "
                             "B02001_008E",
        ("sex", "male"):     "B01001_002E",
        ("sex", "female"):   "B01001_026E"
    }, index_cols=['NAME'])

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

    _, jd_persons = cat.joint_distribution(
        p_pums,
        cat.category_combinations(p_acs_cat.columns),
        {"age": age_cat, "race": race_cat, "sex": sex_cat}
    )

    def cars_cat(r):
        if r.VEH == 0:
            return "none"
        elif r.VEH == 1:
            return "one"
        return "two or more"

    def children_cat(r):
        if r.NOC > 0:
            return "yes"
        return "no"

    def income_cat(r):
        if r.FINCP > 100000:
            return "gt100"
        elif r.FINCP > 35000:
            return "gt35-lt100"
        return "lt35"

    def workers_cat(r):
        if r.WIF == 3:
            return "two or more"
        elif r.WIF == 2:
            return "two or more"
        elif r.WIF == 1:
            return "one"
        return "none"

    _, jd_households = cat.joint_distribution(
        h_pums,
        cat.category_combinations(h_acs_cat.columns),
        {"cars": cars_cat, "children": children_cat,
         "income": income_cat, "workers": workers_cat}
    )

    return h_acs_cat, p_acs_cat, jd_households, jd_persons
