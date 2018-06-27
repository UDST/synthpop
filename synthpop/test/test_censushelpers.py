import pytest
from ..census_helpers import Census
import numpy as np
from pandas.util.testing import assert_series_equal
import os


@pytest.fixture
def c():
    return Census('bfa6b4e541243011fab6307a31aed9e91015ba90')


def test_block_group_and_tract_query(c):
    income_columns = ['B19001_0%02dE' % i for i in range(1, 18)]
    vehicle_columns = ['B08201_0%02dE' % i for i in range(1, 7)]
    workers_columns = ['B08202_0%02dE' % i for i in range(1, 6)]
    families_columns = ['B11001_001E', 'B11001_002E']
    block_group_columns = income_columns + families_columns
    tract_columns = vehicle_columns + workers_columns
    df = c.block_group_and_tract_query(block_group_columns,
                                       tract_columns, "06", "075",
                                       merge_columns=['tract', 'county',
                                                      'state'],
                                       block_group_size_attr="B11001_001E",
                                       tract_size_attr="B08201_001E",
                                       tract="030600")

    assert len(df) == 3
    assert_series_equal(
      df["B11001_001E"], df["B08201_001E"], check_names=False)
    assert np.all(df.state == "06")
    assert np.all(df.county == "075")

    df = c.block_group_and_tract_query(block_group_columns,
                                       tract_columns, "06", "075",
                                       merge_columns=['tract', 'county',
                                                      'state'],
                                       block_group_size_attr="B11001_001E",
                                       tract_size_attr="B08201_001E",
                                       tract=None)

    # number of block groups in San Francisco
    assert len(df) == 581
    assert_series_equal(
      df["B11001_001E"], df["B08201_001E"], check_names=False)
    assert np.all(df.state == "06")
    assert np.all(df.county == "075")


def test_wide_block_group_query(c):
    population = ['B01001_001E']
    sex = ['B01001_002E', 'B01001_026E']
    race = ['B02001_0%02dE' % i for i in range(1, 11)]
    male_age_columns = ['B01001_0%02dE' % i for i in range(3, 26)]
    female_age_columns = ['B01001_0%02dE' % i for i in range(27, 50)]
    all_columns = population + sex + race + male_age_columns + \
        female_age_columns
    df = c.block_group_query(all_columns, "06", "075", tract="030600")

    assert len(df) == 3
    assert np.all(df.state == "06")
    assert np.all(df.county == "075")
    assert len(df.columns) > 50


def test_tract_to_puma(c):
    puma = c.tract_to_puma("06", "075", "030600")[0]
    assert puma == "07506"


def test_download_pums(c):
    puma = "07506"
    c.download_population_pums("06", puma)
    c.download_household_pums("06", puma)
    c.download_population_pums("10")
    c.download_household_pums("10")
