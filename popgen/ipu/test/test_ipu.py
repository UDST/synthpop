import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from .. import ipu


@pytest.fixture
def frequency_columns():
    return pd.MultiIndex.from_tuples(
        [('household', 1),
         ('household', 2),
         ('person', 1),
         ('person', 2),
         ('person', 3)])


@pytest.fixture
def frequency_table(frequency_columns):
    df = pd.DataFrame(
        [(1, 0, 1, 1, 1),
         (1, 0, 1, 0, 1),
         (1, 0, 2, 1, 0),
         (0, 1, 1, 0, 2),
         (0, 1, 0, 2, 1),
         (0, 1, 1, 1, 0),
         (0, 1, 2, 1, 2),
         (0, 1, 1, 1, 0)],
        index=[1, 2, 3, 4, 5, 6, 7, 8],
        columns=frequency_columns)

    return df


@pytest.fixture
def constraints(frequency_columns):
    return pd.Series(
        [35, 65, 91, 65, 104], index=frequency_columns, dtype='float')


def test_calculate_fit_quality(frequency_table, constraints):
    weights = pd.Series(
        np.ones(len(frequency_table)), index=frequency_table.index)
    column = frequency_table['household'][1]
    constraint = constraints['household'][1]

    npt.assert_allclose(
        ipu.calculate_fit_quality(column, weights, constraint), 0.9143,
        atol=0.0001)

    weights = pd.Series(
        [12.37, 14.61, 8.05, 16.28, 16.91, 8.97, 13.78, 8.97],
        index=frequency_table.index)
    column = frequency_table['person'][2]
    constraint = constraints['person'][2]

    npt.assert_allclose(
        ipu.calculate_fit_quality(column, weights, constraint), 0.3222,
        atol=0.0003)
