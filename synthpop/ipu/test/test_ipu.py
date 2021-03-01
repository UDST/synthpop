import numpy as np
import numpy.testing as npt
import pandas as pd
import random
import pytest
from pandas.util import testing as pdt

from .. import ipu


@pytest.fixture(scope='module')
def household_columns():
    return pd.MultiIndex.from_product(
        [('yes',), ('blue', 'red')],
        names=['cat_owner', 'car_color'])


@pytest.fixture(scope='module')
def person_columns():
    return pd.MultiIndex.from_product(
        [(7, 8, 9), ('pink',)], names=['shoe_size', 'shirt_color'])


@pytest.fixture(scope='module')
def household_freqs(household_columns):
    return pd.DataFrame(
        [(1, 0),
         (1, 0),
         (1, 0),
         (0, 1),
         (0, 1),
         (0, 1),
         (0, 1),
         (0, 1)],
        index=range(1, 9),
        columns=household_columns)


@pytest.fixture(scope='module')
def person_freqs(person_columns):
    return pd.DataFrame(
        [(1, 1, 1),
         (1, 0, 1),
         (2, 1, 0),
         (1, 0, 2),
         (0, 2, 1),
         (1, 1, 0),
         (2, 1, 2),
         (1, 1, 0)],
        index=range(1, 9),
        columns=person_columns)


@pytest.fixture(scope='module')
def household_constraints(household_columns):
    return pd.Series([35, 65], index=household_columns)


@pytest.fixture(scope='module')
def person_constraints(person_columns):
    return pd.Series([91, 65, 104], index=person_columns)


@pytest.fixture(scope='module')
def geography():
    dtypes = ['serie', 'list']
    dtype = random.choice(dtypes)

    if dtype == 'serie':
        geography = pd.Series({'state': '02',
                               'county': '270',
                               'tract': '000100',
                               'block group': '1'})
    else:
        geography = ['02', '270']

    return geography


@pytest.fixture
def freq_wrap(
        household_freqs, person_freqs, household_constraints,
        person_constraints):
    return ipu._FrequencyAndConstraints(
        household_freqs, household_constraints, person_freqs,
        person_constraints)


def test_drop_zeros_households(household_freqs):
    df = list(ipu._drop_zeros(household_freqs))

    assert len(df) == 2
    assert df[0][0] == ('yes', 'blue')
    npt.assert_array_equal(df[0][1], [1, 1, 1])
    npt.assert_array_equal(df[0][2], [0, 1, 2])
    assert df[1][0] == ('yes', 'red')
    npt.assert_array_equal(df[1][1], [1, 1, 1, 1, 1])
    npt.assert_array_equal(df[1][2], [3, 4, 5, 6, 7])


def test_drop_zeros_person(person_freqs):
    df = list(ipu._drop_zeros(person_freqs))

    assert len(df) == 3
    assert df[0][0] == (7, 'pink')
    npt.assert_array_equal(df[0][1], [1, 1, 2, 1, 1, 2, 1])
    npt.assert_array_equal(df[0][2], [0, 1, 2, 3, 5, 6, 7])


def test_fit_quality(
        household_freqs, person_freqs, household_constraints,
        person_constraints):
    weights = np.ones(len(household_freqs), dtype='float')
    column = household_freqs[('yes', 'blue')]
    constraint = household_constraints[('yes', 'blue')]

    npt.assert_allclose(
        ipu._fit_quality(column, weights, constraint), 0.9143,
        atol=0.0001)

    weights = np.array([12.37, 14.61, 8.05, 16.28, 16.91, 8.97, 13.78, 8.97])
    column = person_freqs[(8, 'pink')]
    constraint = person_constraints[(8, 'pink')]

    npt.assert_allclose(
        ipu._fit_quality(column, weights, constraint), 0.3222,
        atol=0.0003)


def test_average_fit_quality(household_freqs, freq_wrap):
    weights = np.ones(len(household_freqs), dtype='float')
    npt.assert_allclose(
        ipu._average_fit_quality(freq_wrap, weights),
        0.9127,
        atol=0.0001)

    weights = np.array([12.37, 14.61, 8.05, 16.28, 16.91, 8.97, 13.78, 8.97])
    npt.assert_allclose(
        ipu._average_fit_quality(freq_wrap, weights),
        0.0954,
        atol=0.0001)


def test_update_weights(
        household_freqs, person_freqs, household_constraints,
        person_constraints):
    column = household_freqs[('yes', 'blue')]
    column = column.iloc[column.values.nonzero()[0]]
    constraint = household_constraints[('yes', 'blue')]
    weights = pd.Series(
        np.ones(len(column)),
        index=column.index)

    npt.assert_allclose(
        ipu._update_weights(column, weights, constraint),
        [11.67, 11.67, 11.67],
        atol=0.01)

    column = person_freqs[(9, 'pink')]
    column = column.iloc[column.values.nonzero()[0]]
    constraint = person_constraints[(9, 'pink')]
    weights = pd.Series(
        [8.05, 9.51, 8.05, 10.59, 11.0, 8.97, 8.97, 8.97],
        index=range(1, 9)).loc[column.index]

    npt.assert_allclose(
        ipu._update_weights(column, weights, constraint),
        [12.37, 14.61, 16.28, 16.91, 13.78],
        atol=0.01)


def test_household_weights(
        household_freqs, person_freqs, household_constraints,
        person_constraints, geography, ignore_max_iters=False):
    weights, fit_qual, iterations = ipu.household_weights(
        household_freqs, person_freqs, household_constraints,
        person_constraints, geography, ignore_max_iters, convergence=1e-7)
    npt.assert_allclose(
        weights.values,
        [1.36, 25.66, 7.98, 27.79, 18.45, 8.64, 1.47, 8.64],
        atol=0.02)
    npt.assert_allclose(fit_qual, 8.51e-6, atol=1e-8)
    npt.assert_allclose(iterations, 637, atol=5)


def test_household_weights_max_iter(
        household_freqs, person_freqs, household_constraints,
        person_constraints, geography, ignore_max_iters=False):
    with pytest.raises(RuntimeError):
        ipu.household_weights(
            household_freqs, person_freqs, household_constraints,
            person_constraints, geography, ignore_max_iters, convergence=1e-7, max_iterations=10)


def test_FrequencyAndConstraints(freq_wrap):
    assert freq_wrap.ncols == 5
    assert len(list(freq_wrap.iter_columns())) == 5

    iter_cols = iter(freq_wrap.iter_columns())

    key, col, constraint, nz = next(iter_cols)
    assert key == ('yes', 'blue')
    npt.assert_array_equal(col, [1, 1, 1])
    assert constraint == 35
    npt.assert_array_equal(nz, [0, 1, 2])

    key, col, constraint, nz = next(iter_cols)
    assert key == ('yes', 'red')
    npt.assert_array_equal(col, [1, 1, 1, 1, 1])
    assert constraint == 65
    npt.assert_array_equal(nz, [3, 4, 5, 6, 7])

    # should be into person cols now
    key, col, constraint, nz = next(iter_cols)
    assert key == (7, 'pink')
    npt.assert_array_equal(col, [1, 1, 2, 1, 1, 2, 1])
    assert constraint == 91
    npt.assert_array_equal(nz, [0, 1, 2, 3, 5, 6, 7])

    # test getting a column by name
    key, col, constraint, nz = freq_wrap.get_column((7, 'pink'))
    assert key == (7, 'pink')
    npt.assert_array_equal(col, [1, 1, 2, 1, 1, 2, 1])
    assert constraint == 91
    npt.assert_array_equal(nz, [0, 1, 2, 3, 5, 6, 7])
