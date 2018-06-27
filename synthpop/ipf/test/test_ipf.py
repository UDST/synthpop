import pandas as pd
import pytest
from pandas.util import testing as pdt

from .. import ipf


def test_trivial_ipf():
    # Test IPF in a situation where the desired totals and observed
    # sample have the same proportion and there is only one super-category.
    midx = pd.MultiIndex.from_product([('cat_owner',), ('yes', 'no')])
    marginals = pd.Series([60, 40], index=midx)
    joint_dist = pd.Series(
        [6, 4], index=pd.Series(['yes', 'no'], name='cat_owner'))

    expected = pd.Series(marginals.values, index=joint_dist.index)
    constraints, iterations = ipf.calculate_constraints(marginals, joint_dist)

    pdt.assert_series_equal(constraints, expected, check_dtype=False)
    assert iterations == 2


def test_larger_ipf():
    # Test IPF with some data that's slightly more meaningful,
    # but for which it's harder to know the actual correct answer.
    marginal_midx = pd.MultiIndex.from_tuples(
        [('cat_owner', 'yes'),
         ('cat_owner', 'no'),
         ('car_color', 'blue'),
         ('car_color', 'red'),
         ('car_color', 'green')])
    marginals = pd.Series([60, 40, 50, 30, 20], index=marginal_midx)
    joint_dist_midx = pd.MultiIndex.from_product(
        [('yes', 'no'), ('blue', 'red', 'green')],
        names=['cat_owner', 'car_color'])
    joint_dist = pd.Series([8, 4, 2, 5, 3, 2], index=joint_dist_midx)

    expected = pd.Series(
        [31.78776824, 17.77758309, 10.43464846,
         18.21223176, 12.22241691, 9.56535154],
        index=joint_dist.index)
    constraints, _ = ipf.calculate_constraints(marginals, joint_dist)

    pdt.assert_series_equal(constraints, expected, check_dtype=False)

    with pytest.raises(RuntimeError):
        ipf.calculate_constraints(marginals, joint_dist, max_iterations=2)
