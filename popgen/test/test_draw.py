import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from pandas.util import testing as pdt

from .. import draw


@pytest.fixture
def seed(request):
    current = np.random.get_state()

    def fin():
        np.random.set_state(current)
    request.addfinalizer(fin)

    np.random.seed(0)


@pytest.fixture
def df():
    return pd.DataFrame(
        {'a': [1, 2, 3, 4, 5]},
        index=['v', 'w', 'x', 'y', 'z'])


@pytest.fixture
def weights(df):
    return df.a.copy()


@pytest.fixture
def num():
    return 10


def test_draw_no_weights(df, num, seed):
    draws, drawn_indexes = draw.draw(df, num)

    expected = pd.DataFrame(
        {'a': [3, 4, 4, 3, 3, 4, 3, 5, 5, 2]},
        index=range(num))

    npt.assert_array_equal(
        drawn_indexes, ['x', 'y', 'y', 'x', 'x', 'y', 'x', 'z', 'z', 'w'])
    pdt.assert_frame_equal(draws, expected)


def test_draw_with_weights(df, weights, num, seed):
    draws, drawn_indexes = draw.draw(df, num, weights)

    expected = pd.DataFrame(
        {'a': [4, 5, 4, 4, 4, 4, 4, 5, 5, 3]},
        index=range(num))

    npt.assert_array_equal(
        drawn_indexes, ['y', 'z', 'y', 'y', 'y', 'y', 'y', 'z', 'z', 'x'])
    pdt.assert_frame_equal(draws, expected)
