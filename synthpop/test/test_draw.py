import numpy as np
import numpy.testing as npt
import pytest

from .. import draw


@pytest.fixture
def seed(request):
    current = np.random.get_state()

    def fin():
        np.random.set_state(current)
    request.addfinalizer(fin)

    np.random.seed(0)


@pytest.fixture
def index():
    return np.array(['v', 'w', 'x', 'y', 'z'], dtype=np.str_)


@pytest.fixture
def weights():
    return np.array([1, 2, 3, 4, 5])


@pytest.fixture
def num():
    return 10


def test_simple_draw(index, weights, num, seed):
    drawn_indexes = draw.simple_draw(num, weights, index)

    npt.assert_array_equal(
        drawn_indexes, ['y', 'z', 'y', 'y', 'y', 'y', 'y', 'z', 'z', 'x'])
