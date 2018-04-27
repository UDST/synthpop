import pytest
from ...synthesizer import *
from ..starter2 import Starter as Starter2


@pytest.fixture
def key():
    return "bfa6b4e541243011fab6307a31aed9e91015ba90"


def test_starter2(key):
    st = Starter2(key, "CA", "Napa County")
    synthesize_all(st, num_geogs=1)
