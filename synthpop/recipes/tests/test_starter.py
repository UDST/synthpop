import pytest
from ...synthesizer import *
from ..starter import Starter


@pytest.fixture
def key():
    return "827402c2958dcf515e4480b7b2bb93d1025f9389"


def test_starter(key):
    st = Starter(key, "CA", "Napa County")
    synthesize_all(st, num_geogs=1)
