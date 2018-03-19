import pytest
from ...synthesizer import *
from ..starter2 import Starter as Starter2


@pytest.fixture
def key():
    return "827402c2958dcf515e4480b7b2bb93d1025f9389"


# commented out as it is to slow for travis
def test_starter2(key):
    st = Starter2(key, "CA", "Napa County")
    # just run it for now
    synthesize_all(st, num_geogs=1)
    