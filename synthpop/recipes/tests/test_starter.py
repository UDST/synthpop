import pytest
from synthpop.synthesizer import *
from synthpop.recipes.starter import Starter
from synthpop.recipes.starter2 import Starter as Starter2


@pytest.fixture
def key():
    return "827402c2958dcf515e4480b7b2bb93d1025f9389"


def test_starter(key):
    st = Starter(key, "CA", "Napa County")
    # just run it for now
    synthesize_all(st, num_geogs=1)


# commented out if it is to slow for travis
def test_starter2(key):
    st = Starter2(key, "CA", "Alpine County")
    # just run it for now
    synthesize_all(st, num_geogs=1)
