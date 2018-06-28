import pytest

from synthpop.synthesizer import *
from synthpop.recipes.starter import Starter
from synthpop.recipes.starter2 import Starter as Starter2


@pytest.fixture
def key():
    return "827402c2958dcf515e4480b7b2bb93d1025f9389"


def test_starter(key):
    st = Starter(key, "CA", "Alpine County")
    # just run it for now
    synthesize_all(st, num_geogs=1)


# no synthesizer bc it's too memory intensive for travis
def test_starter2(key):
    Starter2(key, "CA", "Alpine County")
