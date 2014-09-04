import pytest
from ..starter import Starter
from ...synthesizer import SynthPop

@pytest.fixture
def key():
    return "827402c2958dcf515e4480b7b2bb93d1025f9389"

def test_starter(key):
    st = Starter(key, "CA", "Napa County")
    sp = SynthPop(st)
    # just run it for now
    sp.synthesize_all(debug=True)
