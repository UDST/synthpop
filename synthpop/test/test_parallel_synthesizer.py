import pytest

from synthpop.synthesizer import synthesize_all_in_parallel
from synthpop.recipes.starter import Starter


@pytest.fixture
def key():
    return "827402c2958dcf515e4480b7b2bb93d1025f9389"


def test_parallel_synth(key):
    num_geogs = 2
    st = Starter(key, "CA", "Napa County")
    _, _, fit = synthesize_all_in_parallel(st, num_geogs=num_geogs)

    for bg_named_tuple in list(fit.keys()):
        assert bg_named_tuple.state == '06'
        assert bg_named_tuple.county == '055'
        assert bg_named_tuple.tract == '200201'
        assert bg_named_tuple.block_group in [
            str(x) for x in list(range(1, num_geogs + 1))]

    for fit_named_tuple in list(fit.values()):
        assert fit_named_tuple.people_chisq > 10
        assert fit_named_tuple.people_p < 0.5
