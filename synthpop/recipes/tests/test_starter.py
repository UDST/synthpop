import pytest
from ...synthesizer import *
from ..starter import Starter


@pytest.fixture
def key():
    return "827402c2958dcf515e4480b7b2bb93d1025f9389"


def test_starter(key):
    enable_logging()
    st = Starter(key, "TX", "Travis County")
    all_households, all_persons, fit_quality = synthesize_all(st)

    hh_file_name = "result/household_{}_{}.csv".format(geog_id.state, geog_id.county)
    people_file_name = "result/people_{}_{}.csv".format(geog_id.state, geog_id.county)

    all_households.to_csv(hh_file_name, index=None, header=True)
    all_persons.to_csv(people_file_name, index=None, header=True)
