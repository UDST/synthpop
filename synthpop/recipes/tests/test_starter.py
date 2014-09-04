import pytest
from .. import starter
from ...census_helpers import Census


@pytest.fixture
def c():
    return Census("827402c2958dcf515e4480b7b2bb93d1025f9389")


def test_starter(c):
    hmarg, pmarg, h_jd, p_jd = \
        starter.marginals_and_joint_distribution(c,
                                                 "CA",
                                                 "San Francisco County",
                                                 "030600")


def test_starter_no_tract(c):
    hmarg, pmarg, h_jd, p_jd = \
        starter.marginals_and_joint_distribution(c,
                                                 "CA",
                                                 "San Francisco County")
