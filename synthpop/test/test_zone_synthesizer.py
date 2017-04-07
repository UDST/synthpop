import os
import pytest
import pandas as pd

import synthpop.zone_synthesizer as zs


@pytest.fixture
def hh_marg():
    fname = os.path.join(os.path.dirname(__file__),
                         'test_data/hh_marginals.csv')
    return fname


@pytest.fixture
def p_marg():
    fname = os.path.join(os.path.dirname(__file__),
                         'test_data/person_marginals.csv')
    return fname


@pytest.fixture
def hh_sample():
    fname = os.path.join(os.path.dirname(__file__),
                         'test_data/household_sample.csv')
    return fname


@pytest.fixture
def p_sample():
    fname = os.path.join(os.path.dirname(__file__),
                         'test_data/person_sample.csv')
    return fname


def test_run(hh_marg, p_marg, hh_sample, p_sample):
    hh_marg, p_marg, hh_sample, p_sample, xwalk = zs.load_data(hh_marg,
                                                               p_marg,
                                                               hh_sample,
                                                               p_sample)
    all_households, all_persons, all_stats = zs.synthesize_all_zones(hh_marg,
                                                                     p_marg,
                                                                     hh_sample,
                                                                     p_sample,
                                                                     xwalk)


def test_run_multi(hh_marg, p_marg, hh_sample, p_sample):
    hhm, pm, hhs, ps, xwalk = zs.load_data(hh_marg, p_marg,
                                           hh_sample, p_sample)
    all_persons, all_households, all_stats = zs.multiprocess_synthesize(hhm, pm,
                                                                        hhs, ps,
                                                                        xwalk)
