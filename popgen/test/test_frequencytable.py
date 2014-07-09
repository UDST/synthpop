import itertools
from collections import OrderedDict

import pandas as pd
from pandas.util import testing as pdt

from ..frequencytable import FrequencyTable


def test_frequency_table():
    index = range(5)
    hhcols = OrderedDict()
    hhcols[1] = pd.Series([1, 1, 1], index=[0, 1, 2])
    hhcols[2] = pd.Series([1, 1], index=[3, 4])
    pcols = OrderedDict()
    pcols[1] = pd.Series([1, 2, 2], index=[0, 2, 4])
    pcols[2] = pd.Series([1, 1], index=[1, 3])
    pcols[3] = pd.Series([2, 1, 1], index=[0, 1, 3])

    ft = FrequencyTable(index, household_cols=hhcols, person_cols=pcols)

    assert len(ft) == 5
    pdt.assert_index_equal(ft.index, pd.Index(index))
    pdt.assert_series_equal(ft.household[1], hhcols[1])
    pdt.assert_series_equal(ft['person'][1], pcols[1])

    for (hp, col_name, col), (exp_name, exp_col) in itertools.izip(
            ft.itercols(),
            itertools.chain(hhcols.iteritems(), pcols.iteritems())):
        assert col_name == exp_name
        pdt.assert_series_equal(col, exp_col)
