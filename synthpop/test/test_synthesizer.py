import pandas as pd
from pandas.util import testing as pdt

from .. import synthesizer


def test_execute_draw():
    hh_df = pd.DataFrame(
        {'a': range(5),
         'b': range(5, 10),
         'serialno': [11, 22, 33, 44, 55]},
        index=pd.Index(['a', 'b', 'c', 'd', 'e'], name='hh_id'))

    pp_df = pd.DataFrame(
        {'x': range(100, 110),
         'y': range(110, 120),
         'serialno': [22, 33, 11, 55, 22, 33, 44, 55, 11, 33]},
        index=pd.Index(['q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']))

    indexes = ['c', 'a', 'd', 'e', 'a', 'c', 'e', 'e', 'a', 'c', 'e']

    synth_hh, synth_pp = synthesizer.execute_draw(indexes, hh_df, pp_df)

    pdt.assert_index_equal(synth_hh.index, pd.Index(range(11)))
    pdt.assert_series_equal(
        synth_hh.serialno,
        pd.Series([33, 11, 44, 55, 11, 33, 55, 55, 11, 33, 55]))
    assert list(synth_hh.columns) == ['a', 'b', 'serialno']

    pdt.assert_index_equal(synth_pp.index, pd.Index(range(24)))
    pdt.assert_series_equal(
        synth_pp.serialno,
        pd.Series(([33] * 9) + ([11] * 6) + ([55] * 8) + [44]))
    pdt.assert_series_equal(
        synth_pp.hh_id,
        pd.Series(
            ([0, 5, 9] * 3) + ([1, 4, 8] * 2) + ([3, 6, 7, 10] * 2) + [2]))
