import pandas as pd


class FrequencyTable(object):
    """
    Utility class for holding the frequency table used in popgen.

    The main benefit is that instead of storing every row of every column
    you can store just the non-zero elements of each column as a Series.

    Parameters
    ----------
    index : sequence
        The full index of household IDs.
    household_cols : dict, optional
        Dictionary of household category columns (as Series).
    person_cols : dict, optional
        Dictionary of person category columns (as Series).

    """
    def __init__(self, index, household_cols=None, person_cols=None):
        self.index = pd.Index(index)
        self.household = household_cols or {}
        self.person = person_cols or {}

    def itercols(self):
        """
        Iterate over both household and person columns, household first.
        Yields 3-tuples of ('person|household', column name, column series).

        """
        for name in self.household:
            yield 'household', name, self.household[name]

        for name in self.person:
            yield 'person', name, self.person[name]

    def __getitem__(self, key):
        if key == 'household':
            return self.household
        elif key == 'person':
            return self.person
        else:
            raise KeyError('Key not found: {}'.format(key))

    def __len__(self):
        return len(self.index)

    @property
    def ncols(self):
        """
        Number of combined household and person columns.

        """
        return len(self.household) + len(self.person)
