import pandas as pd
import itertools


def categorize(df, eval_d, index_cols=None):
    cat_df = pd.DataFrame(index=df.index)

    for index, expr in eval_d.iteritems():
        cat_df[index] = df.eval(expr)

    if index_cols is not None:
        cat_df[index_cols] = df[index_cols]
        cat_df = cat_df.set_index(index_cols)

    cat_df.columns = pd.MultiIndex.from_tuples(cat_df.columns,
                                               names=['cat_name', 'cat_value'])

    cat_df = cat_df.sort_index(axis=1)
    return cat_df


def sum_accross_category(df, subtract_mean=True):
    """
    This is a convenience function to sum the categorical values for each
    category - the mean across each category is then subtracted so all the
    cells in the table should be close to zero.  The reason why it's not
    exactly zero is because of rounding errors in the scaling of any tract
    variables down to block group variables
    """
    df = df.stack(level=1).fillna(0).groupby(level=0).sum()
    if subtract_mean:
        df = df.sub(df.mean(axis=1), axis="rows")
    return df


def category_combinations(index):
    """
    THis method converts a hierarchical multindex of category names and
    category values and converts to the cross-product of all possible
    category combinations.
    """
    d = {}
    for cat_name, cat_value in index:
        d.setdefault(cat_name, [])
        d[cat_name].append(cat_value)
    for cat_name in d.keys():
        if len(d[cat_name]) == 1:
            del d[cat_name]
    df = pd.DataFrame(list(itertools.product(*d.values())))
    df.columns = cols = d.keys()
    df.index.name = "cat_id"
    df = df.reset_index().set_index(cols)
    return df


def joint_distribution(sample_df, category_df, mapping_functions):

    # set counts to zero
    category_df["frequency"] = 0

    category_names = category_df.index.names
    for name in category_names:
        assert name in mapping_functions, "Every category needs to have a " \
                                          "mapping function with the same " \
                                          "name to define that category for " \
                                          "the pums sample records"
        sample_df[name] = sample_df.apply(mapping_functions[name], axis=1)

    category_df["frequency"] = sample_df.groupby(category_names).size()
    category_df["frequency"] = category_df["frequency"].fillna(0)

    # do the merge to add the category id
    sample_df = pd.merge(sample_df, category_df[["cat_id"]],
                         left_on=category_names, right_index=True)

    return sample_df, category_df


def _frequency_table(sample_df, category_df):
    """
    Take the result that comes out of the method above and turn it in to the
    frequencytable format used by the ipu
    """
    print sample_df.serialno.value_counts()
    df = sample_df.groupby(['hh_id', 'cat_id']).size().\
        unstack().fillna(0)

    # need to manually add in case we missed a whole cat_id in the sample
    for cat_id in category_df.cat_id:
        if cat_id not in df.columns:
            df[cat_id] = 0

    assert len(df.columns) == len(category_df)
    assert df.sum().sum() == len(sample_df)

    return df


def frequency_tables(persons_sample_df, households_sample_df,
                     persons_category_df, households_category_df):

    households_sample_df.index.name = "hh_id"
    households_sample_df = households_sample_df.reset_index().\
        set_index("serialno")

    h_freq_table = _frequency_table(households_sample_df,
                                    households_category_df)

    persons_sample_df = pd.merge(persons_sample_df,
                                 households_sample_df[["hh_id"]],
                                 left_on=["serial_no"], right_index=True,
                                 how="left")

    p_freq_table = _frequency_table(persons_sample_df,
                                    persons_category_df)

    return h_freq_table, p_freq_table
