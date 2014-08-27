import pandas as pd


def aggregate_categorize(df, eval_d, index_cols=None):
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


def aggregate_sum_accross_category(df, subtract_mean=True):
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
