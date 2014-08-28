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
    df.index.name = "id"
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

    return sample_df, category_df
