import numpy as np
import pandas as pd


def calculate_constraints(
        marginals, joint_dist, tolerance=1e-3, max_iterations=1000):
    """
    Calculate constraints on household or person classes using
    single category marginals and the observed class proportions
    in a population sample.

    Constraints are calculated via an iterative proportional fitting
    procedure.

    Parameters
    ----------
    marginals : pandas.Series
        The total count of each observed subcategory tracked.
        This should have a pandas.MultiIndex with the outer level containing
        high-level category descriptions and the inner level containing
        the individual subcategory breakdowns.
    joint_dist : pandas.Series
        The observed counts of each household or person class in some sample.
        The index will be a pandas.MultiIndex with a level for each observed
        class in the sample. The levels should be named for ease of
        introspection.
    tolerance : float, optional
        The condition for stopping the IPF procedure. If the change in
        constraints is less than or equal to this value after an iteration
        the calculations are stopped.
    max_iterations : int, optional
        Maximum number of iterations to do before stopping and raising
        an exception.

    Returns
    -------
    constraints : pandas.Series
        Will have the index of `joint_dist` and contain the desired
        totals for each class.
    iterations : int
        Number of iterations performed.

    """
    flat_joint_dist = joint_dist.reset_index()

    constraints = joint_dist.values.copy().astype('float')
    prev_constraints = constraints.copy()
    prev_constraints += tolerance  # ensure we run at least one iteration

    def calc_diff(x, y):
        return np.abs(x - y).sum()

    iterations = 0

    list_of_loc = [
        ((flat_joint_dist[idx[0]] == idx[1]).values, marginals[idx])
        for idx in marginals.index
    ]

    while calc_diff(constraints, prev_constraints) > tolerance:
        prev_constraints[:] = constraints

        for loc, target in list_of_loc:
            constraints[loc] *= target / constraints[loc].sum()

        iterations += 1

        if iterations > max_iterations:
            raise RuntimeError(
                'Maximum number of iterations reached during IPF: {}'.format(
                    max_iterations))

    return pd.Series(constraints, index=joint_dist.index), iterations
