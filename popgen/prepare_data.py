import itertools
import os

import numpy as np
import pandas as pd

from . import adjusting_sample_joint_distribution as jd
from . import pseudo_sparse_matrix as ps


# this function adds the unique_ids that were previously added by
# adjusting_sample_joint_distribution.create_update_string and
# add_unique_id
def prepare_data(data_dir, hh_sample_file, per_sample_file,
                 hh_marginals_file, per_marginals_file):
    os.chdir(data_dir)
    hh_sample = pd.read_csv(hh_sample_file, header=0)
    hh_sample = hh_sample.astype('int')
    # identifies the household control variables
    hh_vars = np.array((hh_sample.columns)[4:])
    hh_var_list = list(hh_sample.columns[4:])
    # identifies number of categories per household control variable
    hh_dims = np.array((hh_sample.max())[4:])
    hh_var_dims = zip(hh_var_list, list(hh_dims))
    # identifies the number of housing units to build the Master Matrix
    hhld_units = len(hh_sample.index)
    hh_sample['group_id'] = ''
    for var in hh_var_list:
        hh_sample[var + '_str'] = hh_sample[var].astype('str')
        hh_sample.group_id = hh_sample.group_id + hh_sample[var + '_str']
        hh_sample = hh_sample.drop([var + '_str'], axis=1)
    vals = list(hh_dims)
    hhid_comb = []
    for item in itertools.product(*[range(1, x + 1) for x in vals]):
        hhid_comb.append(item)
    hhid_var = pd.DataFrame(data=hhid_comb)
    hhid_var['hhld_uniqueid'] = hhid_var.index + 1
    hhid_var['group_id'] = ''
    for val in range(len(vals)):
        hhid_var[str(val) + '_str'] = hhid_var[val].astype('str')
        hhid_var['group_id'] = hhid_var[
            'group_id'] + hhid_var[str(val) + '_str']
        hhid_var = hhid_var.drop([str(val) + '_str'], axis=1)
    hhid = hhid_var[['group_id', 'hhld_uniqueid']]
    agg = dict.fromkeys(hh_var_list, 'min')
    # hhid_var = hh_sample.groupby(['group_id'], as_index=False).agg(agg)
    # hhid_var['hhld_uniqueid'] = hhid_var.index + 1
    # hhid = hhid_var[['group_id', 'hhld_uniqueid']]
    hh_sample = pd.merge(
        hh_sample,
        hhid,
        how='left',
        left_on='group_id',
        right_on='group_id')
    hh_sample = hh_sample.drop('group_id', axis=1)
    hh_marginals = pd.read_csv(hh_marginals_file, header=0)

    per_sample = pd.read_csv(per_sample_file, header=0)
    # identifies the person control variables
    per_vars = list(per_sample.columns)[5:]
    per_dims = np.array(per_sample.astype('int').max())[5:]
    per_sample['group_id'] = ''
    for var in per_vars:
        per_sample[var + '_str'] = per_sample[var].astype('str')
        per_sample.group_id = per_sample.group_id + per_sample[var + '_str']
        per_sample = per_sample.drop([var + '_str'], axis=1)
    vals = list(per_dims)
    per_comb = []
    for item in itertools.product(*[range(1, x + 1) for x in vals]):
        per_comb.append(item)
    pid_var = pd.DataFrame(data=per_comb)
    pid_var['person_uniqueid'] = pid_var.index + 1
    pid_var['group_id'] = ''
    for val in range(len(vals)):
        pid_var[str(val) + '_str'] = pid_var[val].astype('str')
        pid_var['group_id'] = pid_var['group_id'] + pid_var[str(val) + '_str']
        pid_var = pid_var.drop([str(val) + '_str'], axis=1)
    pid = pid_var[['group_id', 'person_uniqueid']]
    p_agg = dict.fromkeys(per_vars, 'min')
    # pid_var = per_sample.groupby(['group_id'], as_index=False).agg(p_agg)
    # pid_var['person_uniqueid'] = pid_var.index + 1
    # pid = pid_var[['group_id', 'person_uniqueid']]
    per_sample = pd.merge(
        per_sample,
        pid,
        how='left',
        left_on='group_id',
        right_on='group_id')
    per_sample = per_sample.drop('group_id', axis=1)
    per_vars_dims = dict(zip(per_vars, per_dims))
    per_marginals = pd.read_csv(per_marginals_file, header=0)

    varcorrdict = {}
    for var in range(len(hh_var_list)):
        for i in range(1, hh_dims[var] + 1):
            varcorrdict[hh_var_list[var] + str(i)] = [hh_var_list[var], i]

    matrix = ps.populate_master_matrix(
        hh_dims,
        per_dims,
        hhld_units,
        hh_sample,
        per_sample)
    sparse_matrix = ps.pseudo_sparse_matrix(
        data_dir,
        hh_sample,
        hh_dims,
        per_sample)
    sparse_matrix1, index_matrix = ps.generate_index_matrix(sparse_matrix)

    housing_synthetic_data = pd.DataFrame(
        columns=[
            'state',
            'county',
            'tract',
            'bg',
            'hhid',
            'serialno',
            'frequency',
            'hhuniqueid'])
    person_synthetic_data = pd.DataFrame(
        columns=[
            'state',
            'county',
            'tract',
            'bg',
            'hhid',
            'serialno',
            'pnum',
            'frequency',
            'personuniqueid'])

    performance_statistics = pd.DataFrame(
        columns=[
            'state',
            'county',
            'tract',
            'bg',
            'chivalue',
            'pvalue',
            'synpopiter',
            'heuriter',
            'aardvalue'])

    hhld_0_joint_dist = jd.hhld_0_joint_dist(hh_sample, hh_var_list)
    person_0_joint_dist = jd.person_0_joint_dist(per_sample, per_vars)

    return {
        'matrix': matrix,
        'sparse_matrix': sparse_matrix,
        'sparse_matrix1': sparse_matrix1,
        'index_matrix': index_matrix,
        'housing_synthetic_data': housing_synthetic_data,
        'person_synthetic_data': person_synthetic_data,
        'hhld_0_joint_dist': hhld_0_joint_dist,
        'person_0_joint_dist': person_0_joint_dist
    }
