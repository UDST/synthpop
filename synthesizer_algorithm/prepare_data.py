import os
import pandas as pd
import numpy as np
import synthesizer_algorithm.adjusting_sample_joint_distribution as jd
import drawing_households
import synthesizer_algorithm.pseudo_sparse_matrix as ps
import time
from scipy import sparse

# this function adds the unique_ids that were previously added by adjusting_sample_joint_distribution.create_update_string and add_unique_id
def prepare_data(data_dir, hh_sample_file, per_sample_file, hh_marginals_file, per_marginals_file):
    os.chdir(data_dir)
    hh_sample = pd.read_csv(hh_sample_file, header = 0)
    hh_sample = hh_sample.astype('int')
    hh_vars = np.array((hh_sample.columns)[4:]) # identifies the household control variables
    hh_var_list = list(hh_sample.columns[4:])
    hh_dims = np.array((hh_sample.max())[4:]) # identifies number of categories per household control variable
    hhld_units = len(hh_sample.index)  # identifies the number of housing units to build the Master Matrix
    hh_sample['group_id'] = ''
    for var in hh_var_list:
        hh_sample[var + '_str'] = hh_sample[var].astype('str')
        hh_sample.group_id = hh_sample.group_id + hh_sample[var + '_str']
        hh_sample = hh_sample.drop([var + '_str'], axis=1)
    hh_marginals = pd.read_csv(hh_marginals_file, header = 0)
    hhid = hh_sample.groupby(['group_id'], as_index=False)['state'].min()
    hhid['hhld_uniqueid'] = hhid.index + 1
    hhid = hhid[['group_id', 'hhld_uniqueid']]
    hh_sample = pd.merge(hh_sample, hhid, how='left', left_on='group_id', right_on='group_id')
    hh_sample = hh_sample.drop('group_id', axis=1)

    per_sample = pd.read_csv(per_sample_file, header = 0)
    per_vars = list(per_sample.columns)[5:] # identifies the person control variables
    per_sample['group_id'] = ''
    for var in per_vars:
        per_sample[var + '_str'] = per_sample[var].astype('str')
        per_sample.group_id = per_sample.group_id + per_sample[var + '_str']
        per_sample = per_sample.drop([var + '_str'], axis=1)
    pid = per_sample.groupby(['group_id'], as_index=False)['state'].min()
    pid['person_uniqueid'] = pid.index + 1
    pid = pid[['group_id', 'person_uniqueid']]
    per_sample = pd.merge(per_sample, pid, how='left', left_on='group_id', right_on='group_id')
    per_sample = per_sample.drop('group_id', axis=1)
    per_dims = np.array(per_sample.astype('int').max())[5:]
    per_vars_dims = dict(zip(per_vars, per_dims))
    per_marginals = pd.read_csv(per_marginals_file, header = 0)
    
    matrix = ps.populate_master_matrix(hh_dims, per_dims, hhld_units, hh_sample)
    sparse_matrix = ps.pseudo_sparse_matrix(data_dir, hh_sample)
    index_matrix = ps.generate_index_matrix(sparse_matrix)
    
    housing_synthetic_data = pd.DataFrame(columns=['state','county','tract','bg','hhid','serialno','frequency','hhuniqueid'])
    person_synthetic_data = pd.DataFrame(columns=['state','county','tract','bg','hhid','serialno','pnum','frequency','personuniqueid'])
    
    performance_statistics = pd.DataFrame(columns=['state','county','tract','bg','chivalue','pvalue','synpopiter','heuriter','aardvalue'])
    
    hhld_0_joint_dist = jd.hhld_0_joint_dist(hh_sample, hh_var_list)
    person_0_joint_dist = jd.person_0_joint_dist(per_sample, per_vars)

    return {'matrix':matrix, 'sparse_matrix':sparse_matrix, 'index_matrix':index_matrix, 'housing_synthetic_data':housing_synthetic_data, 'person_synthetic_data':person_synthetic_data, 'hhld_0_joint_dist':hhld_0_joint_dist, 'person_0_joint_dist':person_0_joint_dist}

