import itertools
import os

import numpy as np
import pandas as pd


def prepare_data(data_dir, hh_sample_file, hh_marginals_file, per_sample_file, per_marginals_file):
    os.chdir(data_dir)
    hh_sample = pd.read_csv(hh_sample_file, header = 0)
    hh_sample = hh_sample.astype('int')
    hh_vars = np.array((hh_sample.columns)[4:])
    hh_var_list = list(hh_sample.columns[4:])
    hh_dims = np.array((hh_sample.max())[4:])
    hh_var_dims = zip(hh_var_list, list(hh_dims))
    hhld_units = len(hh_sample.index)
    hh_sample['group_id'] = ''
    for var in hh_var_list:
        hh_sample[var + '_str'] = hh_sample[var].astype('str')
        hh_sample.group_id = hh_sample.group_id + hh_sample[var + '_str']
        hh_sample = hh_sample.drop([var + '_str'], axis=1)
    vals = list(hh_dims)
    hhid_comb = []
    for item in itertools.product(*[range(1,x+1) for x in vals]):
        hhid_comb.append(item)
    hhid_var = pd.DataFrame(data=hhid_comb, columns=hh_var_list)
    hhid_var['hhld_uniqueid'] = hhid_var.index + 1
    hhid_var['group_id'] = ''
    for val in hh_var_list:
        hhid_var[val+'_str'] = hhid_var[val].astype('str')
        hhid_var['group_id'] = hhid_var['group_id'] + hhid_var[val+'_str']
        hhid_var = hhid_var.drop([val+'_str'], axis=1)
    hhid = hhid_var[['group_id','hhld_uniqueid']]
    agg = dict.fromkeys(hh_var_list,'min')
    hh_sample = pd.merge(hh_sample, hhid, how='left', left_on='group_id', right_on='group_id')
    hh_marginals = pd.read_csv(hh_marginals_file, header = 0)

    per_sample = pd.read_csv(per_sample_file, header = 0)
    per_vars = list(per_sample.columns)[5:]
    per_dims = np.array(per_sample.astype('int').max())[5:]
    per_sample['group_id'] = ''
    for var in per_vars:
        per_sample[var + '_str'] = per_sample[var].astype('str')
        per_sample.group_id = per_sample.group_id + per_sample[var + '_str']
        per_sample = per_sample.drop([var + '_str'], axis=1)
    vals = list(per_dims)
    per_comb = []
    for item in itertools.product(*[range(1,x+1) for x in vals]):
        per_comb.append(item)
    pid_var = pd.DataFrame(data=per_comb, columns=per_vars)
    pid_var['person_uniqueid'] = pid_var.index + 1
    pid_var['group_id'] = ''
    for val in per_vars:
        pid_var[val+'_str'] = pid_var[val].astype('str')
        pid_var['group_id'] = pid_var['group_id'] + pid_var[val+'_str']
        pid_var = pid_var.drop([val+'_str'], axis=1)
    pid = pid_var[['group_id','person_uniqueid']]
    p_agg = dict.fromkeys(per_vars,'min')
    per_sample = pd.merge(per_sample, pid, how='left', left_on='group_id', right_on='group_id')
    per_vars_dims = dict(zip(per_vars, per_dims))
    per_marginals = pd.read_csv(per_marginals_file, header = 0)
    return hh_sample, hh_marginals, per_sample, per_marginals, hh_var_list, per_vars, hhld_units, hh_dims, per_dims, hhid_var, pid_var
    
def variables(hh_var_list, hh_dims, per_vars, per_dims):
    hh_varcorrdict = {}
    for var in range(len(hh_var_list)):
        for i in range(1,hh_dims[var]+1):
            hh_varcorrdict[hh_var_list[var]+str(i)] = [hh_var_list[var], i]
    per_varcorrdict = {}
    for var in range(len(per_vars)):
        for i in range(1,per_dims[var]+1):
            per_varcorrdict[per_vars[var]+str(i)] = [per_vars[var], i]
    return hh_varcorrdict, per_varcorrdict
