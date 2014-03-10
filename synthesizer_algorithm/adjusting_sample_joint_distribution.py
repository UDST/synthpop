import os
import pandas as pd
import numpy as np
import synthesizer_algorithm.drawing_households
import synthesizer_algorithm.pseudo_sparse_matrix
import time
from scipy import sparse

def hhld_0_joint_dist(hh_sample, hh_var_list):
    hhld_0_joint_dist = hh_sample
    hhld_0_joint_dist['frequency'] = 1
    agg = dict.fromkeys(hh_var_list,'min')
    agg['frequency'] = 'sum'
    hhld_0_joint_dist = hhld_0_joint_dist.groupby('hhld_uniqueid', as_index=False).agg(agg)
    hhld_0_joint_dist['pumano'] = 0
    hhld_0_joint_dist['tract'] = 0
    hhld_0_joint_dist['bg'] = 0
    cols = ['pumano','tract','bg']
    cols.extend(hh_var_list)
    cols.extend(['frequency','hhld_uniqueid'])
    hhld_0_joint_dist = hhld_0_joint_dist[cols]
    return hhld_0_joint_dist
    
def person_0_joint_dist(per_sample, per_vars):
    person_0_joint_dist = per_sample
    person_0_joint_dist['frequency'] = 1
    agg = dict.fromkeys(per_vars,'min')
    agg['frequency'] = 'sum'
    person_0_joint_dist = person_0_joint_dist.groupby('person_uniqueid', as_index=False).agg(agg)
    person_0_joint_dist['pumano'] = 0
    person_0_joint_dist['tract'] = 0
    person_0_joint_dist['bg'] = 0
    cols = ['pumano','tract','bg']
    cols.extend(per_vars)
    cols.extend(['frequency','person_uniqueid'])
    person_0_joint_dist = person_0_joint_dist[cols]
    return person_0_joint_dist

def create_joint_dist():
    housing_synthetic_data = pd.DataFrame(columns=['state','county','tract','bg','hhid','serialno','frequency','hhuniqueid'])
    person_synthetic_data = pd.DataFrame(columns=['state','county','tract','bg','hhid','serialno','pnum','frequency','personuniqueid'])

