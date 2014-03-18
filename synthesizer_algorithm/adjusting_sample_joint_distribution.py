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
    hhld_0_joint_dist['county'] = 0
    hhld_0_joint_dist['tract'] = 0
    hhld_0_joint_dist['bg'] = 0
    cols = ['pumano','county','tract','bg']
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
    person_0_joint_dist['county'] = 0
    person_0_joint_dist['tract'] = 0
    person_0_joint_dist['bg'] = 0
    cols = ['pumano','county','tract','bg']
    cols.extend(per_vars)
    cols.extend(['frequency','person_uniqueid'])
    person_0_joint_dist = person_0_joint_dist[cols]
    
def create_person_joint_dist(data_dir, geocorr_file, per_vars, per_sample):
    geocorr = pd.read_csv(geocorr_file, header=0)
    pumas = list(geocorr.pumano.unique())
    pumas_jd = ['person_'+str(puma)+'_joint_dist' for puma in pumas]
    pumas = zip(pumas, pumas_jd)
    person_joint_dists = {}
    for puma, puma_jd in pumas:
        agg = dict.fromkeys(per_vars,'min')
        agg['frequency'] = 'sum'
        agg['pumano'] = 'min'
        jd = per_sample[per_sample.pumano==puma]
        if len(jd.index) > 0:
            jd['frequency'] = 1
            jd = jd.groupby('person_uniqueid', as_index=False).agg(agg)
            jd = pd.merge(pid_var, jd, how='left', on='person_uniqueid', suffixes=('','_x'))
            jd.frequency = jd.frequency.fillna(0)
            jd.pumano = jd.pumano.fillna(puma)
            jd['county'] = 0
            jd['tract'] = 0
            jd['bg'] = 0
            cols = ['pumano','county','tract','bg']
            cols.extend(per_vars)
            cols.extend(['frequency','person_uniqueid'])
            jd = jd[cols]
            person_joint_dists[puma_jd] = jd
    return person_joint_dists
    
def create_hhld_joint_dist():
    geocorr = pd.read_csv(geocorr_file, header=0)
    pumas = list(geocorr.pumano.unique())
    pumas_jd = ['hhld_'+str(puma)+'_joint_dist' for puma in pumas]
    pumas = zip(pumas, pumas_jd)
    hhld_joint_dists = {}
    for puma, puma_jd in pumas:
        agg = dict.fromkeys(hh_var_list,'min')
        agg['frequency'] = 'sum'
        agg['pumano'] = 'min'
        jd = hh_sample[hh_sample.pumano==puma]
        if len(jd.index) > 0:
            jd['frequency'] = 1
            jd = jd.groupby('hhld_uniqueid', as_index=False).agg(agg)
            jd = pd.merge(hhid_var, jd, how='left', on='hhld_uniqueid', suffixes=('','_x'))
            jd.frequency = jd.frequency.fillna(0)
            jd.pumano = jd.pumano.fillna(puma)
            jd['county'] = 0
            jd['tract'] = 0
            jd['bg'] = 0
            cols = ['pumano','county','tract','bg']
            cols.extend(hh_var_list)
            cols.extend(['frequency','hhld_uniqueid'])
            jd = jd[cols]
            hhld_joint_dists[puma_jd] = jd
    return hhld_joint_dists

def hhld_estimated_constraint(hhld_joint_dists, geogs):    
    hhld_estimated_constraint = {}
    for name in hhld_joint_dists:
        puma_id = int(name[5:8])
        puma_name = 'ec_' + str(puma_id)
        hhld_joint_dist = hhld_joint_dists[name]
        geogs = zip(list(geocorr.county[geocorr.pumano == puma_id]), list(geocorr.tract[geocorr.pumano == puma_id]), list(geocorr.bg[geocorr.pumano == puma_id]))
        hhld_estimated_constraint[puma_name] = {}
        for county, tract, bg in geogs:
            bg_jd = hhld_joint_dist[(hhld_joint_dist.county==county)&(hhld_joint_dist.tract==tract)&(hhld_joint_dist.bg==bg)]
            bg_jd = bg_jd.sort(columns=hh_var_list)
            bg_jd = bg_jd[['frequency']]
            ec_id = '%s, %s, %s' %(county, tract, bg)
            hhld_estimated_constraint[puma_name][ec_id] = bg_jd
    return hhld_estimated_constraint
    
def person_estimated_constraint(person_joint_dists, geogs):
    person_estimated_constraint = {}
    for name in person_joint_dists:
        puma_id = int(name[7:10])
        puma_name = 'ec_' + str(puma_id)
        person_joint_dist = person_joint_dists[name]
        geogs = zip(list(geocorr.county[geocorr.pumano == puma_id]), list(geocorr.tract[geocorr.pumano == puma_id]), list(geocorr.bg[geocorr.pumano == puma_id]))
        person_estimated_constraint[puma_name] = {}
        for county, tract, bg in geogs:
            bg_jd = person_joint_dist[(person_joint_dist.county==county)&(person_joint_dist.tract==tract)&(person_joint_dist.bg==bg)]
            bg_jd = bg_jd.sort(columns=per_vars)
            bg_jd = bg_jd[['frequency']]
            ec_id = '%s, %s, %s' %(county, tract, bg)
            person_estimated_constraint[puma_name][ec_id] = bg_jd
    return person_estimated_constraint

def create_joint_dist():
    housing_synthetic_data = pd.DataFrame(columns=['state','county','tract','bg','hhid','serialno','frequency','hhuniqueid'])
    person_synthetic_data = pd.DataFrame(columns=['state','county','tract','bg','hhid','serialno','pnum','frequency','personuniqueid'])

