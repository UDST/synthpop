import os
import pandas as pd
import numpy as np
import synthesizer_algorithm.adjusting_sample_joint_distribution
import synthesizer_algorithm.drawing_households
import time
from scipy import sparse

def populate_master_matrix(hh_dims, per_dims, hhld_units, hh_sample):
    total_cols = 4 + hh_dims.prod() + per_dims.prod()
    total_rows = hhld_units
    matrix = pd.DataFrame(index=np.arange(0,total_rows), columns=np.arange(0, total_cols), dtype='int')
    matrix = matrix.fillna(0)
    result = hh_sample.sort('hhid')[['state','pumano','hhid','serialno','hhld_uniqueid']]
    matrix[0] = np.array(result['state'])
    matrix[1] = np.array(result['pumano'])
    matrix[2] = np.array(result['hhid'])
    matrix[3] = np.array(result['serialno'])
    mat_col = zip(np.array(result.hhid), np.array(result.hhld_uniqueid))
    for hh in mat_col:
        row = hh[0]
        col = hh[1]
        matrix[col+4][matrix[2]==row] = matrix[col+4][matrix[2]==row]+1
    return matrix

def pseudo_sparse_matrix(data_dir, hh_sample):
    sparse_matrix = hh_sample[['pumano','hhld_uniqueid']]
    sparse_matrix['rowno'] = np.arange(0, len(sparse_matrix.index))
    sparse_matrix['freq'] = 1
    sparse_matrix['colno'] = sparse_matrix.hhld_uniqueid + 4
    sparse_matrix['hhpumsid'] = sparse_matrix.pumano
    sparse_matrix = sparse_matrix[['hhpumsid','rowno','colno','freq']]
    return sparse_matrix

def generate_index_matrix(sparse_matrix):
    sparse_matrix1 = sparse_matrix.sort(columns=['colno','rowno'])
    sparse_matrix1['id'] = sparse_matrix1.index + 1
    sparse_matrix1['min'] = sparse_matrix1.id
    sparse_matrix1['max'] = sparse_matrix1.id
    index_matrix = sparse_matrix1.groupby('colno', as_index=False).agg({'min':'min','max':'max'})
    return index_matrix

