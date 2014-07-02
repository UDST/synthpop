import numpy as np
import pandas as pd


def populate_master_matrix(hh_dims, per_dims, hhld_units, hh_sample, per_sample):
    total_cols = 4 + hh_dims.prod() + per_dims.prod()
    total_rows = hhld_units
    matrix = pd.DataFrame(index=np.arange(0,total_rows), columns=np.arange(0, total_cols), dtype='int')
    matrix = matrix.fillna(0)
    hh_result = hh_sample.sort('hhid')[['state','pumano','hhid','serialno','hhld_uniqueid']]
    matrix[0] = np.array(hh_result['state'])
    matrix[1] = np.array(hh_result['pumano'])
    matrix[2] = np.array(hh_result['hhid'])
    matrix[3] = np.array(hh_result['serialno'])
    mat_col_hh = zip(np.array(hh_result.hhid), np.array(hh_result.hhld_uniqueid))
    for hh in mat_col_hh:
        row = hh[0]
        col = hh[1]
        matrix[col+3][matrix[2]==row] = matrix[col+3][matrix[2]==row]+1
    per_result = per_sample.sort('hhid')[['state','pumano','hhid','serialno','person_uniqueid']]
    mat_col_per = zip(np.array(per_result.hhid), np.array(per_result.person_uniqueid))
    for per in mat_col_per:
        row = per[0]
        col = per[1]
        matrix[col + 3 + hh_dims.prod()][matrix[2]==row] = matrix[col + 3 + hh_dims.prod()][matrix[2]==row] + 1
    return matrix

def pseudo_sparse_matrix(data_dir, hh_sample, hh_dims, per_sample):
    sparse_matrix = hh_sample[['pumano', 'hhid', 'hhld_uniqueid']]
    sparse_matrix['rowno'] = sparse_matrix.hhid - 1
    sparse_matrix['freq'] = 1
    sparse_matrix['colno'] = sparse_matrix.hhld_uniqueid + 3
    sparse_matrix['hhpumsid'] = sparse_matrix.pumano
    sparse_matrix = sparse_matrix[['hhpumsid','rowno','colno','freq']]
    sparse_matrix_p = per_sample[['pumano', 'hhid', 'person_uniqueid']]
    sparse_matrix_p['rowno'] = sparse_matrix_p.hhid - 1
    sparse_matrix_p['freq'] = 1
    sparse_matrix_p['colno'] = sparse_matrix_p.person_uniqueid + 3 + hh_dims.prod()
    sparse_matrix_p['hhpumsid'] = sparse_matrix_p.pumano
    sparse_matrix_p = sparse_matrix_p[['hhpumsid','rowno','colno','freq']]
    sparse_matrix = pd.concat([sparse_matrix, sparse_matrix_p])
    return sparse_matrix

def generate_index_matrix(sparse_matrix):
    sparse_matrix1 = sparse_matrix.sort(columns=['colno','rowno'])
    sparse_matrix1['id'] = sparse_matrix1.index + 1
    sparse_matrix1['min'] = sparse_matrix1.id
    sparse_matrix1['max'] = sparse_matrix1.id
    index_matrix = sparse_matrix1.groupby('colno', as_index=False).agg({'min':'min','max':'max'})
    sparse_matrix1 = sparse_matrix1[['id', 'hhpumsid', 'rowno', 'colno', 'freq']]
    return sparse_matrix1, index_matrix

