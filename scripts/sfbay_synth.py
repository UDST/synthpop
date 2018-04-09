import time 
import os
import pandas as pd, numpy as np

from synthpop.census_helpers import Census
from synthpop.recipes.starter2 import Starter
from synthpop.synthesizer import synthesize_all, synthesize_all_in_parallel, enable_logging

pd.set_option('display.max_columns', 500)
import warnings 
warnings.filterwarnings('ignore')




counties = [
    "Napa County", "Santa Clara County", "Solano County", "San Mateo County",
    "Marin County", "San Francisco County", "Sonoma County",
    "Contra Costa County", "Alameda County"]
# county_tuples = []
# for county in counties:
#     print('Starting {0}'.format(county))
#     starter = Starter(os.environ["CENSUS"], "CA", county)
#     county_dfs = synthesize_all(starter)
#     county_tuples.append(county_dfs)
# hh_all = pd.concat([county[0] for county in county_tuples])
# p_all = pd.concat([county[1] for county in county_tuples])
# fits_all = {}
# for county in county_tuples:
#     fits_all.update(county[2])


if __name__ == '__main__':

    for county in counties:
        c = Census(os.environ["CENSUS"])
        starter = Starter(os.environ["CENSUS"], "CA", county)

        county_dfs = synthesize_all_in_parallel(starter)

        hh_all = county_dfs[0]
        p_all = county_dfs[1]
        fits_all = county_dfs[2]

        hh_all.index.name = 'household_id'
        p_all.index.name = 'person_id'
        p_all.rename(columns = {'hh_id':'household_id'}, inplace = True)

        hh_all['age_of_head'] = p_all[p_all.RELP == 0].groupby('household_id').AGEP.max()
        hh_all['race_of_head'] = p_all[p_all.RELP == 0].groupby('household_id').RAC1P.max()
        hh_all['workers'] = p_all[p_all.ESR.isin([1, 2, 4, 5])].groupby('household_id').size()
        hh_all['children'] = p_all[p_all.AGEP < 18].groupby('household_id').size()
        hh_all['tenure'] = 2
        hh_all.tenure[hh_all.TEN < 3] = 1 #tenure coded 1:own, 2:rent
        hh_all['recent_mover'] = 0
        hh_all.recent_mover[hh_all.MV < 4] = 1 #1 if recent mover (within last five years)
        hh_all = hh_all.rename(columns = {'VEH':'cars', 'HINCP':'income', 'NP':'persons', 'BLD':'building_type'})


        for col in hh_all.columns:
            if col not in ['persons', 'income',
                               'age_of_head', 'race_of_head', 'hispanic_head',
                               'workers', 'children', 'cars',
                               'tenure', 'recent_mover', 'building_type', 'serialno', 
                               'state', 'county', 'tract', 'block group']:
                    del hh_all[col]

        p_all.rename(columns = {'AGEP':'age', 'RAC1P':'race_id', 'NP':'persons', 'SPORDER':'member_id', 'HISP': 'hispanic',
                                'RELP':'relate', 'SEX':'sex', 'WKHP':'hours', 'SCHL':'edu', 'PERNP':'earning'},
                                inplace = True)
        p_all['student'] = 0
        p_all.student[p_all.SCH.isin([2,3])] = 1
        p_all['work_at_home'] = 0
        p_all.work_at_home[p_all.JWTR == 11] = 1
        p_all['worker'] = 0
        p_all.worker[p_all.ESR.isin([1, 2, 4, 5])] = 1

        for col in p_all.columns:
            if col not in ['household_id', 'member_id',
                               'relate', 'age', 'sex', 'race_id', 'hispanic',
                               'student', 'worker', 'hours',
                               'work_at_home', 'edu', 'earning']:
                del p_all[col]

        hh_all.to_csv('{0}_hh_synth_parallel.csv'.format(county))
        p_all.to_csv('{0}_p_synth_parallel.csv'.format(county))

    # concat all the county dfs
    hh_fnames = glob('*hh*.csv')

    p_df_list = []
    hh_df_list = []
    hh_index_start = 0
    p_index_start = 0

    for hh_file in hh_fnames:
        county = hh_file.split('_hh')[0]
        hh_df = pd.read_csv(hh_file, index_col='household_id', header=0)
        p_df = pd.read_csv(glob(county + '_p*.csv')[0], index_col='person_id', header=0)
        print(county + ': {0}'.format(str(hh_df.iloc[0].county)))
        hh_df.index += hh_index_start
        p_df.household_id += hh_index_start
        p_df.index += p_index_start
        hh_df_list.append(hh_df)
        p_df_list.append(p_df)
        hh_index_start = hh_df.index.values[-1] + 1
        p_index_start = p_df.index.values[-1] + 1
    print(len(hh_all.iloc[hh_all.index.duplicated(keep=False)]))
    print(len(p_all.iloc[p_all.index.duplicated(keep=False)]))
    p_all.to_csv('sfbay_persons_2018_04_08.csv')
    hh_all.to_csv('sfbay_households_2018_04_08.csv')