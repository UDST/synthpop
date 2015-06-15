from synthpop.recipes.starter import Starter
from synthpop.synthesizer import synthesize_all, enable_logging 
import os
import pandas as pd, numpy as np
from multiprocessing import Pool
import pickle

counties = ['Adams County', 'Arapahoe County', 'Boulder County', 'Broomfield County', 
            'Clear Creek County', 'Denver County', 'Douglas County', 'Elbert County', 
            'Gilpin County', 'Jefferson County', 'Weld County']

def synthesize_county(county):
    starter = Starter(os.environ["CENSUS"], "CO", county)
    synthetic_population = synthesize_all(starter)
    return synthetic_population
    
if __name__ == '__main__':
    pool = Pool(4)
    regional_population = pool.map(synthesize_county, counties)

    households = []
    persons = []
    for population in regional_population:
        hh = population[0]
        p = population[1]
        households.append(hh)
        persons.append(p)
        
        val_metrics = population[2]
        cid = val_metrics.keys()[0][1]
        pickle.dump(val_metrics, open( "save%s.p" % cid, "wb" ))
        
    all_households = pd.concat(households, ignore_index=True)
    all_persons = pd.concat(persons, ignore_index=True)

    all_households.to_csv('households_drcog.csv')
    all_persons.to_csv('persons_drcog.csv')

    ##Calculate validation metrics
    hh = all_households
    print hh.groupby('county').size()

    hh_marginals = []
    for county in counties:
        starter = Starter(os.environ["CENSUS"], "CO", county)
        hh_marginals.append(starter.h_acs_cat)
    hh_marginals = pd.concat(hh_marginals)
    control_variables = hh_marginals.columns.levels[0]

    control_categories = dict()
    for control_variable in control_variables:
        control_categories[control_variable] = list(hh_marginals[control_variable].columns.values)

    val_df = pd.DataFrame(index = hh_marginals.index)
    first_control = 'income' #calculate total synth hh by summing across the categories within this control
    first_var_cats = control_categories[first_control]
    val_df['total_hh_obs'] = np.zeros(len(val_df))
    val_df['total_hh_syn'] = hh.groupby(['state', 'county', 'tract', 'block group']).size()
    for cat in first_var_cats:
        val_df['total_hh_obs'] = val_df['total_hh_obs'] + hh_marginals[first_control][cat]
        
    for control_variable in control_categories.keys():
        for var_cat in control_categories[control_variable]:
            observed = hh_marginals[control_variable][var_cat]
            synthesized = hh[hh[control_variable] == var_cat].groupby(['state', 'county', 'tract', 'block group']).size()
            val_df[control_variable + '_' + var_cat + '_obs'] = observed
            val_df[control_variable + '_' + var_cat + '_syn'] = synthesized

    val_df.to_csv('hh_bg_validation_metrics.csv')