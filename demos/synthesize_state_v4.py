import os
import sys
import argparse

from synthpop.recipes.starter3 import Starter
from synthpop.synthesizer import synthesize_all, enable_logging
import pandas as pd
import numpy as np


def run(state='AL', county='all', year=2013):
    def synthesize_county(state, county, year):
        starter = Starter(os.environ["CENSUS"], state, county, acsyear=year)
        synthetic_population = synthesize_all(starter)
        hh = synthetic_population[0]
        p = synthetic_population[1]
        state_fips = hh['state'].unique()[0]
        county_fips = hh['county'].unique()[0]

        # verify if there is a folder with the name of the state where we can save the synthesized files
        if not(os.path.isdir(state_fips)):
            os.mkdir(state_fips)

        p.to_csv('./{}/p_{}_{}_{}.csv'.format(state_fips, state_fips, county_fips, year))
        hh.to_csv('./{}/hh_{}_{}_{}.csv'.format(state_fips, state_fips, county_fips, year))

    if year >= 2018:
        url_national_data = "https://storage.googleapis.com/synthpop-public/PUMS2018/pums_2018_acs5/"
    else:
        url_national_data = "https://s3-us-west-1.amazonaws.com/synthpop-data2/"
    national_data = pd.read_csv(url_national_data + 'national_county.txt', dtype='str')
    state_data = national_data[national_data['State'] == state]

    if county != 'all':
        state_data = state_data[state_data['County ANSI'].isin(county.split(','))]

    for index, row in state_data.iterrows():
        county_name = row['County Name']
        synthesize_county(state, county_name, year)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("-s", "--state", type=str, help="State to synthesize")
        parser.add_argument("-c", "--county", type=str,
                            help="County or list of counties to synthesize in ANSI")
        parser.add_argument("-y", "--year", type=int, help="Year to synthesize")

        args = parser.parse_args()
        state = args.state if args.state else 'AL'
        county = args.county if args.county else 'all'
        year = args.year if args.year else 2013
        run(state, county, year)
    else:
        run()
