import os
import re
import zipfile
import urllib, urllib2
import pandas as pd, numpy as np
from bs4 import BeautifulSoup
from spandex import TableLoader

loader = TableLoader()

soup = BeautifulSoup(urllib2.urlopen("http://www2.census.gov/acs2013_5yr/pums/"))

tags = soup.find_all(href=re.compile("csv_h..\.zip"))
hpums_links = []
for t in tags:
    hpums_links.append(t['href'])
    
tags = soup.find_all(href=re.compile("csv_p..\.zip"))
ppums_links = []
for t in tags:
    ppums_links.append(t['href'])

pums_links = hpums_links + ppums_links
for pums_file in pums_links:
    print pums_file
    pums_file_dl = urllib.URLopener()
    pums_file_dl.retrieve("http://www2.census.gov/acs2013_5yr/pums/%s" % pums_file, 
                      os.path.join(loader.get_path('pums'), pums_file))

for pums_file in pums_links:
    filepath = os.path.join(loader.get_path('pums'), pums_file)
    
    if os.path.exists(filepath):
        print 'Unzipping %s' % pums_file
        
        with zipfile.ZipFile(filepath, "r") as z:
            z.extractall(loader.get_path('pums'))

for pums_file in ['ss13husa.csv', 'ss13husb.csv', 
                  'ss13husc.csv', 'ss13husd.csv',
                  'ss13pusa.csv', 'ss13pusb.csv',
                  'ss13pusc.csv', 'ss13pusd.csv']:
    print 'Processing %s' % pums_file
    pums = pd.read_csv(os.path.join(loader.get_path('pums'), pums_file))

    for state_id in np.unique(pums['ST']):
        print '    Processing pums for state %s' % state_id
        pum_state = pums[pums['ST'] == state_id]
        state_id = '{:>02}'.format(state_id)
        if pums_file[4] == 'h':
            pums_state_filename = 'puma_h_%s.csv' % (state_id)
        elif pums_file[4] == 'p':   
            pums_state_filename = 'puma_p_%s.csv' % (state_id)
        pum_state.to_csv(os.path.join(loader.get_path('pums'), pums_state_filename), index = False)

        print '        Slicing up pums files by 2000 pumas'
        for puma00 in np.unique(pum_state['PUMA00']):
            if puma00 != -9:
                print puma00
                df = pum_state[pum_state['PUMA00'] == puma00]
                puma00 = '{:>05}'.format(puma00)
                if pums_file[4] == 'h':
                    output_filename = 'puma00_h_%s_%s.csv' % (state_id, puma00)
                elif pums_file[4] == 'p':   
                    output_filename = 'puma00_p_%s_%s.csv' % (state_id, puma00)
                df.to_csv(os.path.join(loader.get_path('pums'), output_filename), index = False)

        print '        Slicing up pums files by 2010 pumas'
        for puma10 in np.unique(pum_state['PUMA10']):
            if puma10 != -9:
                print puma10
                df = pum_state[pum_state['PUMA10'] == puma10]
                puma10 = '{:>05}'.format(puma10)
                if pums_file[4] == 'h':
                    output_filename = 'puma10_h_%s_%s.csv' % (state_id, puma10)
                elif pums_file[4] == 'p':   
                    output_filename = 'puma10_p_%s_%s.csv' % (state_id, puma10)
                df.to_csv(os.path.join(loader.get_path('pums'), output_filename), index = False)