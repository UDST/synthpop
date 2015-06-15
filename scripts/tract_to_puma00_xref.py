import os
import urllib
import zipfile
from spandex import TableLoader
from spandex.io import exec_sql
from spandex.spatialtoolz import conform_srids, tag
import pandas as pd, numpy as np

import pandas.io.sql as sql
def db_to_df(query):
    """Executes SQL query and returns DataFrame."""
    conn = loader.database._connection
    return sql.read_frame(query, conn)

loader = TableLoader()

# Download puma 2000 geometry zip files
for i in range(73): 
    if i < 10:
        filename = 'p50%s_d00_shp.zip' % i
    else:
        filename = 'p5%s_d00_shp.zip' % i
    
    try:
        pumageom_file = urllib.URLopener()
        pumageom_file.retrieve("http://www2.census.gov/geo/tiger/PREVGENZ/pu/p500shp/%s" % filename, 
                          os.path.join(loader.get_path('puma_geom'), filename))
        print 'Downloading %s' % filename
    except:
        continue

# Unzip and add prj file to puma 2000 geometry
for i in range(73): 
    if i < 10:
        filename = 'p50%s_d00_shp.zip' % i
    else:
        filename = 'p5%s_d00_shp.zip' % i
    filepath = os.path.join(loader.get_path('puma_geom'), filename)
    
    if os.path.exists(filepath):
        print 'Unzipping and adding prj to %s' % filename
        
        with zipfile.ZipFile(filepath, "r") as z:
            z.extractall(loader.get_path('puma_geom'))
            
        # PUMA 2000 shapefile doesn't come with .prj file - create one
        shape_prjname = filename[:8] + '.prj'
        prj_filepath = os.path.join(loader.get_path('puma_geom'), shape_prjname)
        text_file = open(prj_filepath, "w")
        text_file.write('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]')
        text_file.close()

##Next step- do the same for tracts
tract_file = urllib.URLopener()
tract_file.retrieve("http://www2.census.gov/geo/tiger/TIGER2010DP1/Tract_2010Census_DP1.zip",
                    os.path.join(loader.get_path('tract2010_geom'), "Tract_2010Census_DP1.zip"))


with zipfile.ZipFile(os.path.join(loader.get_path('tract2010_geom'), "Tract_2010Census_DP1.zip"), "r") as z:
    z.extractall(loader.get_path('tract2010_geom'))

with loader.database.cursor() as cur:
    cur.execute("""
        CREATE EXTENSION IF NOT EXISTS postgis;
        CREATE SCHEMA IF NOT EXISTS staging;
    """)
loader.database.refresh()


shapefiles = {
    'staging.tracts10':
    'tract2010_geom/Tract_2010Census_DP1.shp',
}

loader.load_shp_map(shapefiles)


shapefiles = {}
for i in range(73): 
    if i < 10:
        filename = 'p50%s_d00.shp' % i
    else:
        filename = 'p5%s_d00.shp' % i
    filepath = os.path.join(loader.get_path('puma_geom'), filename)
    
    if os.path.exists(filepath):
        subfile_name = filename[:-4]
        shapefiles['staging.%s' % subfile_name] = 'puma_geom/%s' % filename
        
loader.load_shp_map(shapefiles)


conform_srids(loader.srid, schema=loader.tables.staging, fix=True)

exec_sql("DROP table if exists staging.puma00;")

sql_str = ""
for i in range(73): 
    if i < 10:
        filename = 'p50%s_d00.shp' % i
    else:
        filename = 'p5%s_d00.shp' % i
    filepath = os.path.join(loader.get_path('puma_geom'), filename)
    
    if os.path.exists(filepath):
        subfile_name = filename[:-4]
        sql_str = sql_str + 'select area, perimeter, puma5, name, geom from staging.%s' % subfile_name
        if i < 72:
            sql_str = sql_str + ' UNION ALL '
        
sql_str = 'with a as (' + sql_str + ') select * into staging.puma00 from a'
exec_sql(sql_str)

exec_sql('ALTER TABLE staging.puma00 ADD COLUMN gid BIGSERIAL PRIMARY KEY')

exec_sql("""
CREATE INDEX puma00_gist ON staging.puma00
  USING gist (geom);
""")

loader.database.refresh()

# Tag tracts with a parcel_id
tag(loader.tables.staging.tracts10, 'puma00_id', loader.tables.staging.puma00, 'puma5')

tract10_puma10_rel_file = urllib.URLopener()
tract10_puma10_rel_file.retrieve("http://www2.census.gov/geo/docs/maps-data/data/rel/2010_Census_Tract_to_2010_PUMA.txt", 
                  os.path.join(loader.get_path('tract2010_geom'), 'tract10_puma10_rel_file.csv'))

tract10_puma10_rel = pd.read_csv(os.path.join(loader.get_path('tract2010_geom'), 'tract10_puma10_rel_file.csv'), 
                                 dtype={
                                            "STATEFP": "object",
                                            "COUNTYFP": "object",
                                            "TRACTCE": "object",
                                            "PUMA5CE": "object"
                                        })
tract10_puma00 = db_to_df('select geoid10, namelsad10, puma00_id from staging.tracts10;')

##Need statefp/countyfp/tractce columns on tracts (split from geoid)
tract10_puma00['STATEFP'] = tract10_puma00.geoid10.str.slice(0,2)
tract10_puma00['COUNTYFP'] = tract10_puma00.geoid10.str.slice(2,5)
tract10_puma00['TRACTCE'] = tract10_puma00.geoid10.str.slice(5,)

print len(tract10_puma00)
print len(tract10_puma10_rel)

tract_puma_xref = pd.merge(tract10_puma10_rel, tract10_puma00, 
                           left_on = ['STATEFP', 'COUNTYFP', 'TRACTCE'], right_on = ['STATEFP', 'COUNTYFP', 'TRACTCE'])

tract_puma_xref = tract_puma_xref.rename(columns = {'STATEFP':'statefp', 'COUNTYFP':'countyfp', 'TRACTCE':'tractce', 
                                                    'PUMA5CE':'puma10_id'})

tract_puma_xref = tract_puma_xref[['statefp', 'countyfp', 'tractce', 'puma10_id', 'puma00_id']]

tract_puma_xref.to_csv('tract10_to_puma.csv', index = False)

