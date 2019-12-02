# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:21:31 2019

@author: peterpiontek
"""

import h5py
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pandahouse import read_clickhouse


# Set working dir
os.chdir("C:/Users/peterpiontek/Google Drive/tensor-flow-state/tensor-flow-state")

# Define directories
datadir = "./data/"
plotdir = './plots/'

# Don't limit, truncate or wrap columns displayed by pandas
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 200) # Accepted line width before wrapping
pd.set_option('display.max_colwidth', -1)
# Display decimals instead of scientific with pandas
pd.options.display.float_format = '{:.2f}'.format

plt.rcParams['figure.figsize'] = (15, 15)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['image.cmap'] = 'viridis'



# Establish db connection
connection = {'host': 'http://leda.geodan.nl:8123'}
# Load KNMI data from db 
knmi_raw = read_clickhouse("SELECT x, y FROM knmi.MFBSNL25_05m GROUP BY x, y", connection = connection)
# ndw = read_clickhouse("SELECT x, y FROM ndw.mstpoints GROUP BY x, y", connection = connection)

# Convert knmi to GeoDataFrame
knmi = gpd.GeoDataFrame(knmi_raw, geometry = gpd.points_from_xy(knmi_raw.x, knmi_raw.y))
# Load NDW data
ndw = gpd.read_file(os.path.join(datadir, "ndw_shapefile/Telpunten_WGS84.shp"), layer='Telpunten_WGS84')

# plot
fig, ax = plt.subplots(figsize = (15, 15))
plt.rcParams['agg.path.chunksize'] = 10000
knmi.plot()
ndw.plot()

knmi.crs
knmi.crs = "+proj=stere +lat_0=90 +lon_0=0.0 +lat_ts=60.0 +a=6378.137 +b=6356.752 +x_0=0 +y_0=0"

knmi = knmi.to_crs({'init': 'epsg:4326'})
knmi = knmi.to_crs("+proj=longlat +a=3396000 +b=3396000")


import numpy as np

x = np.array(knmi['geometry'][:])
y = np.array(knmi.variables['y'][:])
xv,  yv = np.meshgrid(x, y)
p = pyproj.Proj("+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-105 +k=90 
            +x_0=0 +y_0=0 +a=6371200 +b=6371200 +units=m +no_defs")
lons, lats = p(xv, yv, inverse=True)



print(x)







# knmi.to_file(os.path.join(datadir, "knmi_gdf.geojson"), driver='GeoJSON')


























# KNMI proj
# +proj=stere +lat_0=90 +lon_0=0.0 +lat_ts=60.0 +a=6378.137 +b=6356.752 +x_0=0 +y_0=0


# knmi data:  https://data.knmi.nl/datasets/rad_nl25_rac_mfbs_5min/2.0?q=RAD_NL25_RAC_MFBS_5min
# regen_coords = np.array(client.execute("""
# SELECT x, y FROM knmi.MFBSNL25_05m GROUP BY x, y
# """))
# mst_coords = np.array(client.execute("""
# SELECT x, y FROM ndw.mstpoints GROUP BY x, y
# """))
# plt.plot(regen_coords[:, 0]-230, -1*regen_coords[:, 1] + 890, '.')
# plt.plot(mst_coords[:, 0], mst_coords[:, 1], '.')




# f = h5py.File('data/test2.h5', 'r')
# df = pd.DataFrame(f['image1']['image_data'])
# print(df)
# f.close()