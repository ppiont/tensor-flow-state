### MAIN ###

# import libraries
import os
import numpy as np
import pandas as pd
#from datetime import datetime, date
import pickle
import matplotlib.pyplot as plt
import seaborn
#import holidays
#from sklearn.preprocessing import OneHotEncoder

# set working dir
os.chdir("C:/Users/peterpiontek/Google Drive/tensor-flow-state/tensor-flow-state")

# import homebrew
#from modules.pgConnect import pgConnect
from modules.ndwGet import ndwGet
from modules.knmiGet import knmiGet

# define directories
datadir = "./data/"
plotdir = './plots/'

# create directories
if not os.path.exists(datadir):
    os.makedirs(datadir)
if not os.path.exists(plotdir):
    os.makedirs(plotdir)

# fetch traffic data
traffic_data = ndwGet('06-01-2019', '08-31-2019', 'RWS01_MONIBAS_0021hrl0403ra')

# get weather data (from Schiphol as defualt)
weather_data = knmiGet('2019060101', '2019083124', [240])

df = pd.merge(traffic_data, weather_data, how='left', on = ['date', 'hour']) #Fix

## remove null vals
#df.dropna(inplace=True)

# save to pickle
df.to_pickle(datadir + '3months_weather.pkl')


















## define knmi query
#knmi_query = """
#SELECT * FROM ndw.weerdata
#WHERE stationid = 240
#AND time >= '06-01-2019' 
#AND time <= '08-31-2019'
#"""
#
#cursor.execute(knmi_query)
#knmi_records = cursor.fetchall()
#
## create np array of data and set data type
#knmi_array = np.array(knmi_records, dtype=[('stationid', 'i4'), ('datetime', datetime), ('sunduration', 'i4'), ('radiation', 'i4'), 
#                                       ('temperature', 'i4'), ('rain', 'i4'), ('humidity', 'i4')])

#weather_data = pd.DataFrame(knmi_records) # find out why column names aren't carried over
#weather_data.columns = ['stationid', 'datetime', 'sunduration', 'radiation', 'temperature', 'rain', 'humidity']
#
# get pd dataframe
##create a datehour field to join datasets on (weather is hourly)
#traffic_data['datehour'] = traffic_data['datetime'].astype(str).str[0:13]
#weather_data['datehour'] = weather_data['datetime'].astype(str).str[0:13]



#df = merged[['datetime_x', 'weekday', 'hour', 'speed', 'flow', 'radiation', 'temperature', 'rain']].copy()
#df.columns = ['datetime' if x=='datetime_x' else x for x in df.columns]

# remove clutter variables
#del(knmi_array, knmi_records, ndw_array, ndw_records, ndw_query, knmi_query, merged)


#df.info()
#df.describe()
#print(df.isnull().any())


#### display settings fix for less truncation when using print(). Probably wise to reset these at some point
#pd.options.display.max_columns = 50  # None -> No Restrictions
#pd.options.display.max_rows = 200    # None -> Be careful with this 
#pd.options.display.max_colwidth = 100
#pd.options.display.precision = 3
#pd.options.display.width = None
#### 

