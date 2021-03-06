### MAIN ###

# Import libraries
import os
import pandas as pd
import feather

# Set working dir
os.chdir("C:/Users/peterpiontek/Google Drive/tensor-flow-state/tensor-flow-state")

# Import homebrew
#from modules.pgConnect import pgConnect
from modules.ndw_get import ndw_get
# from modules.knmi_get import knmi_get

# Define directories
datadir = "./data/"
plotdir = './plots/'

# create directories
if not os.path.exists(datadir):
    os.makedirs(datadir)
if not os.path.exists(plotdir):
    os.makedirs(plotdir)
    
# Don't limit, truncate or wrap columns displayed by pandas
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 200) # accepted line width before wrapping
pd.set_option('display.max_colwidth', -1)
# Display decimals instead of scientific with pandas
pd.options.display.float_format = '{:.2f}'.format

# Fetch traffic data
traffic_data = ndw_get('06-01-2019', '10-31-2019', 'RWS01_MONIBAS_0021hrl0414ra')

# Get weather data (from Schiphol as defualt)
weather_data = knmi_get('2019060101', '2019103124', [240])

df = pd.merge(traffic_data, weather_data, how='left', on = ['date', 'hour']) #Fix

## Remove null vals
#df.dropna(inplace=True)

# Save to pickle
traffic_data.to_pickle(datadir + 'RWS01_MONIBAS_0021hrl0414ra_jun_oct.pkl')



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
