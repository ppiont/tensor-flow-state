# import necessary libs
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point #, Polygon
import matplotlib as mpl
import matplotlib.pyplot as plt
# from shapely.wkt import loads
import numpy as np
import owslib
import seaborn as sns

# set working dir
os.chdir("C:/Users/peterpiontek/Google Drive/tensor-flow-state")

# define directories
datadir = "./data/"
plotdir = './plots/'

# import homebrew
from modules.pgConnect import pgConnect
/*
All measurement sites along the left side of the A2, excluding ramps and connectors
field leading_id references to the next measurement site along the road (if available)
*/
SELECT *,
lead(mst_id) OVER (ORDER BY mst_id) as leading_id
FROM ndw.mst_points_latest
WHERE mst_id LIKE 'RWS01_MONIBAS_0021hrl%'
ORDER BY mst_id


        # connect to postgres and create cursor
        connection = pgConnect()
        cursor = connection.cursor()
        print("Connected to PostgreSQL")
        #define query
        query = "SELECT *, \
        lead(mst_id) OVER (ORDER BY mst_id) as leading_id  \
        FROM ndw.mst_points_latest \
        WHERE mst_id LIKE 'RWS01_MONIBAS_0021hrl%' \
        ORDER BY mst_id""
        
        # execute query
        cursor.execute(query)
        
        # fetch all to pandas df
        df = pd.DataFrame(cursor.fetchall(), columns = ['lon','lat','datetime',
                                'flow','speed','location','hour','weekday'])
        
        # remove timezone
        df.datetime = df['datetime'].dt.tz_localize(None)
        
        # add hours, minutes stuff
        df['date'] = df['datetime'].dt.date
        df['minute'] = df['datetime'].dt.minute
        
        # make column order more OCD friendly
        df = df[['lon','lat','datetime','location','date','hour','minute',
                                             'weekday','flow','speed']]
        
        # create hour/minute cols with continuously spaced vals (sine, cosine)
        df['hour_sine'] = np.sin(2*np.pi*df.hour/24)
        df['hour_cosine'] = np.cos(2*np.pi*df.hour/24)
        df['minute_sine'] = np.sin(2*np.pi*df.minute/60)
        df['minute_cosine'] = np.cos(2*np.pi*df.minute/60)
        
        # add weekend binary
        df['weekend'] = np.where(df['weekday'] > 4, 1, 0)
        
        # add holiday binary
        df['holiday'] = np.array([int(x in holidays.NL()) for x in df['date']])
        
        # one-hot encode weekday
        onehot_encoder = OneHotEncoder(sparse=False, categories = 'auto')
        integer_encoded = df.weekday.values.reshape(len(df.weekday), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        days = ['monday','tuesday','wednesday','thursday','friday','saturday',
                                                                     'sunday']
        for day in days:
            df[day] = onehot_encoded[:,days.index(day)]
        
        # # set index to time (ERROR, just changes the index name)
        # df.index.name = 'datetime'

        # close cursor
        cursor.close()
        
        return df
    
    except psycopg2.Error as e:
        print("Failed to read data from table:", e)
        
    finally:
        if (connection): # don't know why it's in parentheses
            connection.close()
            print("The PostgreSQL connection is closed\nFiles have been\
 fetched and processed")