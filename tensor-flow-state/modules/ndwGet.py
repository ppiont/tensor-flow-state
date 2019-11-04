### NDW import module ###

import numpy as np
import pandas as pd
import psycopg2
import holidays
from sklearn.preprocessing import OneHotEncoder
from modules.pgConnect import pgConnect

def ndwGet(start, end, ID='RWS01_MONIBAS_0021hrl0339ra'):
    
    try:
        # Connect to postgres and create cursor
        connection = pgConnect()
        cursor = connection.cursor()
        print("Connected to PostgreSQL")
        # Define query
        query = "WITH pts AS \
        (SELECT * FROM ndw.mst_points_latest WHERE mst_id = '" + ID + "') \
        SELECT ST_X(geom) lon, ST_Y(geom) lat, b.date, b.flow_sum, b.speed_avg, \
        mst_id, date_part('hour', b.date), \
        date_part('dow', b.date) \
        FROM pts AS a \
        INNER JOIN ndw.trafficspeed AS b \
        ON a.mst_id = b.location \
        WHERE b.date::date >= '" + start + "' \
        AND b.date::date <= '" + end + "' \
        ORDER BY b.date"
        
        # Execute query
        cursor.execute(query)

        # Fetch all to pandas df
        df = pd.DataFrame(cursor.fetchall(), columns = ['lon','lat','datetime',
                                'flow','speed','sensor_id','hour','weekday'])
        
        # Remove timezone
        df.datetime = df['datetime'].dt.tz_localize(None)
        
        # Add hours, minutes stuff
        df['date'] = df['datetime'].dt.date
        df['minute'] = df['datetime'].dt.minute
        
        # Make column order more OCD friendly
        df = df[['lon','lat','datetime','sensor_id','date','hour','minute',
                                             'weekday','flow','speed']]
        
        # Create hour/minute cols with continuously spaced vals (sine, cosine)
        df['hour_sine'] = np.sin(2*np.pi*df.hour/24)
        df['hour_cosine'] = np.cos(2*np.pi*df.hour/24)
        df['minute_sine'] = np.sin(2*np.pi*df.minute/60)
        df['minute_cosine'] = np.cos(2*np.pi*df.minute/60)

        # Fix weekday
        df['weekday'] += 1
        # Add weekend binary
        df['weekend'] = np.where(df['weekday'] > 5, 1, 0)
        
        # Add holiday binary
        df['holiday'] = np.array([int(x in holidays.NL()) for x in df['date']])
        
        # One-hot encode weekday
        onehot_encoder = OneHotEncoder(sparse=False, categories = 'auto')
        integer_encoded = df.weekday.values.reshape(len(df.weekday), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        days = ['monday','tuesday','wednesday','thursday','friday','saturday',
                                                                     'sunday']
        for day in days:
            df[day] = onehot_encoded[:,days.index(day)]
        
        # # set index to time (ERROR, just changes the index name)
        # df.index.name = 'datetime'

        # Close cursor
        cursor.close()
        
        return df

    except psycopg2.Error as e:
        print("Failed to read data from table:", e)
        
    finally:
        if (connection): # don't know why it's in parentheses
            connection.close()
            print("The PostgreSQL connection is closed\nFiles have been\
 fetched and processed")