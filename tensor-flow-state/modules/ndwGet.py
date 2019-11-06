### NDW import module ###

# Note to self: date_trunc with modulo magic then avg(speed) sum(flow) for aggregating time 

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
        SELECT b.date::timestamp without time zone AS timestamp, \
        ST_X(geom) AS lon, \
        ST_Y(geom) AS lat, \
        mst_id AS sensor_id, \
        date_trunc('day', b.date::timestamp without time zone) AS date, \
        date_part('hour', b.date)::SMALLINT, \
        date_part('minute', b.date)::SMALLINT as minute, \
        (date_part('dow', b.date)::SMALLINT + 1) AS weekday, \
        (b.flow_sum / 60)::INTEGER AS flow, \
        b.speed_avg::SMALLINT AS speed \
        FROM pts AS a \
        INNER JOIN ndw.trafficspeed AS b \
        ON a.mst_id = b.location \
        WHERE b.date::date >= '" + start + "' \
        AND b.date::date <= '" + end + "' \
        ORDER BY b.date"
        
        # Execute query
        cursor.execute(query)

        # Fetch all to pandas df
        df = pd.DataFrame(cursor.fetchall(), \
                          columns = ['timestamp', 'lon', 'lat', 'sensor_id', 
                                     'date', 'hour', 'minute', 'weekday', 
                                     'flow', 'speed'])
        
        
        # # Convert to pd tz
        # df['timestamp'] = pd.to_datetime(df.timestamp, utc=True).dt.tz_convert('Europe/Amsterdam')
        # df['date'] = pd.to_datetime(df.date, utc=True).dt.tz_convert('Europe/Amsterdam')
        
        # # Remove timezone
        # df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        # df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        
        # # format date
        df['date'] = df['date'].dt.strftime("%Y-%m-%d")
        
        
            
        # Create hour/minute cols with continuously spaced vals (sine, cosine)
        df['hour_sine'] = np.sin(2 * np.pi * df.hour / 24)
        df['hour_cosine'] = np.cos(2 * np.pi * df.hour / 24)
        df['minute_sine'] = np.sin(2 * np.pi * df.minute / 60)
        df['minute_cosine'] = np.cos(2 * np.pi * df.minute / 60)

        # Add weekend binary
        df['weekend'] = np.where(df['weekday'] > 5, 1, 0).astype(np.int16)
        
        # # Add holiday binary
        df['holiday'] = np.array([int(x in holidays.NL()) for x in df['date']]).astype(np.int16)
        
        # One-hot encode weekday
        onehot_encoder = OneHotEncoder(sparse=False, categories = 'auto')
        integer_encoded = df.weekday.values.reshape(len(df.weekday), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        days = ['monday','tuesday','wednesday','thursday','friday','saturday',
                                                                     'sunday']
        for day in days:
            df[day] = onehot_encoded[:,days.index(day)].astype(np.int16)

        # Close cursor
        cursor.close()
        
        # Print success statement and return df
        print("Files have been fetched and processed")
        return df
    

    except psycopg2.Error as e:
        print("Failed to read data from table:\n", e)
        
    finally:
        if (connection): # don't know why it's in parentheses
            connection.close()
            print("The PostgreSQL connection is closed")