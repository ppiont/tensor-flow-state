# import necessary libs
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import psycopg2

# set working dir
os.chdir("C:/Users/peterpiontek/Google Drive/tensor-flow-state/tensor-flow-state")

# define directories
datadir = "./data/"
plotdir = './plots/'

# import homebrew
from modules.pgConnect import pgConnect


def fetchLike(pattern):
    try:
        # connect to postgres and create cursor
        connection = pgConnect()
        cursor = connection.cursor()
        print("Connected to PostgreSQL")
        #define query
        query = "SELECT *, \
        lead(mst_id) OVER (ORDER BY mst_id) as leading_id \
        FROM ndw.mst_points_latest \
        WHERE mst_id LIKE '" + pattern + "' \
        ORDER BY mst_id"

        # execute query
        cursor.execute(query)
        
        # fetch all to pandas df
        df = pd.DataFrame(cursor.fetchall())# , columns = ['lon','lat','datetime','flow','speed','location','hour','weekday'])

        cursor.close()
        
        return df
    
    except psycopg2.Error as e:
        print("Failed to read data from database:", e)
        
    finally:
        if (connection): # don't know why it's in parentheses
            connection.close()
            print("The PostgreSQL connection is closed\nFiles have been\
 fetched and processed")


def getAll(start, end, id_list):

    try:
        df = pd.DataFrame(columns = ['id','lon','lat','datetime','flow','speed','weekday'])
        # connect to postgres and create cursor
        connection = pgConnect()
        cursor = connection.cursor()
        print("Connected to PostgreSQL")
        #define query
        for id in id_list[:10]:
            query = "WITH pts AS \
            (SELECT * FROM ndw.mst_points_latest WHERE mst_id = '" + id + "') \
            SELECT mst_id, ST_X(geom) lon, ST_Y(geom) lat, b.date, b.flow_sum, b.speed_avg, \
            date_part('dow', b.date) \
            FROM pts AS a \
            INNER JOIN ndw.trafficspeed AS b \
            ON a.mst_id = b.location \
            WHERE b.date >= '" + start + "' \
            AND b.date <= '" + end + "' \
            ORDER BY b.date"
            # execute query
            cursor.execute(query)
            # create df
            appendix = pd.DataFrame(cursor.fetchall(), columns = ['id', 'lon','lat','datetime','flow','speed','weekday'])
            # append
            df = df.append(appendix, ignore_index=True)
        # # remove timezone
        # df.iloc[2] = df.iloc[2].dt.tz_localize(None)
        #         # add hours, minutes stuff
        # df['date'] = df['datetime'].dt.date
        # # make column order more OCD friendly
        # df = df[['lon','lat','datetime','flow','speed','weekday']]
        # # close cursor
        # cursor.close()
        # # add hours, minutes stuff
        # df['date'] = df[2].dt.date
        return df
        # close cursor
        cursor.close()

    except psycopg2.Error as e:
        print("Failed to read data from table:", e)
        
    finally:
        if (connection): # don't know why it's in parentheses
            connection.close()
            print("The PostgreSQL connection is closed\nFiles have been\
 fetched and processed")

# the working query
query = """
SELECT location AS sensor, date::date AS date, SUM(flow_avg / 60) AS intensity
FROM ndw.trafficspeed
WHERE location LIKE 'RWS01_MONIBAS_0021hrl%'
AND date >= '06-01-2019' AND date <= '08-31-2019'
GROUP BY location, date::date
ORDER by location, date::date
"""


IDs = list(fetchLike('RWS01_MONIBAS_0021hrl%')[2])

test = getAll('06-01-2019', '08-31-2019', IDs)
