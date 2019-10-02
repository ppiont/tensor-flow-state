# import necessary libs
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import psycopg2
import pickle
import holidays

# set working dir
os.chdir("C:/Users/peterpiontek/Google Drive/tensor-flow-state/tensor-flow-state")

# define directories
datadir = "./data/"
plotdir = './plots/'

# import homebrew
from modules.pgConnect import pgConnect

def getIdPatternDaySum(start, end, pattern):
    """
    Filter database by sensors containing a pattern, within a specified time period.
    Return a dataframe with coords and traffic volume + average speed,
    grouped by date, for the filtered set."
    """

    try:
        df = pd.DataFrame(columns = ['sensor_id','date','volume','speed_avg','lon','lat'])
        # connect to postgres and create cursor
        connection = pgConnect()
        cursor = connection.cursor()
        print("Connected to PostgreSQL")
        #define query
        query = "WITH pts AS \
            (SELECT * FROM ndw.mst_points_latest WHERE mst_id LIKE '" + pattern + "') \
            SELECT b.location AS sensor_id, b.date::date AS date, SUM(b.flow_sum / 60) AS volume, \
            AVG(b.speed_avg) AS speed_avg, ST_X(a.geom) AS lon, ST_Y(a.geom) AS lat \
            FROM pts AS a \
            INNER JOIN ndw.trafficspeed AS b \
            ON a.mst_id = b.location \
            WHERE b.date::date >= '" + start + "' \
            AND b.date::date <= '" + end + "' \
            GROUP BY b.location, a.geom, date::date \
            ORDER by b.location, a.geom, date::date"
            # execute query
        cursor.execute(query)
        # create df
        appendix = pd.DataFrame(cursor.fetchall(), columns = ['sensor_id','date','volume','speed_avg','lon','lat'])
        # append
        df = df.append(appendix, ignore_index=True)
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


test = getIdPatternDaySum('2019-06-01', '2019-08-31', 'RWS01_MONIBAS_0021hrl%')
# add holiday binary
test['holiday'] = np.array([int(x in holidays.NL()) for x in test['date']])
# add weekday
test['weekday'] = pd.to_datetime(test['date']).dt.dayofweek.astype(int) + 1
# add weekend
test['weekend'] = np.where(test['weekday'] > 5, 1, 0)
# convert strings to f64
test['speed_avg'] = test.speed_avg.astype('float64')
test['volume'] = test.volume.astype('float64')
# pickle
test.to_pickle(datadir + '3months_summed_daily_volume.pkl')
