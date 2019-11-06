# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 22:07:13 2019

@author: peterpiontek
"""
import pandas as pd

def fixStandardTime2019(df):
    """Replaces glitched timestamps with values that are consistent with 
    the rows surrounding them to maintain a contiguous time series.
    The code is very fragile as it is tailored specifically to fix the 
    transition from DST to standard time in ndw.trafficspeed on Metis10. """
    
    # Assign constants
    idx_start = df[df.timestamp == '2019-10-27 01:56:00'].index[1]
    idx_stop = df[df.timestamp == '2019-10-27 02:59:00'].index[0]
    col_idx = df.columns.get_loc('timestamp')
    
    # Replace duplicate values with correct offset from last unique timestamp
    for i in range (idx_start, idx_stop + 1):
        offset = i - idx_start + 1
        df.iat[i, col_idx] = pd.to_datetime('2019-10-27 01:56:00') + \
            pd.Timedelta(minutes=offset)

    # Delete remaining extraneous rows
    rows_for_deletion = df[df.timestamp == '2019-10-27 02:59:00'].index[1:]
    df.drop(rows_for_deletion, inplace = True)
    df.reset_index(inplace = True)
    
   

    # # Redo cols that were dependant on timestamp
    # ['datetime']
    # if 'hour' in df.columns:
    #     idx_start = df[df.timestamp == '2019-10-27 01:56:00'].index[0]
    #     idx_stop = df[df.timestamp == '2019-10-27 02:59:00'].index[0] + 1
    #     col_idx = df.columns.get_loc('hour')
    #     for i in range (idx_start, idx_stop):
    #     df.iat[i, col_idx] = df.iloc['datetime'].dt.hour
    # df['minute'] = df['datetime'].dt.minute
        
    
    # # create hour/minute cols with continuously spaced vals (sine, cosine)
    # df['hour_sine'] = np.sin(2*np.pi*df.hour/24)
    # df['hour_cosine'] = np.cos(2*np.pi*df.hour/24)
    # df['minute_sine'] = np.sin(2*np.pi*df.minute/60)
    # df['minute_cosine'] = np.cos(2*np.pi*df.minute/60)
        
    # Check manually if the transition values have been fixed
    print(df[(df['timestamp'] > pd.to_datetime('2019-10-27 01:49:00')) & \
             (df['timestamp'] < pd.to_datetime('2019-10-27 03:11:00'))]\
              [['timestamp', 'hour', 'minute', 'flow', 'speed']])
        
    # Return fixed dataframe
    return df