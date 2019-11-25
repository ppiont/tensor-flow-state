# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 22:07:13 2019

@author: peterpiontek
"""
import pandas as pd

def fix_standard_time_2019(df):
    """DEPRECATED AND USELESS:    
    Replaces glitched timestamps with values that are consistent with 
    the rows surrounding them to maintain a contiguous time series.
    The code is very fragile as it is tailored specifically to fix the 
    transition from DST to standard time in ndw.trafficspeed on Metis10. """
    
    # Assign constants
    idx_start = df[df.timestamp == '2019-10-27 01:56:00'].index[1]
    idx_stop = df[df.timestamp == '2019-10-27 02:59:00'].index[0]
    col_idx = df.columns.get_loc('timestamp')
    
    hour = True if 'hour' in df.columns else False
    minute = True if 'minute' in df.columns else False
    
    # Replace duplicate values with correct offset from last unique timestamp
    for i in range (idx_start, idx_stop + 1):
        offset = i - idx_start + 1
        df.iat[i, col_idx] = pd.to_datetime('2019-10-27 01:56:00') + \
            pd.Timedelta(minutes=offset)
        if hour:
            df.at[df.index[i], 'hour'] = int(df.iat[i, col_idx].strftime('%H'))
        if minute:
            df.at[df.index[i], 'minute'] = int(df.iat[i, col_idx].strftime('%M'))
            
    # Delete remaining extraneous rows
    df.drop_duplicates(inplace = True)
    df.reset_index(drop = True, inplace = True)
    
    # # Check manually if the transition values have been fixed
    # print(df[(df['timestamp'] > pd.to_datetime('2019-10-27 01:49:00')) & \
    #          (df['timestamp'] < pd.to_datetime('2019-10-27 03:11:00'))]\
    #           [['timestamp', 'hour', 'minute', 'flow', 'speed']])
        
    # Return fixed dataframe
    return df
        