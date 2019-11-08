# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:49:46 2019

@author: peterpiontek
"""

import pandas as pd

def repairTimeSeries(dataframe, timestamp_col = 'timestamp', cols_not_to_fill = [], fillna_method = 'pad', freq = 'T'):
    """Removes duplicate timestamps; adds missing timestamps; fills missing values.
    Specifically made for glitchy timeseries such as ndw.trafficspeed on Metis10."""
    # Remove duplicates
    dataframe.drop_duplicates(inplace = True)

    # Ensure series passed is explicitly datetime to create DateTimeIndex
    datetime_series = pd.to_datetime(dataframe[timestamp_col])
    # Set as new index
    dataframe.set_index(pd.DatetimeIndex(datetime_series.values), inplace = True)
    
    # Reindex with date range to fill missing timestamps (minutes in this case)
    dataframe = dataframe.reindex(pd.date_range(start = dataframe.index.min(), end = dataframe.index.max(), freq = freq))
    # Add filled timestamps from index into timestamp col as well
    dataframe[timestamp_col] = dataframe.index
    
    # Find cols with null vals to be filled statically (excludes cols selected in cols_not_to_fill)
    nan_cols = [col for col in dataframe.columns[dataframe.isna().any()].tolist() if col not in cols_not_to_fill]
    
    # Fill said cols with static vals
    dataframe[nan_cols] = dataframe[nan_cols].fillna(method = fillna_method)
    
    # # Return to RangeIndex
    # dataframe.reset_index(drop = True, inplace = True)
    
    # # Pad null values with the previous valid value
    # dataframe.fillna(method = fillna_method, inplace = True)
    
    return dataframe