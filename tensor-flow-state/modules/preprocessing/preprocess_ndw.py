# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:45:39 2019

@author: peterpiontek
"""

# Import libraries
import os
import pandas as pd
import numpy as np

# Display and Plotting
import matplotlib.pyplot as plt

# Set working dir
os.chdir("C:/Users/peterpiontek/Google Drive/tensor-flow-state/tensor-flow-state")

# Set pandas and matplotlib settings
exec(open('modules/Settings.py').read())

# Define directories
datadir = "./data/"

# Load feather (1m data resolution)
fname =  os.path.join(datadir, 'RWS01_MONIBAS_0021hrl0414ra_jun_oct_repaired.feather')
df = pd.read_feather(fname) 
df.set_index('timestamp', inplace = True, drop = True)

# Make new 10m aggregate df
df_10m = df.resample('10T').mean()

# Add speed limit information
df_10m['speed_limit'] = np.where((df_10m.index.hour < 19) & (df_10m.index.hour >= 6), 100, 120)

# Find mean and sd for training batch
mean100_10m, mean120_10m = df_10m[: -(31 * 24 * 6)].groupby(['speed_limit']).mean().unstack().values
sd100_10m, sd120_10m = df_10m[: -(31 * 24 * 6)].groupby(['speed_limit']).std().unstack().values

# Stajdard9ze speed on training batch mean/sd (10m)
df_10m['speed_standardized'] = np.where(df_10m.speed_limit == 100, (df_10m.speed - mean100_10m) / sd100_10m, (df_10m.speed - mean120_10m) / sd120_10m)

# Plot Normalized distribution of training batch
df_10m.iloc[: -(31 * 24 * 6), df_10m.columns.get_loc('speed_standardized')].hist(bins = int(df_10m.speed.max() + 1))

df_10m.reset_index(inplace = True)
df_10m.to_feather(os.path.join(datadir, '5_months_10m_resolution_standardized_speed.feather'))
