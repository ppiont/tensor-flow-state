# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 09:55:35 2019

@author: peterpiontek
"""

# Import libraries
import os
import pandas as pd
import numpy as np

# Display and Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score, mean_absolute_error

# Set working dir
os.chdir("C:/Users/peterpiontek/Google Drive/tensor-flow-state/tensor-flow-state")

# Define directories
datadir = "./data/"
plotdir = './plots/'

# Don't limit, truncate or wrap columns displayed by pandas
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 200) # Accepted line width before wrapping
pd.set_option('display.max_colwidth', -1)
# Display decimals instead of scientific with pandas
pd.options.display.float_format = '{:.2f}'.format
# Increase figure size and set modern aspect ratio
plt.rcParams['figure.figsize'] = (16, 9)

# Load data
fname =  os.path.join(datadir, 'RWS01_MONIBAS_0021hrl0414ra_jun_oct_repaired.pkl')
df = pd.read_pickle(fname)

df.drop(['timestamp', 'date', 'lon', 'lat', 'flow', 'sensor_id'], axis = 1, inplace = True)

# sns.distplot(df['speed'])
# sns.despine()
# plt.tight_layout()
# plt.savefig(plotdir + 'distplot.png', dpi = 600)

### Improve below!!
# for mins in [1, 5, 10, 15, 30, 45, 60]:
#     df[f'lag_{mins}'] = df.speed.shift(mins).fillna('bfill').copy()
df['lag_1'] = df['speed'].shift(1).fillna(method = 'bfill').copy()
df['lag_5'] = df['speed'].shift(5).fillna(method = 'bfill').copy()
df['lag_10'] = df['speed'].shift(10).fillna(method = 'bfill').copy()
df['lag_15'] = df['speed'].shift(15).fillna(method = 'bfill').copy()
df['lag_30'] = df['speed'].shift(30).fillna(method = 'bfill').copy()
df['lag_45'] = df['speed'].shift(45).fillna(method = 'bfill').copy()
df['lag_60'] = df['speed'].shift(60).fillna(method = 'bfill').copy()
df['lag_1_week'] = df['speed'].shift(7*24*60).fillna(method = 'bfill').copy() # this method is an issue for a whole week. It will highly affect the R2

print(df.head(60))

# naive r2 for 5 and 10 mins
r2_1 = r2_score(df.speed, df.lag_1)
r2_5 = r2_score(df.speed, df.lag_5)
r2_10 = r2_score(df.speed, df.lag_10)
r2_15 = r2_score(df.speed, df.lag_15)
r2_30 = r2_score(df.speed, df.lag_30)
r2_45 = r2_score(df.speed, df.lag_45)
r2_60 = r2_score(df.speed, df.lag_60)
r2_1_week = r2_score(df.speed, df.lag_1_week)
mae_1 = mean_absolute_error(df.speed, df.lag_1)
mae_5 = mean_absolute_error(df.speed, df.lag_5)
mae_10 = mean_absolute_error(df.speed, df.lag_10)
mae_15 = mean_absolute_error(df.speed, df.lag_15)
mae_30 = mean_absolute_error(df.speed, df.lag_30)
mae_45 = mean_absolute_error(df.speed, df.lag_45)
mae_60 = mean_absolute_error(df.speed, df.lag_60)
mae_1_week = mean_absolute_error(df.speed, df.lag_1_week)


naive_baseline = (f'NAIVE BASELINE\n\
     1m lag || MAE: {mae_1:.2f} kph || R2: {r2_1:.2f}\n\
     5m lag || MAE: {mae_5:.2f} kph || R2: {r2_5:.2f}\n\
    10m lag || MAE: {mae_10:.2f} kph || R2: {r2_10:.2f}\n\
    15m lag || MAE: {mae_15:.2f} kph || R2: {r2_15:.2f}\n\
    30m lag || MAE: {mae_30:.2f} kph || R2: {r2_30:.2f}\n\
    45m lag || MAE: {mae_45:.2f} kph || R2: {r2_45:.2f}\n\
    60m lag || MAE: {mae_60:.2f} kph || R2: {r2_60:.2f}\n\
     1w lag || MAE: {mae_1_week:.2f} kph || R2: {r2_1_week:.2f}')

print(naive_baseline)

