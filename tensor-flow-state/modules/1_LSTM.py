# Import libraries
import os
import pandas as pd
import numpy as np


# Stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
from scipy import signal

# Display and Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Set working dir
os.chdir("C:/Users/peterpiontek/Google Drive/tensor-flow-state/tensor-flow-state")

# Import homebrew
from modules.repairTimeSeries import repairTimeSeries
from modules.tsPlot import tsPlot

# Define directories
datadir = "./data/"
plotdir = './plots/'

# Don't limit, truncate or wrap columns displayed by pandas
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 200) # accepted line width before wrapping
pd.set_option('display.max_colwidth', -1)
# Display decimals instead of scientific with pandas
pd.options.display.float_format = '{:.2f}'.format

###############################################################################

# Read df
df = pd.read_pickle(datadir + 'RWS01_MONIBAS_0021hrl0414ra_jun_oct.pkl')[['timestamp', 'date', 'lon', 'lat', 'sensor_id', 'speed', 'flow']]

# Repair time series
df = repairTimeSeries(df, 'timestamp', 'pad')

# Fetch date vals for new rows
df['date'] = df.timestamp.dt.date

# Find cols with null vals to be filled statically (excludes speed and flow)
nan_cols = [col for col in df.columns[df.isna().any()].tolist() if col not in ('speed', 'flow')]

# Fill said cols with static vals
df[nan_cols] = df[nan_cols].fillna('pad')


# Interpolate the first week of data for speed and flow
df.iloc[:7 * 24 * 60 + 1, df.columns.get_loc('speed')] = df['speed'][:7 * 24 * 60 + 1].interpolate(method = 'linear')
df.iloc[:7 * 24 * 60 + 1, df.columns.get_loc('flow')] = df['flow'][:7 * 24 * 60 + 1].interpolate(method = 'linear')


# Replace remaining nulls with value from 1 week previous
speed_col = df.columns.get_loc('speed'); flow_col = df.columns.get_loc('flow'); check = speed_col + 1; week = 7*24*60

for row in df.itertuples():
    if np.isnan(row[check]):
        df.iat[row[0], speed_col] = df.iat[(row[0] - week), speed_col]
        df.iat[row[0], flow_col] = df.iat[(row[0] - week), flow_col]


# Make ts, hist, ac and pac plots
tsPlot(df['speed'], 'Speed')
tsPlot(df['flow'], 'Flow')


tsPlot(df['speed'][:7*24*60], 'Speed')

















speed_col = df.columns.get_loc('speed')
flow_col = df.columns.get_loc('flow')
# For remaining nulls, take value from previous week at same time
for i in np.arange(df.shape[0])[df['speed'].isna()]:
    df.iloc[i, speed_col] = df.iat[i - (7*24*60), speed_col]
    df.iloc[i, flow_col] = df.iat[i - (7*24*60), flow_col]
    
def func(x):
    if np.isnan(x):
        return ...
    else: return x
df.speed.apply(func)


test = df.interpolate(method = 'time', axis = 0)

nan_val = df[df['speed'].isna()].speed

test = df['speed'].shift(freq='D')

print(df[(df['timestamp'] > pd.to_datetime('2019-09-14')) &  (df['timestamp'] < pd.to_datetime('2019-09-16'))])
print(df[df['date'] == '2019-10-15'])
print(df[df.duplicated(subset = 'timestamp', keep=False)])





















# # # Define features and target
# # features = ['timestamp', 'speed']
# # target = ['speed']


# # Create train and test df

# train_data = df[df['date'] < '2019-10']
# test_data = df[df['date'] > '2019-09']


# train_data_10min = train_data.resample('10T').agg({'speed': 'mean', 'flow': 'sum'})
# train_data_day = train_data.resample('D').agg({'speed': 'mean', 'flow': 'sum'})


# # df.groupby(['name', pd.Grouper(key='date', freq='M')])['ext price'].sum()


# # groups = plotting_data.groupby(pd.Grouper(freq='W'))  #key='weekday'))['speed'].mean().sort_index()
# # weeks = pd.DataFrame()
# # for name, group in groups:
# # 	weeks[name] = group.values
# # weeks.plot(subplots=True, legend=False)
# # plt.show()


# plot_test_10min = plot_test.resample('H').agg({'speed': 'mean', 'flow': 'sum'})

# fig, ax = plt.subplots(figsize=(8,6));
# ax.plot(plot_test_10min['timestamp'], plot_test['speed'])
# fig.tight_layout();


# train_data_day['speed'].plot(ax=ax)
# ax.set_title('10m avg speed, Jun-Oct')
# ax.set_ylabel('Kph');
# ax.set_xlabel('Time');
# ax.xaxis.set_ticks_position('bottom')
# fig.tight_layout();

