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
from modules.repair_timeseries import repair_time_series
from modules.ts_plot import correlation_plot, decompose_plot

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

###############################################################################

# Read df
df = pd.read_pickle(datadir + 'RWS01_MONIBAS_0021hrl0414ra_jun_oct.pkl')[['timestamp', 'date', 'lon', 'lat', 'sensor_id', 'speed', 'flow']]

# Repair time series
df = repair_time_series(dataframe = df, timestamp_col = 'timestamp', cols_not_to_fill = ['date', 'speed', 'flow'],
                      fillna_method = 'pad', freq = 'T')

# Fetch date vals for new rows
df['date'] = df.timestamp.dt.date


# Interpolate null vals for the first week of data of speed and flow cols
week = 7 * 24 * 60
df.iloc[:week + 1, df.columns.get_loc('speed')] = df['speed'][:week + 1].interpolate(method = 'time')
df.iloc[:week + 1, df.columns.get_loc('flow')] = df['flow'][:week + 1].interpolate(method = 'time')

# Return to RangeIndex for the next operation
df.reset_index(drop = True, inplace = True)

# Replace remaining nulls with value from 1 week previous
def shiftWeek(df):
    speed_col = df.columns.get_loc('speed')
    flow_col = df.columns.get_loc('flow')
    check = speed_col + 1
    week = 7 * 24 * 60
    for row in df.itertuples():
        if np.isnan(row[check]):
            df.iat[row[0], speed_col] = df.iat[(row[0] - week), speed_col]
            df.iat[row[0], flow_col] = df.iat[(row[0] - week), flow_col]
    return df

df = shiftWeek(df)

# Return to DateTimeIndex again
df.set_index(pd.DatetimeIndex(df.timestamp.values), inplace = True)



downsampled = df.speed.resample('D').mean()






result = smt.seasonal_decompose(downsampled, model='multiplicative')
result.plot()
plt.show()








fig = plt.figure(figsize = (12,8), dpi = 300)
y = df.speed
x = df.timestamp
plt.plot(x, signal.savgol_filter(y, window_length = (24*60*7 + 1), polyorder = 4))


downsampled = df.speed.resample('W').mean()
print(downsampled)
downsampled.plot()
# august[['target_speed', 'missing']].plot(style=['b.', 'rx'], figsize=(20, 10))


















groups = nulls.groupby(pd.Grouper(freq='W'))
for name, group in groups:
    weeks = pd.DataFrame(data = group.values, columns = [name.month])
    months.plot(subplots=True, legend=False)
    plt.show()



one = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
two = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
hello = [one, two]

print(hello[0])


fig = plt.figure(figsize=(15,10))
df
# layout = (2, 2)
# ts_ax   = plt.subplot2grid(layout, (0, 0))
# hist_ax = plt.subplot2grid(layout, (0, 1))
# acf_ax  = plt.subplot2grid(layout, (1, 0))
# pacf_ax = plt.subplot2grid(layout, (1, 1))

null_df.plot(ax=ts_ax)
ts_ax.set_title(title, fontsize=12, fontweight='bold')
y.plot(ax=hist_ax, kind='hist', bins=25)
hist_ax.set_title('Histogram')
smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
sns.despine()
plt.tight_layout()
plt.show()



[print(null_df)]






# Make ts, hist, ac and pac plots
decomposePlot(df['speed'], 'Speed')
decomposePlot(df['flow'], 'Flow')


decomposePlot(df['speed'][:7*24*60], 'Speed')

























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

