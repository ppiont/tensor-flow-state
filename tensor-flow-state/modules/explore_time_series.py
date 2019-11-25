# Import libraries
import os
import pandas as pd
import numpy as np

# Stats
import statsmodels.tsa.api as smt

# Display and Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Set working dir
os.chdir("C:/Users/peterpiontek/Google Drive/tensor-flow-state/tensor-flow-state")

# Import homebrew
from modules.repair_time_series import repair_time_series
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

################################## Repair TS ##################################

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

# df.to_pickle(os.path.join(datadir, 'RWS01_MONIBAS_0021hrl0414ra_jun_oct_repaired.pkl'))

###############################################################################

# Plot time series with 4 sampling ranges
y = df.speed.copy()

fig =  plt.figure(figsize = (16, 9))
fig.suptitle('RWS01_MONIBAS_0021hrl0414ra', fontsize = 12, fontweight = 'bold', y = 1.04)
layout = (4, 1)
min_ax = plt.subplot2grid(layout, (0, 0))
min_ax.set_title('Minutes')
hour_ax = plt.subplot2grid(layout, (1, 0))
hour_ax.set_title('Hours')
day_ax = plt.subplot2grid(layout, (2, 0))
day_ax.set_title('Days')
week_ax = plt.subplot2grid(layout, (3, 0))
week_ax.set_title('Weeks')
min_ax.get_shared_x_axes().join(min_ax, hour_ax, day_ax, week_ax)
y.plot(ax = min_ax)
y.resample('H').mean().plot(ax = hour_ax)
y.resample('D').mean().plot(ax = day_ax)
y.resample('W').mean().plot(ax = week_ax)
sns.despine()
plt.tight_layout()
fig.savefig(plotdir + 'Time_series__4_samplings__RWS01_MONIBAS_0021hrl0414ra__jun_oct.svg', dpi = 600)
plt.show()


# Plot seasonal decomposition
y_hour = y.resample('H').mean()
y_day = y.resample('D').mean()
y_week = y.resample('W').mean()

decomp = smt.seasonal_decompose(y_hour, model = 'additive', freq = 24 * 7)
fig =  plt.figure(figsize = (16, 9))
fig.suptitle(f'RWS01_MONIBAS_0021hrl0414ra decomposed: hourly vals, weekly cycle', fontsize = 12, fontweight = 'bold', y = 0.995)
layout = (4, 1)
observed_ax = plt.subplot2grid(layout, (0, 0))
observed_ax.set_ylabel('Observed')
trend_ax = plt.subplot2grid(layout, (1, 0))
trend_ax.set_ylabel('Trend')
seasonal_ax = plt.subplot2grid(layout, (2, 0))
seasonal_ax.set_ylabel('Seasonal')
residual_ax = plt.subplot2grid(layout, (3, 0))
residual_ax.set_ylabel('Residual')
observed_ax.get_shared_x_axes().join(observed_ax, trend_ax, seasonal_ax, residual_ax)
decomp.observed.plot(ax = observed_ax)
decomp.trend.plot(ax = trend_ax)
decomp.seasonal.plot(ax = seasonal_ax)
decomp.resid.plot(ax = residual_ax)
sns.despine()
plt.tight_layout()
plt.savefig(plotdir + 'Time_series_decomposition__RWS01_MONIBAS_0021hrl0414ra__jun_oct__weekly_seasonality.svg', dpi = 600)
plt.show()


# Make ts, hist, ac and pac plots
fig = correlation_plot(df['speed'], 'Speed', figsize = (16,9))[0]
fig.savefig(plotdir + 'Time_series_ACF_PACF_HIST__RWS01_MONIBAS_0021hrl0414ra__jun_oct.png', dpi = 600)
plt.show()

