# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:42:16 2019

@author: peterpiontek
"""
# Import base libraries
import os
import pandas as pd
import numpy as np

# Stats
from sklearn.metrics import r2_score

# Display and Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Set working dir
os.chdir("C:/Users/peterpiontek/Google Drive/tensor-flow-state/tensor-flow-state")

# Import homebrew
from modules.repair_time_series import repair_time_series

# Define directories
datadir = "./data/"
plotdir = "./plots/"

# Don't limit, truncate or wrap columns displayed by pandas
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 200) # Accepted line width before wrapping
pd.set_option('display.max_colwidth', -1)
# Display decimals instead of scientific with pandas
pd.options.display.float_format = '{:.2f}'.format

plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['image.cmap'] = 'viridis'

###############################################################################

# Read timeseries to dataframe
timeseries = pd.read_pickle(datadir + 'RWS01_MONIBAS_0021hrl0414ra_jun_oct.pkl')[['timestamp', 'date', 'speed', 'flow']]

# Repair timeseries (add missing timestamps and remove duplicates)
timeseries = repair_time_series(dataframe = timeseries, timestamp_col = 'timestamp', cols_not_to_fill = ['date', 'speed', 'flow'],
                      fillna_method = 'pad', freq = 'T')

# Fill in missing date vals
timeseries['date'] =  timeseries.timestamp.dt.date

# Plot timeseries
timeseries['speed'].plot(title = 'Speed (raw)')
sns.despine()
plt.tight_layout()
# Save
# plt.rcParams['agg.path.chunksize'] = 100000
plt.savefig(plotdir + 'speed_raw.png')




# timeseries_repaired = pd.read_pickle(os.path.join(datadir, 'RWS01_MONIBAS_0021hrl0414ra_jun_oct_repaired.pkl'))

# merged = pd.DataFrame()
# merged['raw'] = timeseries['speed']
# merged['complete'] = timeseries_repaired['speed']
# merged['fixed'] = np.nan
# merged.fixed[merged.raw.isna()] = merged.complete
# plt.rcParams['agg.path.chunksize'] = 50000
# merged[['raw', 'fixed']].plot(style = ['b-', 'g-'], figsize = (16, 9), title = 'Speed (imputed)')
# sns.despine()
# plt.tight_layout()
# plt.savefig(plotdir + 'Time_series_imputed.png', dpi = 600)



# August looks fairly complete so let's, make sure
timeseries['null'] = np.where(((timeseries.speed.isna()) | (timeseries.flow.isna())), 1, np.nan)
timeseries.groupby(pd.Grouper(key='timestamp', freq='M'))['null'].sum()
# timeseries['speed'].isna().sum()



# August is the most complete, so we move on with it as our test case
august = timeseries[timeseries['timestamp'].dt.month == 8][['speed']]

# Fill in the null vals for reference
august = august.assign(reference_speed = august.speed.interpolate(method = 'time'))
# Add a col with interpolated values in the missing values position (the rest null) for plotting
august = august.assign(missing = np.nan)
august.missing[august.speed.isna()] = august.reference_speed

# Plot it (red are missing)
august[['speed', 'missing']].plot(style = ['b.', 'rx'], figsize = (16, 9), alpha = 0.6, title = 'August')
sns.despine()
plt.tight_layout()
plt.savefig(plotdir + 'August.png')


# Luckily they are not all clumped, so the interpolation should be smooth

# Now drop unnecessary speed col
august.drop('speed', axis = 1, inplace = True)

# There are no null vals left since we already interpolated reference_speed.
august['reference_speed'].isnull().any()

# Randomly null out 10% of the values in a new col target_speed that we want to test interpolation methods on
august['target_speed'] = august['reference_speed'].mask(np.random.choice([True, False], size=august['reference_speed'].shape, p = [.1,.9]))

# We now have thousands of null vals
august['target_speed'].isna().sum()

# Make a new column with only the reference values that aren't in the target col, like earlier but with the larger
# set of randomly removed values
august = august.assign(missing = np.nan)
august.missing[august.target_speed.isna()] = august.reference_speed
august.info()

# Plot it (red are missing)
august[['target_speed', 'missing']].plot(style=['b.', 'rx'], figsize=(16, 9), alpha = 0.6, title = 'August /w 10% data removed')
sns.despine()
plt.tight_layout()
plt.savefig(plotdir + 'August_10p_removed.png')

# Apply different interpolation methods (this can take a little while)
# Mean interpolation
august = august.assign(FillMean = august.target_speed.fillna(august.target_speed.mean()))
# Median interpolation
august = august.assign(FillMedian = august.target_speed.fillna(august.target_speed.median()))
# Rolling mean interpolation (hourly) ! doesn't work if whole window is null
august = august.assign(RollingMean = august.target_speed.fillna(august.target_speed.rolling('H', min_periods = 1).mean()))
# Rolling median interpolation (hourly) ! doesn't work if whole winodw is null
august = august.assign(RollingMedian = august.target_speed.fillna(august.target_speed.rolling('H', min_periods = 1).median()))
# Using linear interpolation
august = august.assign(InterpolateLinear = august.target_speed.interpolate(method = 'linear'))
# Using time interpolation
august = august.assign(InterpolateTime = august.target_speed.interpolate(method = 'time'))
# Quadratic interpolation
august = august.assign(InterpolateQuadratic = august.target_speed.interpolate(method = 'quadratic'))
# Cubic interpolation
august = august.assign(InterpolateCubic = august.target_speed.interpolate(method = 'cubic'))
# Slinear (spline order 1) interpolation
august = august.assign(InterpolateSLinear = august.target_speed.interpolate(method = 'slinear'))
# Akima spline interpolation
august = august.assign(InterpolateAkima = august.target_speed.interpolate(method = 'akima'))
# Polynomial (order 5) interpolation
august = august.assign(InterpolatePoly5 = august.target_speed.interpolate(method = 'polynomial', order = 5))
# Polynomial (order 7) interpolation
august = august.assign(InterpolatePoly7 = august.target_speed.interpolate(method = 'polynomial', order = 7))
# Spline (order 3) interpolation
august = august.assign(InterpolateSpline3 = august.target_speed.interpolate(method = 'spline', order = 3))
# Spline (order 4) interpolation
august = august.assign(InterpolateSpline4 = august.target_speed.interpolate(method = 'spline', order = 4))
# Spline (order 5) interpolation
august = august.assign(InterpolateSpline5 = august.target_speed.interpolate(method = 'spline', order = 5))

# Get R2 for each method and plot rankings
results = [(method, r2_score(august.reference_speed, august[method])) for method in list(august)[august.columns.get_loc('FillMean'):]]
results_df = pd.DataFrame(np.array(results), columns=['method', 'R2'])
results_df = results_df.sort_values(by = 'R2', ascending=False).reset_index(drop = True)
results_df.index += 1
print(results_df)


# august['InterpolateTime'].plot(style=['b-'], figsize=(16, 9), alpha = 0.6, title = 'August (time interpolated)')
# sns.despine()
# plt.tight_layout()
# plt.savefig(plotdir + 'August_time_interpolated.png')









