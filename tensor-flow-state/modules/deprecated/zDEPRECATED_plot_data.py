### Data visualization ###

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
sns.set(context='paper', style='white')
#print(plt.style.available)


os.chdir("C:/Users/peterpiontek/Google Drive/tensor-flow-state/tensor-flow-state")

# directories
datadir = "./data/"
plotdir = "./plots/"

# read data
df = pd.read_pickle(datadir + '3months_weather.pkl')

x = df.datetime
y = df.flow
y2 = df.speed

# calculate moving average
def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

f_avg = movingaverage(y, 24*60)
s_avg = movingaverage(y2, 24*60)


# initialize shared x axis plot
fig, ax = plt.subplots()
# plot y1 (avg flow)
ax.plot(x, f_avg)
ax.set_ylabel('Flow (day avg)')
ax.set_ylim([1500,6000])
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
# create second ax on same x-axis
ax2 = ax.twinx()
# plot y2 (avg speed)
ax2.plot(x, s_avg, 'r-')
ax2.set_ylabel('Speed (day avg)', color='r')
ax2.set_ylim([60, 120])
for tl in ax2.get_yticklabels():
    tl.set_color('r')
    
#plt.savefig(plotdir + 'flow_vs_speed_3_months_sharedx.png', dpi=600)


# initialize 2 subplots figure
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, squeeze=True)
# plot y1 (avg flow)
ax[0].plot(x, f_avg)
ax[0].set_ylabel('Flow (day avg)')
ax[0].set_ylim([1500,6000])
#for tick in ax[0].get_xticklabels():
#    tick.set_rotation(30)
# create second ax on same x-axis
#ax2 = ax.twinx()
# plot y2 (avg speed)
ax[1].plot(x, s_avg, 'r-')
ax[1].set_ylabel('Speed (day avg)', color='r')
ax[1].set_ylim([60, 120])
for tl in ax[1].get_yticklabels():
    tl.set_color('r')

#plt.savefig(plotdir + 'flow_vs_speed_3_months_2subs.png')








sns.distplot(df.speed)
sns.distplot(df.flow)



#s_sd = np.std(df.speed)
#s_mean = np.mean(df.speed)
#
#df_rm_outliers = df[(df['speed'] > s_mean-s_sd*3) & (df['speed'] < s_mean+s_sd*3)]















