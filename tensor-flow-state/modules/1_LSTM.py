# Import libraries
import os
import pandas as pd
import numpy as np

# Stats
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# import statsmodels.tsa.api as smt
# from scipy import signal


# Display and Plotting
import matplotlib.pyplot as plt
# import seaborn as sns

# Set working dir
os.chdir("C:/Users/peterpiontek/Google Drive/tensor-flow-state/tensor-flow-state")

# Import homebrew
from modules.fixStandardTime2019 import fixStandardTime2019

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

df = pd.read_pickle(datadir + 'RWS01_MONIBAS_0021hrl0414ra_jun_oct.pkl')

print(df.shape)
df = fixStandardTime2019(df)
print(df.shape)

print(df[df['date'] == '2019-10-15']['timestamp'])

# Make sure the series to pass to DateTimeIndex is explicitly datetime
datetime_series = pd.to_datetime(df['timestamp'])
# Create DateTimeIndex passing the datetime series
datetime_index = pd.DatetimeIndex(datetime_series.values)
# Set the new index
df.set_index(datetime_index, inplace = True)

# Create a date_range for reindexing and filling missing dates
new_index = pd.date_range(start = df.index[0], end = df.index[-1], freq = 'T')
# Reindex 
df = df.reindex(new_index)


print(df[df.duplicated(subset = 'timestamp', keep=False)])




print(test)


# # Define features and target
# features = ['timestamp', 'speed']
# target = ['speed']


# Create train and test df

train_data = df[df['date'] < '2019-10']
test_data = df[df['date'] > '2019-09']


train_data_10min = train_data.resample('10T').agg({'speed': 'mean', 'flow': 'sum'})
train_data_day = train_data.resample('D').agg({'speed': 'mean', 'flow': 'sum'})


# df.groupby(['name', pd.Grouper(key='date', freq='M')])['ext price'].sum()


# groups = plotting_data.groupby(pd.Grouper(freq='W'))  #key='weekday'))['speed'].mean().sort_index()
# weeks = pd.DataFrame()
# for name, group in groups:
# 	weeks[name] = group.values
# weeks.plot(subplots=True, legend=False)
# plt.show()


plot_test_10min = plot_test.resample('H').agg({'speed': 'mean', 'flow': 'sum'})

fig, ax = plt.subplots(figsize=(8,6));
ax.plot(plot_test_10min['timestamp'], plot_test['speed'])
fig.tight_layout();


train_data_day['speed'].plot(ax=ax)
ax.set_title('10m avg speed, Jun-Oct')
ax.set_ylabel('Kph');
ax.set_xlabel('Time');
ax.xaxis.set_ticks_position('bottom')
fig.tight_layout();








# create train and test sets
X_train, y_train, X_test, y_test = train_data[features], train_data[target], \
                                test_data[features],test_data[target]

# rescale features
mm_xscaler = preprocessing.MinMaxScaler()
X_train_minmax = mm_xscaler.fit_transform(X_train)
X_test_minmax = mm_xscaler.transform(X_test)
# rescale target
mm_yscaler = preprocessing.MinMaxScaler()
y_train_minmax = mm_yscaler.fit_transform(y_train)
y_test_minmax = mm_yscaler.transform(y_test)


## for univariate LSTM
#full_y = np.concatenate((y_train_minmax, y_test_minmax), axis=0)


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


# choose a number of time steps
n_steps = 60
# split into samples
X, y = split_sequence(y_train_minmax, n_steps)


# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
n_features = 1
n_seq = 2
n_steps = 2
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))

# split y_test into "Xt, yt"
n_steps = 4
Xt, yt = split_sequence(y_test_minmax, n_steps)
n_steps = 2
Xt = Xt.reshape((Xt.shape[0], n_seq, n_steps, n_features))

n_features=1

### SET UP MODEL ###
model = Sequential([
    TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu',
                           input_shape=(None, n_steps, n_features))),
    TimeDistributed(MaxPooling1D(pool_size=2)),
    TimeDistributed(Flatten()),
    LSTM(50, activation='relu'),
    Dense(1)
    ])
    
# compile the model
model.compile(optimizer='adagrad', loss='mse')

# fit the model
model.fit(X, y, epochs=10, verbose=1, validation_data=(Xt, yt))

# show a summary of the model architecture
model.summary()

yhat = model.predict(Xt)


