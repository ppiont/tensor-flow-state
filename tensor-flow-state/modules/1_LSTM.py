# Import libraries
import os
import pandas as pd
import numpy as np

# Display and Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# ML
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import r2_score, mean_absolute_error

# Set working dir
os.chdir("C:/Users/peterpiontek/Google Drive/tensor-flow-state/tensor-flow-state")

# Import homebrew
# from modules.repair_time_series import repair_time_series

# import modules.Settings as settings
# settings.dir_settings()

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

plt.rcParams['figure.figsize'] = (16, 9)

# # Load pickle
# pname =  os.path.join(datadir, 'RWS01_MONIBAS_0021hrl0414ra_jun_oct_repaired.pkl')
# df = pd.read_pickle(pname)
# # df = df[['speed]]

# Load feather
fname =  os.path.join(datadir, 'RWS01_MONIBAS_0021hrl0414ra_jun_oct_repaired.feather')
df = pd.read_feather(fname) 
df.set_index('timestamp', inplace = True, drop = True)

# Add speed limit information
df['speed_limit'] = np.where((df.index.hour < 19) & (df.index.hour >= 6), 100, 120)
df['speed_cat_norm'] = df.speed - df.speed_limit
df['speed_std_scaled'] = (df.speed - df.speed.mean()) / df.speed.std()
df.speed_limit = pd.Categorical(df.speed_limit, categories=[100, 120])

# df.speed.hist(bins = df.speed.max() + 1)
# df.speed_cat_norm.hist(bins = df.speed.max() + 1)
# df.speed_std_scaled.hist(bins = df.speed.max() + 1)


# example
# new month (november) is test
# jun-oct is train/val


arr = np.array(df['speed_cat_norm'])

def generator(data, lookback, delay, min_index = 0, max_index = None, 
              shuffle = False, batch_size = 128, step = 1):
    # if max index not given, subtract prediction horizon - 1 (len to index) from last data point
    if max_index is None:
        max_index = len(data) - delay - 1
    # set i to first idx with valid lookback length behind it
    i = min_index + lookback
    while True:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size = batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
            
    samples = np.zeros((len(rows),
                        lookback // step,
                        data.shape[-1]))
    targets = np.zeros((len(rows),))
    for j, row in enumerate(rows):
        indices = range(rows[j] - lookback, rows[j] - step)
        samples[j] = data[indices]
        targets[j] = data[rows[j] + delay][1]
    yield samples, targets
    


samples, targets = generator(data = arr, lookback = 10, delay = 10, max_index = len(arr) - 60 * 24 * 31)






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

