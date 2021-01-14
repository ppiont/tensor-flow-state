# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 17:58:46 2020

@author: peterpiontek
"""

import pandas as pd
df = pd.read_csv("data/df_imputed_week_shift.csv", index_col = 0, parse_dates = True)


cols = ["speed", "flow", "speed_-2", "speed_-1","speed_+1", "speed_+2", "flow_-2", "flow_-1", "flow_+1", "flow_+2", "speed_limit"]
speed_cols = ["speed", "speed_-2", "speed_-1","speed_+1", "speed_+2"]
flow_cols = ["flow", "flow_-2", "flow_-1", "flow_+1", "flow_+2"]

import numpy as np
def resample_df(df, freq = "10T", method_speed = np.median, method_flow = np.sum):
    copied = df.copy()
    copied = copied.resample(freq).agg({
           "speed": method_speed, "speed_-2": method_speed, "speed_-1": method_speed, "speed_+1": method_speed, "speed_+2": method_speed,
           "flow": method_flow, "flow_-2": method_flow, "flow_-1": method_flow, "flow_+1": method_flow, "flow_+2": method_flow,
           "speed_limit": np.median})
    return copied

r_df = resample_df(df, freq = "10T")

def train_val_test_split(df, val_year, test_year):
    # train, test
    return df[(df.index.year != val_year) & (df.index.year != test_year)], df[df.index.year == val_year], df[df.index.year == test_year]

train, val, test = train_val_test_split(r_df, 2018, 2019)

import numpy as np
def log_transform(df):
    copy = df.copy()
    return np.log(copy.iloc[:, :-1].replace(0, 1e-15)).join(df.iloc[:, -1], how = 'inner')

# Log transform. First set 0s to very low value 'cause you can't log 0.
train_log = log_transform(train)
val_log = log_transform(val)

def calc_mean(df, col):
    # mean(100), mean(120)
    return df.groupby(['speed_limit'])[col].mean().values

def calc_sd(df, col):
    # sd(100), sd(120)
    return df.groupby(['speed_limit'])[col].std().values

def normalize_df(df, cols):
    copy = df.copy()
    for col in cols:
        # Find mean and sd for column
        mean100, mean120 = calc_mean(copy, col)
        sd100, sd120 = calc_sd(copy, col)
        copy[col] = np.where(copy.speed_limit == 100, (copy[col] - mean100) / sd100, (copy[col] - mean120) / sd120)
    return copy

train_norm = normalize_df(train_log, cols[:-1])
val_norm = normalize_df(val_log, cols[:-1])

# Generates sequential 3D batches to feed to the model
def generator(data, lookback, delay, min_index = 0, max_index = None, 
              shuffle = False, batch_size = 128, step = 1, target_col = 0):
    # If max index not given, subtract prediction horizon - 1 (len to index) from last data point
    if max_index is None:
        max_index = len(data) - delay - 1
    # Set i to first idx with valid lookback length behind it
    i = min_index + lookback
    while 1:
        # Use shuffle for non-sequential data
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size = batch_size)
        # Else for sequential (time series)
        else:
            # Check if adding batch exceeds index bounds
            if i + batch_size >= max_index:
                # Return i to beginning
                i = min_index + lookback
            # Select next valid row range
            rows = np.arange(i, min(i + batch_size, max_index))
            # Increment i
            i += len(rows)
        # Initialize sample and target arrays
        samples = np.zeros((len(rows),
                            lookback // step,
                            np.shape(data)[-1]))
        targets = np.zeros((len(rows),))
        # Generate samples, targets
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][target_col]
        yield samples, targets
        
        
# Generator 'settings'

train_data = train_norm.iloc[:, 0].values
train_data = np.reshape(train_data, (np.shape(train_data)[0], 1))
np.shape(train_data)


lookback = 24 * 6 # 24 hours
delay = 3 # 30 minutes
step = 1
batch_size = 128
min_index_train = 0
max_index_train = len(train_data)
train_gen = generator(train_data,
                      lookback = lookback,
                      delay = delay,
                      min_index = min_index_train,
                      max_index = max_index_train,
                      step = step, 
                      batch_size = batch_size,
                      target_col = 0)


val_data = val_norm.iloc[:, 0].values
train_data = np.reshape(train_data, (np.shape(train_data)[0], 1))
np.shape(train_data)

min_index_val = 0
max_index_val= len(val_data)
val_gen = generator(val_data,
                      lookback = lookback,
                      delay = delay,
                      min_index = min_index_val,
                      max_index = max_index_val,
                      step = step, 
                      batch_size = batch_size,
                      target_col = 0)


# ML
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop

# Train Stacked LSTM with Dropout
model = Sequential()
model.add(layers.Conv1D(filters = 32, kernel_size = (1), activation = 'relu', input_shape = (144, 1)))
model.add(layers.LSTM(32, 
                      dropout = 0.1,
                      recurrent_dropout = 0.5,
                      input_shape = (144, 32),
                      return_sequences = True))
model.add(layers.LSTM(32,
                      dropout = 0.1,
                      recurrent_dropout = 0.5,
                      input_shape = (144, 32)))
model.add(layers.Dense(1))

model.compile(optimizer = RMSprop(learning_rate = 0.001), loss = 'mse', metrics = ['mae', 'mape'])
model.build(input_shape = (144, 1))

model.summary()

history = model.fit(train_gen,
                    steps_per_epoch = ((max_index_train + 1) // batch_size),
                    epochs = 20)
