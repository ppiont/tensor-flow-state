# Import libraries
import os
import pandas as pd
import numpy as np

# Display and Plotting
import matplotlib.pyplot as plt

# ML
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop

# Set working dir
os.chdir("C:/Users/peterpiontek/Google Drive/tensor-flow-state/tensor-flow-state")

# Define directories
datadir = "./data/"

# Load dataframe
df = pd.read_feather(os.path.join(datadir, 'data.feather'))
# Set index
df.set_index('timestamp', inplace = True, drop = True)

# Isolate timeseries for use with the NN
data = np.array(df[['speed_standardized']])

# Generates sequential 3D batches to feed to the model
def generator(data, lookback, delay, min_index = 0, max_index = None, 
              shuffle = False, batch_size = 128, step = 1):
    # if max index not given, subtract prediction horizon - 1 (len to index) from last data point
    if max_index is None:
        max_index = len(data) - delay - 1
    # set i to first idx with valid lookback length behind it
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
                # return i to beginning
                i = min_index + lookback
            # select next valid row range
            rows = np.arange(i, min(i + batch_size, max_index))
            # increment i
            i += len(rows)
        # initialize sample and target arrays
        samples = np.zeros((len(rows),
                            lookback // step,
                            np.shape(data)[-1]))
        targets = np.zeros((len(rows),))
        # 
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][0] # Col index should point to target col -- currently ([0])
        yield samples, targets

# Generator 'settings'
lookback = 24 * 6 # 24 hours
delay = 1 # 10 minutes
step = 1
batch_size = 128
min_index_train = 0
max_index_train = len(data) - 31 * 24 * 6 # last month is reserved for val/test
min_index_val = max_index_train
max_index_val = min_index_val + 17 * 24 * 6 # 17 days val
min_index_test = max_index_val 
max_index_test = None # 14 days test

train_gen = generator(data,
                      lookback = lookback,
                      delay = delay,
                      min_index = min_index_train,
                      max_index = max_index_train,
                      step = step, 
                      batch_size = batch_size)

val_gen = generator(data,
                    lookback = lookback,
                    delay = delay,
                    min_index = min_index_val,
                    max_index = max_index_val,
                    step = step,
                    batch_size = batch_size)

test_gen = generator(data,
                     lookback = lookback,
                     delay = delay,
                     min_index = min_index_test,
                     max_index = None,
                     step = step,
                     batch_size = batch_size)

val_steps = (max_index_val - min_index_val - lookback) // batch_size
test_steps = (len(data) - min_index_test - lookback) // batch_size

# Evaluate vs 1 step previous (naive baseline)
def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 0]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print("MAE: ", np.mean(batch_maes))
    
evaluate_naive_method()
# 0.96

# Train Stacked LSTM with Dropout
model = Sequential()
model.add(layers.LSTM(32, 
                      dropout = 0.1,
                      recurrent_dropout = 0.5,
                      batch_input_shape = (128, 24 * 6, 1),
                      # input_shape = (None, data.shape[-1]),
                      stateful = True,
                      return_sequences = True))
model.add(layers.LSTM(64,
                      dropout = 0.1,
                      recurrent_dropout = 0.5,
                      batch_input_shape = (128, 24 * 6, 32),
                      # input_shape = (None, data.shape[-1]),
                      stateful = True))
model.add(layers.Dense(1))

model.summary()

model.compile(optimizer = RMSprop(learning_rate = 0.001), loss = 'mse', metrics = ['mae', 'mape'])
history = model.fit(train_gen,
                    steps_per_epoch = ((max_index_train + 1) // batch_size),
                    epochs = 20,
                    validation_data = val_gen,
                    validation_steps = val_steps)


# Plot training
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Evaluate
score = model.evaluate(test_gen, steps = test_steps)
print(score)