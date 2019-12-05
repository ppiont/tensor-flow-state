# Import libraries
import os
import pandas as pd
import numpy as np
# Display and Plotting

#import matplotlib as mpl
import matplotlib.pyplot as plt
#import seaborn as sns

# ML
#import tensorflow as tf
#from tensorflow import keras
#from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop

# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")

# Set working dir
os.chdir("C:/Users/peterpiontek/Google Drive/tensor-flow-state/tensor-flow-state")

# Set pandas and matplotlib settings
exec(open('modules/Settings.py').read())

# Define directories
datadir = "./data/"
plotdir = './plots/'

# Load feather
fname =  os.path.join(datadir, 'RWS01_MONIBAS_0021hrl0414ra_jun_oct_repaired.feather')
df = pd.read_feather(fname) 
df.set_index('timestamp', inplace = True, drop = True)

df_10m = df.resample('10T').mean()

# Add speed limit information
df['speed_limit'] = np.where((df.index.hour < 19) & (df.index.hour >= 6), 100, 120)
df_10m['speed_limit'] = np.where((df_10m.index.hour < 19) & (df_10m.index.hour >= 6), 100, 120)

# Find mean and sd for training batch
# mean100, mean120 = df[: -(31 * 24 * 60)].groupby(['speed_limit']).mean().unstack().values
# sd100, sd120 = df[: -(31 * 24 * 60)].groupby(['speed_limit']).std().unstack().values
mean100, mean120 = df_10m[: -(31 * 24 * 6)].groupby(['speed_limit']).mean().unstack().values # 10 min agg
sd100, sd120 = df_10m[: -(31 * 24 * 6)].groupby(['speed_limit']).std().unstack().values # 10 min agg


# Normalize speed on training batch mean/sd
df['speed_normalized'] = np.where(df.speed_limit == 100, (df.speed - mean100) / sd100, (df.speed - mean120) / sd120)
# df['mean'] = np.where(df.speed_limit == 100, mean100, mean120)
# df['sd'] = np.where(df.speed_limit == 100, sd100, sd120)
# Plot Normalized distribution of training batch
df.iloc[: -(31 * 24 * 60), df.columns.get_loc('speed_normalized')].hist(bins = df.speed.max() + 1)

# Normalize speed on training batch mean/sd (10m)
df_10m['speed_normalized'] = np.where(df_10m.speed_limit == 100, (df_10m.speed - mean100) / sd100, (df_10m.speed - mean120) / sd120)
# df['mean'] = np.where(df.speed_limit == 100, mean100, mean120)
# df['sd'] = np.where(df.speed_limit == 100, sd100, sd120)
# Plot Normalized distribution of training batch
df_10m.iloc[: -(31 * 24 * 6), df_10m.columns.get_loc('speed_normalized')].hist(bins = int(df_10m.speed.max() + 1))





# Generates sequential 3D batches to feed to the model fitter
def generator(data, lookback, delay, min_index = 0, max_index = None, 
              shuffle = False, batch_size = 128, step = 1):
    # if max index not given, subtract prediction horizon - 1 (len to index) from last data point
    if max_index is None:
        max_index = len(data) - delay - 1
    # set i to first idx with valid lookback length behind it
    i = min_index + lookback
    while 1:
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
                            np.shape(data)[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][0] # Change col index ([0]) if reshaping 
        yield samples, targets


# data = np.array(df[['speed_normalized']])
data = np.array(df_10m[['speed_normalized']])

lookback = 6*24*7
delay = 1
min_index_train = 0
max_index_train = int(len(data) - (31 * 24 * 60) / 10)
min_index_val = max_index_train + 1
max_index_val = int(min_index_val + ((17 * 24 * 60) / 10))
min_index_test = max_index_val + 1
max_index_test = None
step = 1
batch_size = 128

train_gen = generator(data,
                      lookback = lookback,
                      delay = delay,
                      min_index = min_index_train,
                      max_index = max_index_train,
                      step = step, 
                      batch_size = batch_size)

# control = 0
# while control < 5:
#     for samples, targets in train_gen:
#             print(samples, targets)
#             print(samples.shape, targets.shape)
#             control += 1

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
                     max_index = max_index_test,
                     step = step,
                     batch_size = batch_size)

# This is how many steps to draw from `val_gen`
# in order to see the whole validation set:
val_steps = (max_index_val - min_index_val - lookback) // batch_size

# This is how many steps to draw from `test_gen`
# in order to see the whole test set:
test_steps = (len(data) - min_index_test - lookback) // batch_size


# Evaluate vs 1 step previous (naive baseline)
def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 0] # Changed index position [2] from 1 to 0, since its shape was 1
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))
    
evaluate_naive_method()





# MLP
model = Sequential()
model.add(layers.Flatten(input_shape = (lookback // step, data.shape[-1])))
model.add(layers.Dense(32, activation = 'relu'))
model.add(layers.Dense(1))

model.compile(optimizer = RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch = 500,
                              epochs = 20,
                              validation_data = val_gen,
                              validation_steps = val_steps)

# GRU
model = Sequential()
model.add(layers.GRU(32, input_shape = (None, data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer = RMSprop(), loss = 'mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch = 500,
                              epochs = 20,
                              validation_data = val_gen,
                              validation_steps = val_steps)

# GRU Dropout
model = Sequential()
model.add(layers.GRU(32,
                     dropout = 0.2,
                     recurrent_dropout = 0.2,
                     input_shape = (None, data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer = RMSprop(), loss = 'mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch = 500,
                              epochs = 40,
                              validation_data = val_gen,
                              validation_steps = val_steps)

# Train LSTM
model = Sequential()
model.add(layers.LSTM(32, input_shape = (None, data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer = RMSprop(), loss = 'mae')
history = model.fit(train_gen,
                              steps_per_epoch = ((max_index_train + 1) // batch_size),
                              epochs = 20,
                              validation_data = val_gen,
                              validation_steps = val_steps)

# Train Stacked LSTM with Dropout
model = Sequential()
model.add(layers.LSTM(32, 
                      dropout = 0.1,
                      recurrent_dropout = 0.5,
                      return_sequences = True,
                      input_shape = (None, data.shape[-1])))
model.add(layers.LSTM(64,
                      dropout = 0.1,
                      recurrent_dropout = 0.5,
                      input_shape = (None, data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer = RMSprop(), loss = 'mae')
history = model.fit(train_gen,
                              steps_per_epoch = ((max_index_train + 1) // batch_size),
                              epochs = 30,
                              validation_data = val_gen,
                              validation_steps = val_steps)


# PLOT
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()



# Evaluate
score = model.evaluate(test_gen, steps = 7)
print(score)



model.save('LSTM_Stacked_Dropout_1w_look_10m_res_fix.h5')



preds = model.predict(test_gen, steps = 7)
preds[0:10]





predictions = pd.DataFrame(data[-896-110:-110], columns = ['actual'])
predictions['prediction'] = preds

predictions.plot()

# ### SET UP MODEL ###
# model = Sequential([
#     TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu',
#                            input_shape=(None, n_steps, n_features))),
#     TimeDistributed(MaxPooling1D(pool_size=2)),
#     TimeDistributed(Flatten()),
#     LSTM(50, activation='relu'),
#     Dense(1)
#     ])
    
# # compile the model
# model.compile(optimizer='adagrad', loss='mse')

# # fit the model
# model.fit(X, y, epochs=10, verbose=1, validation_data=(Xt, yt))

# # show a summary of the model architecture
# model.summary()

# yhat = model.predict(Xt)

