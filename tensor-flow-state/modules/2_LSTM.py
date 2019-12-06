# Import libraries
import os
import pandas as pd
import numpy as np
# Display and Plotting

#import matplotlib as mpl
import matplotlib.pyplot as plt
#import seaborn as sns

# ML
import tensorflow as tf
from tensorflow import keras
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

# Load feather (1m data resolution)
fname =  os.path.join(datadir, 'RWS01_MONIBAS_0021hrl0414ra_jun_oct_repaired.feather')
df = pd.read_feather(fname) 
df.set_index('timestamp', inplace = True, drop = True)

# Make new 10m aggregate df
df_10m = df.resample('10T').mean()

# Add speed limit information
df_10m['speed_limit'] = np.where((df_10m.index.hour < 19) & (df_10m.index.hour >= 6), 100, 120)

# Find mean and sd for training batch
mean100_10m, mean120_10m = df_10m[: -(31 * 24 * 6)].groupby(['speed_limit']).mean().unstack().values
sd100_10m, sd120_10m = df_10m[: -(31 * 24 * 6)].groupby(['speed_limit']).std().unstack().values

# Normalize speed on training batch mean/sd (10m)
df_10m['speed_normalized'] = np.where(df_10m.speed_limit == 100, (df_10m.speed - mean100_10m) / sd100_10m, (df_10m.speed - mean120_10m) / sd120_10m)

# Plot Normalized distribution of training batch
df_10m.iloc[: -(31 * 24 * 6), df_10m.columns.get_loc('speed_normalized')].hist(bins = int(df_10m.speed.max() + 1))


# Generates sequential 3D batches to feed to the model processes
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
            # Set rows from i to 
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

def test_generator(data, lookback, delay, min_index = 0, max_index = None,
                   shuffle = False, batch_size = 128, step = 1):
    if max_index is None:
        max_index = len(data) - delay - 1

    ## Shift the starting index
    nbatch = (max_index - min_index - lookback) // batch_size
    shift = max_index - min_index - lookback - nbatch * batch_size
    min_index_trunc = min_index + shift + lookback - 1

    i = min_index_trunc
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index_trunc, max_index, size = batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index_trunc
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][0]
        yield samples, targets

# data = np.array(df[['speed_normalized']])
data = np.array(df_10m[['speed_normalized']])

lookback = 3*6
delay = 1
min_index_train = 0
max_index_train = len(data) - 31 * 24 * 6
min_index_val = max_index_train
max_index_val = min_index_val + 17 * 24 * 6
min_index_test = max_index_val
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

val_gen = generator(data,
                    lookback = lookback,
                    delay = delay,
                    min_index = min_index_val,
                    max_index = max_index_val,
                    step = step,
                    batch_size = batch_size)

test_gen = test_generator(data,
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
        preds = samples[:, -1, 0] # Changed index position [2] from 1 to 0, since its shape was 1
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))
    
evaluate_naive_method()


tf.keras.backend.set_floatx('float32')

# Train Stacked LSTM with Dropout
model = Sequential()
model.add(layers.LSTM(32, 
                      dropout = 0.1,
                      recurrent_dropout = 0.5,
                      return_sequences = True,
                      input_shape = (None, data.shape[-1])))
model.add(layers.LSTM(64,
                      dropout = 0.2,
                      recurrent_dropout = 0.5,
                      input_shape = (None, data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer = RMSprop(learning_rate = 0.001), loss = 'mae')
history = model.fit(train_gen,
                              steps_per_epoch = ((max_index_train + 1) // batch_size),
                              epochs = 50,
                              validation_data = val_gen,
                              validation_steps = val_steps)


# PLOT training
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()




# save model
model.save('./models/insert_whatever.h5')


model.summary()
# Evaluate
score = model.evaluate(test_gen, steps = test_steps)
print(score)



# model.save('LSTM_Stacked_Dropout_1w_look_10m_res_fix.h5')



preds = model.predict(test_gen, steps = test_steps)
predictions = df_10m[-len(preds):].rename(columns = {'speed': 'actual'})
predictions['prediction_normalized'] = preds
date_index = pd.date_range('2019-10-25 18:40:00', periods = 897, freq='10T')
predictions = predictions.reindex(date_index)
predictions['predictions_normalized'] = predictions['prediction_normalized'].shift(1)
predictions['predicted'] = np.where(predictions.speed_limit == 100, (predictions.predictions_normalized * sd100_10m + mean100_10m), (predictions.predictions_normalized * sd120_10m + mean120_10m))
predictions.fillna(method = 'bfill', inplace = True)
predictions[:288][['speed_limit', 'actual', 'predicted']].plot()





















