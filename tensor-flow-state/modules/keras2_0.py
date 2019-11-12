### keras playground 2.0 LSTM (univariate)###

import os
import pickle
import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, BatchNormalization, Dropout, \
                         LeakyReLU, LSTM
from keras import losses
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn import preprocessing

# set working dir
os.chdir("C:/Users/peterpiontek/Google Drive/tensor-flow-state/tensor-flow-state")

# import homebrew


# directories
datadir = "./data/"
plotdir = "./plots/"

# read data
df = pd.read_pickle(datadir + '3months_weather.pkl')

# create train and test df
train_data = df[df['datetime'] < '2019-08-01']
test_data = df[df['datetime'] > '2019-07-31']

# define features and target
features = ['minute_sine', 'minute_cosine', 'hour_sine', 'hour_cosine', 'monday', 
            'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
            'weekend', 'holiday', 'speed', 'windspeed_avg', 'windspeed_max', 
            'temperature', 'sunduration', 'sunradiation', 'precipitationduration',
            'precipitation', 'airpressure', 'relativehumidity', 'mist', 'rain', 
            'snow', 'storm', 'ice']

target = ['flow']

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
n_steps = 5

# SHAPE TRAIN DATA
# split into samples
X, y = split_sequence(y_train_minmax, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# SHAPE TEST DATA
# split into samples
Xt, yt = split_sequence(y_test_minmax, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
Xt = Xt.reshape((Xt.shape[0], Xt.shape[1], n_features))


### SET UP MODEL ###
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(n_steps, n_features)),
#    LeakyReLU(alpha=0.1),
    Activation('relu'),
    LSTM(128),
    Activation('relu'),
    Dense(1),
#    Activation('linear')
])

# summarize model to get an overview
model.summary()

# rmse func
def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

# compile model
model.compile(optimizer = 'adam', loss = 'mse', metrics = [rmse])

## fit model to training features and target(s)
history = model.fit(X, y, epochs=3, verbose=1)





# predict
preds = pd.DataFrame()
# predict and add directly to new df
preds['prediction'] = mm_yscaler.inverse_transform(model.predict(Xt)).ravel()
# reset index of y_test so it can be recombined with predict df, then add it
y_test.reset_index(inplace=True, drop=True)
preds['y_true'] = y_test['flow']
# calculate difference
preds['difference'] = preds.prediction - preds.y_true

RMSE = ((preds.y_true - preds.prediction) ** 2).mean() ** 0.5
print("Test data RMSE:", RMSE)


# "predict" train set
preds_train = pd.DataFrame()
# predict from X_train
preds_train['prediction'] = mm_yscaler.inverse_transform(model.predict(X)).ravel()
# reset index of y_test so it can be recombined with predict df, then add it
y_train.reset_index(inplace=True, drop=True)
preds_train['y_train'] = y_train['flow']
# calculate difference
preds_train['difference'] = preds_train.prediction - preds_train.y_train

RMSE_train = ((preds_train.y_train - preds_train.prediction) ** 2).mean() ** 0.5
print("Train data RMSE:", RMSE_train)


## Plot training & validation loss values
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('Model loss')
#plt.ylabel('Loss')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Validation'], loc='upper right')
#plt.show()


#model.save(datadir + 'model1_4')
#del model
#model = load_model(datadir + 'saved_model.h5')