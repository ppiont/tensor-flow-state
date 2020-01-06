### keras playground 2.2 CNN-LSTM (univariate) ###
# predict 1 minute in advance based on last 4 minutes (can be changed) <---- CHANGE!!!
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, TimeDistributed, \
                                    Conv1D, MaxPooling1D
from tensorflow.keras import backend as K
from sklearn import preprocessing
import matplotlib.pyplot as plt

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




plt.plot(yhat)
plt.plot(yt)






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



# https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
# # evaluate one or more weekly forecasts against expected values
# def evaluate_forecasts(actual, predicted):
# 	scores = list()
# 	# calculate an RMSE score for each day
# 	for i in range(actual.shape[1]):
# 		# calculate mse
# 		mse = mean_squared_error(actual[:, i], predicted[:, i])
# 		# calculate rmse
# 		rmse = sqrt(mse)
# 		# store
# 		scores.append(rmse)
# 	# calculate overall RMSE
# 	s = 0
# 	for row in range(actual.shape[0]):
# 		for col in range(actual.shape[1]):
# 			s += (actual[row, col] - predicted[row, col])**2
# 	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
# 	return score, scores
