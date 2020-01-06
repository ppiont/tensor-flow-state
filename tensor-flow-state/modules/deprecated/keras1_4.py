### keras playground 1.4 (1hot encoded, toying w activation functions patch) ###

import os
import pickle
import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, BatchNormalization
from keras import losses
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn import preprocessing

# set working dir
os.chdir("C:/Users/peterpiontek/Google Drive/Geodan/DL")

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

# rescale (normalize)
mm_xscaler = preprocessing.MinMaxScaler()
X_train_minmax = mm_xscaler.fit_transform(X_train)
X_test_minmax = mm_xscaler.transform(X_test)

mm_yscaler = preprocessing.MinMaxScaler()
y_train_minmax = mm_yscaler.fit_transform(y_train)





model = Sequential([
    Dense(128, input_dim=(28)),
    Activation('relu'),
    Dense(1),
    Activation('linear')
])

# summarize model to get an overview
model.summary()

# rmse func
def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 


# compile model
model.compile(optimizer = 'adam', loss = 'mse', metrics = [rmse])

## fit model to training features and target(s)
model.fit(X_train_minmax, y_train_minmax, epochs=15)

#model.save(datadir + 'model1_4')
#del model
#model = load_model(datadir + 'saved_model.h5')

# predict
preds = pd.DataFrame()
# predict and add directly to new df
preds['prediction'] = mm_yscaler.inverse_transform(model.predict(X_test_minmax)).ravel()
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
preds_train['prediction_train'] = mm_yscaler.inverse_transform(model.predict(X_train_minmax)).ravel()
# reset index of y_test so it can be recombined with predict df, then add it
y_train.reset_index(inplace=True, drop=True)
preds_train['y_train'] = y_train['flow']
# calculate difference
preds_train['difference_train'] = preds_train.prediction_train - preds_train.y_train

RMSE_train = ((preds_train.y_train - preds_train.prediction_train) ** 2).mean() ** 0.5
print("Train data RMSE:", RMSE_train)



#from keras.utils import plot_model
#plot_model(model, show_shapes=True, show_layer_names=True, to_file = plotdir + 'model.png')




## univariate data preparation
#from numpy import array
# 
## split a univariate sequence into samples
#def split_sequence(sequence, n_steps):
#	X, y = list(), list()
#	for i in range(len(sequence)):
#		# find the end of this pattern
#		end_ix = i + n_steps
#		# check if we are beyond the sequence
#		if end_ix > len(sequence)-1:
#			break
#		# gather input and output parts of the pattern
#		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
#		X.append(seq_x)
#		y.append(seq_y)
#	return array(X), array(y)
# 
## define input sequence
#raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
## choose a number of time steps
#n_steps = 3
## split into samples
#X, y = split_sequence(raw_seq, n_steps)
## summarize the data
#for i in range(len(X)):
#	print(X[i], y[i])
#
#
#mlpm = Sequential([
#    Dense(100, input_dim = n_steps, activation = 'relu'),
#    Dense(1)
#    ])
#
#mlpm.compile(optimizer='adam', loss='mse')
#
#mlpm.fit(X, y, epochs = 2000)
#
#
#
#X_input = array([70, 80, 90])
#X_input = X_input.reshape((1,n_steps))
#yhat = mlpm.predict(X_input)
#
#
#
## multivariate data preparation
#from numpy import array
#from numpy import hstack
# 
## split a multivariate sequence into samples
#def split_sequences(sequences, n_steps):
#	X, y = list(), list()
#	for i in range(len(sequences)):
#		# find the end of this pattern
#		end_ix = i + n_steps
#		# check if we are beyond the dataset
#		if end_ix > len(sequences):
#			break
#		# gather input and output parts of the pattern
#		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
#		X.append(seq_x)
#		y.append(seq_y)
#	return array(X), array(y)
# 
## define input sequence
#in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
#in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
#out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
## convert to [rows, columns] structure
#in_seq1 = in_seq1.reshape((len(in_seq1), 1))
#in_seq2 = in_seq2.reshape((len(in_seq2), 1))
#out_seq = out_seq.reshape((len(out_seq), 1))
## horizontally stack columns
#dataset = hstack((in_seq1, in_seq2, out_seq))
## choose a number of time steps
#n_steps = 3
## convert into input/output
#X, y = split_sequences(dataset, n_steps)
#print(X.shape, y.shape)
## summarize the data
#for i in range(len(X)):
#	print(X[i], y[i])


