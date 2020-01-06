### keras playground 1.3 (minmax scaling patch) ###

import os
import pickle
import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, BatchNormalization
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn import preprocessing


# set working dir
os.chdir("C:/Users/peterpiontek/Google Drive/Geodan/DL")

# import homebrew


# directories
datadir = "./data/"

# read data
df = pd.read_pickle(datadir + '3months_weather_no_null.pkl')

# create train and test df
train_data = df[df['datetime'] < '2019-08-01']
test_data = df[df['datetime'] > '2019-07-31']


# define features and target
features = ['weekday', 'hour', 'minute', 'weekend', 'speed', 'windspeed_avg', 
            'windspeed_max','temperature', 'sunduration', 'sunradiation', 
            'precipitationduration', 'precipitation', 'airpressure', 
            'relativehumidity', 'mist', 'rain', 'snow', 'storm', 'ice']

target = ['flow']

# create train and test sets
X_train, y_train, X_test, y_test = train_data[features], train_data[target], \
                                test_data[features],test_data[target]

# rescale (normalize)
mm_scaler = preprocessing.MinMaxScaler()
X_train_minmax = mm_scaler.fit_transform(X_train)
X_test_minmax = mm_scaler.transform(X_test)

mm_scaler2 = preprocessing.MinMaxScaler()
y_train_minmax = mm_scaler2.fit_transform(y_train)
y_test_minmax = mm_scaler2.transform(y_test)

# how to inverse the transformation later
# inverse = mm_scaler.inverse_transform(x)

model = Sequential([
    Dense(128, input_dim=(19), use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(128, input_dim=(128), use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(128, input_dim=(128), use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(128, input_dim=(128), use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(128, input_dim=(128), use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(1)
])

# summarize model to get an overview
model.summary()

#### add this to 1.2 instead and add dropout/regularization ###################################################################
def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

model.compile(optimizer = "rmsprop", loss = rmse, 
              metrics =["accuracy"])


# compile model
model.compile(optimizer = "adam", loss = rmse)

# fit model to training features and target(s)
model.fit(X_train_minmax, y_train_minmax, epochs=20)


model.save(datadir + 'saved_model.h5')
#del model
#model = load_model(datadir + 'saved_model.h5')

# predict
preds = pd.DataFrame()
# predict and add directly to new df
preds['prediction'] = mm_scaler2.inverse_transform(model.predict(X_test_minmax))
# reset index of y_test so it can be recombined with predict df, then add it
y_test.reset_index(inplace=True, drop=True)
preds['y_true'] = y_test['flow']
# calculate difference
preds['difference'] = preds.prediction - preds.y_true

