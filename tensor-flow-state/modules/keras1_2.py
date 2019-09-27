### keras playground 1.2 (dropout patch) ###

import os
import pickle
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
from keras import backend as K
import matplotlib.pyplot as plt


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
features = ['weekday', 'hour', 'radiation', 'temperature', 'rain', 'speed']
target = ['flow']

# create train and test sets
X_train, y_train, X_test, y_test = train_data[features], train_data[target], test_data[features],test_data[target]


model = Sequential([
    Dense(64, input_dim=(6), use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(64, input_dim=(6), use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(64, input_dim=(6), use_bias=False),
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
model.fit(X_train, y_train, epochs=20)

# predict
preds = pd.DataFrame()
preds['prediction'] = model.predict(X_test).ravel()
preds['difference'] = preds.prediction - y_test.flow