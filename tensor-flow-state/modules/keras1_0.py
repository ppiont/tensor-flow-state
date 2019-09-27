### keras playground 1.0 ###

import os
import pickle
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt

# set working dir
os.chdir("C:/Users/peterpiontek/Google Drive/Geodan")

# import homebrew


# directories
dataDir = "./data/"

# read data
df = pd.read_pickle(dataDir + '3months_weather_no_null.pkl')

# create train and test df
train_data = df[df['datetime'] < '2019-08-01']
test_data = df[df['datetime'] > '2019-07-31']

# define features and target
features = ['weekday', 'hour', 'radiation', 'temperature', 'rain', 'speed']
target = ['flow']

# create train and test sets
X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target].df.reset_index(drop=True)

# instantiate sequential model
model = Sequential()

# add input and dense layer -- can use input_shape(r,c) or input_dim()
model.add(Dense(50, input_shape=(6,), activation='relu'))
model.add(Dense(50, input_shape=(50,), activation='relu'))
model.add(Dense(50, input_shape=(50,), activation='relu'))

# add final output neuron layer
model.add(Dense(1))


### TRY BATCH NORMALIZATION ###
#model.add(layers.Dense(64, use_bias=False))
#model.add(layers.BatchNormalization())
#model.add(Activation("relu"))



# the following achieves the same as above
#model_fast = Sequential([
#    Dense(3, input_dim=(5)),
#    Activation('relu'),
#    Dense(1),
#    #Activation('softmax'),
#])

# summarize model to get an overview
model.summary()

# compile model
model.compile(optimizer = "adam", loss = "mse")

# fit model to training features and target(s)
model.fit(X_train, y_train, epochs=30)

# predict
preds = pd.DataFrame()
preds['prediction'] = model.predict(X_test).ravel()
preds['difference'] = preds.prediction - y_test.flow
#test_data['preds'] = model.predict(X_test)

# evaluate results
model.evaluate(X_test, y_test)

testing = model.predict(X_test)


# plot pred vs real
plt.figure(figsize=(15, 10))
plt.plot(test_data['datetime'], test_data['flow'])
plt.plot(test_data['datetime'], preds[:, 0])
plt.xlabel('Time')
plt.ylabel('Flow')
plt.legend(['Actual flow', 'Predicted flow'])

plt.figure(figsize=(15, 10))
plt.plot(test_data['datetime'], preds['difference'])
plt.xlabel('Time')
plt.ylabel('Difference Pred - Actual')
plt.legend(['Actual flow', 'Predicted flow'])
