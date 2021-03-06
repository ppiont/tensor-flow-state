### keras playground 1.5 (optimization visualisation and dropout?) ###

import os
import pickle
import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, BatchNormalization, Dropout, LeakyReLU
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

# rescale features
mm_xscaler = preprocessing.MinMaxScaler()
X_train_minmax = mm_xscaler.fit_transform(X_train)
X_test_minmax = mm_xscaler.transform(X_test)
# rescale target
mm_yscaler = preprocessing.MinMaxScaler()
y_train_minmax = mm_yscaler.fit_transform(y_train)

### SET UP MODEL ###
model = Sequential([
    Dense(128, input_dim=(28)),
    LeakyReLU(alpha=0.1),
#    Activation('relu'),
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
history = model.fit(X_train_minmax, y_train_minmax, validation_split=0.2, epochs=20 , verbose=1)


#model.save(datadir + 'model1_4')
#del model
#model = load_model(datadir + 'saved_model.h5')



# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()




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

