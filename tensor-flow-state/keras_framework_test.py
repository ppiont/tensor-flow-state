import sys
import os
import tensorflow as tf
import pandas as pd
import numpy as np


# set working dir
os.chdir("C:/Users/peterpiontek/Google Drive/tensor-flow-state/tensor-flow-state")

# import homebrew


# directories
datadir = "./data/"
plotdir = "./plots/"

## read data
#df = pd.read_pickle(datadir + '3months_weather.pkl')
#
## create train and test df
#train_data = df[df['datetime'] < '2019-08-01']
#test_data = df[df['datetime'] > '2019-07-31']
#
## define features and target
#features = ['minute_sine', 'minute_cosine', 'hour_sine', 'hour_cosine', 'monday', 
#            'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
#            'weekend', 'holiday', 'speed', 'windspeed_avg', 'windspeed_max', 
#            'temperature', 'sunduration', 'sunradiation', 'precipitationduration',
#            'precipitation', 'airpressure', 'relativehumidity', 'mist', 'rain', 
#            'snow', 'storm', 'ice']
#target =    ['flow']
#
## create train and test sets
#X_train, y_train, X_test, y_test = train_data[features], train_data[target], \
#                                test_data[features],test_data[target]                                
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
#	return np.array(X), np.array(y)
#
## choose a number of time steps
#n_steps = 5
#
## SHAPE TRAIN DATA
## split into samples
#X, y = split_sequence(np.array(y_train), n_steps)
#
## reshape from [samples, timesteps] into [samples, timesteps, features]
#n_features = 1
#X = X.reshape((X.shape[0], X.shape[1], n_features))
#
## SHAPE TEST DATA
## split into samples
#Xt, yt = split_sequence(np.array(y_test), n_steps)
#
## reshape from [samples, timesteps] into [samples, timesteps, features]
#n_features = 1
#Xt = Xt.reshape((Xt.shape[0], Xt.shape[1], n_features))

model_specs = {}
model_specs['Layer type'] = 'LSTM'
model_specs['Hidden layers'] = [100] * 6
model_specs['Input size'] = Xt.shape[1:]                                        # bugger shape
model_specs['Activations'] = ['relu'] * 6
model_specs['Optimization'] = 'adadelta'
model_specs["Regularization"] = 'Dropout'
model_specs['Reg param'] = False
model_specs["Learning rate"] = .005
model_specs["Batch size"] = 32
model_specs["Preprocessing"] = 'Standard'
model_specs["Lambda"] = 0
model_specs['Metrics'] = ['mae'] # must be a list object
model_specs['Name'] = 'Test_model'


def build_nn(model_specs):
    """
    Build and compile a NN given a hash table of the model's parameters.
    :param model_specs:
    :return:
    """
    if model_specs['Layer type'] == 'Dense':
        layer = tf.keras.layer.Dense
#    elif model_specs['Layer type'] == 'LSTM':
#        layer = tf.keras.layers.LSTM                                           # problem with return_sequence defaulting as False
    else:
        sys.exit("Only layers of type 'Dense' are currently supported.")
        
        
    try:
        if model_specs["Regularization"] == "l2":                               # if using L2 regularization
            lambda_ = model_specs['Reg param']                                  # get lambda parameter
            batch_norm, keep_prob = False, False                                # set other regularization tactics

        elif model_specs['Regularization'] == 'Batch norm':                     # batch normalization
            lambda_ = 0
            batch_norm = model_specs['Reg param']                               # get param
            keep_prob = False
            if batch_norm not in ['before', 'after']:                           # ensure reg param is valid
                raise ValueError

        elif model_specs['Regularization'] == 'Dropout':                        # dropout regularization
            lambda_, batch_norm = 0, False
            keep_prob = model_specs['Reg param']
    except:
        lambda_, batch_norm, keep_prob = 0, False, False                        # if no regularization is being used

    hidden, acts = model_specs['Hidden layers'], model_specs['Activations']
    model = tf.keras.models.Sequential(name=model_specs['Name'])
#    model.add(tf.keras.layers.InputLayer((model_specs['Input size'],)))        # create input layer
    first_hidden = True

    for lay, act, i in zip(hidden, acts, range(len(hidden))):                   # create all the hidden layers
        if lambda_ > 0:                                                         # if doing L2 regularization
            if not first_hidden:
                model.add(layer(lay, activation=act, 
                        W_regularizer=tf.keras.regularizers.l2(lambda_), 
                        input_shape=(hidden[i - 1],)))                          # add additional layers
            else:
                model.add(layer(lay, activation=act, 
                        W_regularizer=tf.keras.regularizers.l2(lambda_), 
                        input_shape=model_specs['Input size']))
                first_hidden = False
        else:                                                                   # if not regularizing
            if not first_hidden:
                model.add(layer(lay, 
                                    input_shape=(hidden[i-1], )))               # add un-regularized layers
            else:
                model.add(layer(lay, 
                                    input_shape=model_specs['Input size']))     # if its first layer, connect it to the input layer
                first_hidden = False

        if batch_norm == 'before':
            model.add(tf.keras.layers.BatchNormalization(input_shape=(lay,)))   # add batch normalization layer

        model.add(tf.keras.layers.Activation(act))                              # activation layer is part of the hidden layer

        if batch_norm == 'after':
            model.add(tf.keras.layers.BatchNormalization(input_shape=(lay,)))   # add batch normalization layer

        if keep_prob:
            model.add(tf.keras.layers.Dropout(keep_prob, input_shape=(lay,)))   # dropout layer

    # --------- Adding Output Layer -------------
    model.add(tf.keras.layers.Dense(1, input_shape=(hidden[-1], )))             # add output layer
    if batch_norm == 'before':                                                  # if using batch normalization
        model.add(tf.keras.layers.BatchNormalization(input_shape=(hidden[-1],
                                                                  )))
    model.add(tf.keras.layers.Activation('sigmoid'))                            # apply output layer activation
    if batch_norm == 'after':
        model.add(tf.keras.layers.BatchNormalization(input_shape=(hidden[-1],
                                                                  )))           # add batch norm layer

    if model_specs['Optimization'] == 'adagrad':                                # set an optimization method
        opt = tf.keras.optimizers.Adagrad(lr = model_specs["Learning rate"])
    elif model_specs['Optimization'] == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(lr = model_specs["Learning rate"])
    elif model_specs['Optimization'] == 'adam':
        opt = tf.keras.optimizers.Adam(lr = model_specs["Learning rate"])
    elif model_specs['Optimization'] == 'adadelta':
        opt = tf.keras.optimizers.Adadelta()
    elif model_specs['Optimization'] == 'adamax':
        opt = tf.keras.optimizers.Adamax(lr = model_specs["Learning rate"])
    else:
        opt = tf.keras.optimizers.Nadam(lr = model_specs["Learning rate"])
        
    model.compile(optimizer=opt, loss='mse', 
                  metrics=model_specs['Metrics'])                               # compile model

    return model


model = build_nn(model_specs)

model.summary()








#params = {'hidden_layers_type: {
#    }
#        'n_hidden_layers': 5,
#          }
#input_layer
#hidden_layers = [tf.keras.layers.Dense['n_hidden_layers']]
#output_layer