#!/home/tbg28/anaconda3/envs/py36/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 22:39:28 2018

@author: Tyler Banks
https://github.com/keras-team/keras/issues/401
"""

import numpy as np
import pandas as pd
import scipy.io as sio
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation

from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

def load_mat(filename, selected_model):
    features_name = "x"
    labels_name = "y"
    
    mat = sio.loadmat(filename)
    features = mat[features_name]
    labels = mat[labels_name]
    
    # data currently in the shape:
    # [50 preceding][6 Hilbert amplitude|6 Filtered signals][72000 samples]
    # reshape input to be [channels, time steps, features]
    # or CNN Input data shape:  (2010, 7, 300, 1) 
    features = np.rollaxis(features, -2)
    features = np.rollaxis(features, 2)
    (samples, channels, look_back) = features.shape
    features = features.reshape(samples, channels*look_back)
    temp = []   
    def myfunc(x):
        temp.append(np.reshape(x, (channels, look_back, 1)))
    np.apply_along_axis(myfunc, axis=1, arr=features )
    features = np.array(temp);
    #Now shaped to be (72000, 12, 50, 1)
    
    #Now we remove the bands we don't want
    features = np.delete(features, ([0,1,2,3,5,6,7,8,9,10,11]), axis=1)
    
    ## TO DO DNN YOU HAVE TO SIMPLIFY
    #have to do some simplification because we're getting a CNN dataset
    (samples, channels, look_back, nil) = features.shape
    if selected_model is dnn_model:
        features = features.reshape(samples, channels*look_back)
    if selected_model is rnn_model:
        features = features.reshape(samples, channels, look_back)
        
    #####
    
    return features,labels
    
def load_csv(filename, selected_model, lookback_elements, future_element,stride=1):
    f = np.genfromtxt(filename,delimiter='\n')
    f = f.reshape([f.shape[0],1])
    data = series_to_supervised_Data(f, n_in=lookback_elements, n_out=future_element)
    data = pd.DataFrame(data.values[::stride,:])
    features = np.array(data.iloc[:,:-1].values)
    labels = np.array(data.iloc[:,-1].values)
    
    temp = []   
    def myfunc(x):
        temp.append(np.reshape(x, (1, lookback_elements, 1)))
    np.apply_along_axis(myfunc, axis=1, arr=features )
    features = np.array(temp);
    #Now shaped to be (72000, 1, 50, 1)
    
    (samples, channels, look_back, nil) = features.shape
    if selected_model is dnn_model:
        features = features.reshape(samples, channels*look_back)
    if selected_model is rnn_model:
        features = features.reshape(samples, channels, look_back)
    
    return features,labels
    

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
        
    return agg

def series_to_supervised_Data(data, n_in=1, n_out=1, dropnan=True):
    
    reframed = series_to_supervised(data, n_in, n_out)
    (ndata, nlen) = data.shape
    
    startx = n_in*nlen
    endx = (startx-1) + (nlen*n_out)
    reframed.drop(reframed.columns[range(startx, endx)], axis=1, inplace=True)

    return reframed


def cnn_model(train_X):
    print("CNN data shape: ", train_X.shape)
    n_filters = 500
    filter_width = 20
    filter_height = 1
    
    print("Convolutional NN: %d filters (%d x %d)" % (n_filters, filter_height, filter_width))
    model = Sequential()    
    model.add(Conv2D(n_filters, (filter_width, filter_height),
                     padding="same", kernel_initializer="normal",
                     input_shape=(train_X.shape[1],train_X.shape[2], train_X.shape[3])))
    model.add(MaxPooling2D(pool_size=(1,4)))
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(25, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['accuracy'])
    
    return model
    
def dnn_model(train_X):
    print("DNN data shape: ", train_X.shape)
    model = Sequential()
    model.add(Dense(400, kernel_initializer='normal', activation='relu',input_dim=train_X.shape[1]))
    #model.add(Dense(300, kernel_initializer='normal', activation='relu'))
    model.add(Dense(200, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['accuracy'])
    return model
    
def rnn_model(train_X):
    print("RNN data shape: ", train_X.shape)
    batch_size = 5
    model = Sequential()
    model.add(LSTM(20, batch_input_shape=(batch_size, train_X.shape[1],train_X.shape[2]),stateful = True))
    model.add(Dense(1))
    model.compile(loss='mean_absolute_percentage_error', optimizer='adam',metrics=['accuracy'])
    return model
    
def run_model(run_model, features, labels, threshold,split_kfold=6):   
    rand_state = 42 #For reproducability        
    estimator = KerasRegressor(build_fn=lambda: run_model(features), epochs=1, batch_size=5, verbose=1)
    kfold = KFold(n_splits=split_kfold, shuffle=True, random_state=rand_state)
    results = cross_val_score(estimator, features, labels, cv=kfold)
    return results

def main():
    filename = "train_sub2.mat"
    csv_filename = "raw_subject2.csv"
    thresh = 6.6889
    selected_model = cnn_model
    lookback_elements = 50
    future_element = 10
    stride = 10 #skip every 10 frames
    split_kfold = 4
    
    (features, labels) = load_csv(csv_filename, selected_model, lookback_elements, future_element, stride=stride)
    
    results = run_model(selected_model, features, labels, thresh,split_kfold=split_kfold)
    
    print(str.format("Results: mean:%.2f%% std(%.2f%%)" % (results.mean(), results.std())))
    
main()    