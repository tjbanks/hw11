#!/home/tbg28/anaconda3/envs/py36/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 22:21:11 2018

@author: Tyler Banks (University of Missouri - Nair Neural Engineering Lab)
"""
import math

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM

from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

batch_size = 500

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
    rem_feat = [0,1,2,3,4,6,7,8,9,10,11]
    features = np.delete(features, (rem_feat), axis=1)
    channels = channels-(len(rem_feat))
    print("Num Channels:", channels)

    features = features.reshape(samples, channels*look_back)
    #Scaling done here
    scX = MinMaxScaler()
    features = scX.fit_transform(features)
    scY = MinMaxScaler()
    labels = scY.fit_transform(labels)
    #
    temp=[]
    np.apply_along_axis(myfunc, axis=1, arr=features )
    features = np.array(temp);
    
    ## TO DO DNN YOU HAVE TO SIMPLIFY
    #have to do some simplification because we're getting a CNN dataset
    (samples, channels, look_back, nil) = features.shape
    if selected_model is dnn_model:
        features = features.reshape(samples, channels*look_back)
    if selected_model is rnn_model:
        features = features.reshape(samples, channels, look_back)
        
    #####
    #plt.plot(features.reshape(samples,channels*look_back)[1])
    #plt.show()
    #plt.plot(labels)
    #plt.show()
    #####
    return features,labels,scX,scY
    
def cnn_model(train_X):
    print("CNN data shape: ", train_X.shape)
    n_filters = 1000
    filter_width = 10
    filter_height = 1
    
    print("Convolutional NN: %d filters (%d x %d)" % (n_filters, filter_height, filter_width))
    model = Sequential()    
    model.add(Conv2D(n_filters, (filter_width, filter_height),
                     padding="same", activation="relu", kernel_initializer="normal",
                     input_shape=(train_X.shape[1],train_X.shape[2], train_X.shape[3])))
    model.add(MaxPooling2D(pool_size=(1,4)))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(Dense(300, kernel_initializer='normal', activation='relu'))
    model.add(Dense(25, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    
    return model
    
def dnn_model(train_X):
    print("DNN data shape: ", train_X.shape)
    model = Sequential()
    model.add(Dense(400, kernel_initializer='normal', activation='relu',input_dim=train_X.shape[1]))
    #model.add(Dense(300, kernel_initializer='normal', activation='relu'))
    model.add(Dense(200, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model
    
def rnn_model(train_X):
    print("RNN data shape: ", train_X.shape)    
    model = Sequential()
    model.add(LSTM(20, batch_input_shape=(batch_size, train_X.shape[1],train_X.shape[2]),stateful = True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    return model
    
def run_model(filename, run_model, threshold):   
    rand_state = 42 #For reproducability        
    do_kfold = False 
    epochs = 40
    
    (features, labels, scX, scY) = load_mat(filename, run_model)
    
    estimator = KerasRegressor(build_fn=lambda: run_model(features), epochs=epochs, batch_size=batch_size,verbose=1)
    kfold = KFold(n_splits=6, shuffle=True, random_state=rand_state)
    if do_kfold:
        results = cross_val_score(estimator, features, labels, scoring='accuracy',cv=kfold)
        results = str.format("Results: mean:%.2f%% std(%.2f%%)" % (results.mean(), results.std()))
    else:
        estimator.fit(features,labels)
        prediction = estimator.predict(features)
        testScore = math.sqrt(mean_squared_error(labels, prediction))
        plt.plot(labels)
        plt.plot(prediction)
        plt.legend(['actual','prediction'],loc='upper left')
        plt.show()
        results = str.format('Test Score: %.2f RMSE' % (testScore))
    return results

def main():
    filename = "train_sub2.mat"
    thresh = .68
    selected_model = cnn_model
    
    results = run_model(filename, selected_model, thresh)
    
    print(results)
    
main()    
