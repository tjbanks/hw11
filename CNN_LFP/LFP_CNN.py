#!/home/tbg28/anaconda3/envs/py36/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 22:39:28 2018
@author: Tyler Banks
https://github.com/keras-team/keras/issues/401
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
"""
import math
import sys
import random 
import numpy as np
import pandas as pd
import scipy.io as sio

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
    
def load_csv(filename):
    data = pd.read_csv(filename).values
    return data


def save_model(filename_prefix, model):
    # serialize model to JSON
    model_json = model.to_json()
    with open(filename_prefix+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filename_prefix+".h5")
    print("Saved model to disk")
    
def load_model(filename_prefix):
    # load json and create model
    json_file = open(filename_prefix+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(filename_prefix+".h5")
    print("Loaded model from disk")
    return loaded_model

def cnn_model(train_X):
    
    from keras.models import Sequential
    from keras.layers import Conv1D, MaxPooling1D
    from keras.layers import Activation, Dropout, Flatten, Dense
    
    model = Sequential()
    model.add(Conv1D(100, (5), input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=(2)))
    
    #model.add(Conv1D(32, (3)))
    #model.add(Activation('relu'))
    #model.add(MaxPooling1D(pool_size=(2)))
    
    #model.add(Conv1D(64, (3)))
    #model.add(Activation('relu'))
    #model.add(MaxPooling1D(pool_size=(2)))

    model.add(Flatten())  # this converts our 2D feature maps to 1D feature vectors
    model.add(Dense(40))
    model.add(Activation('relu'))
    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='mean_absolute_percentage_error',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

    
def run_model(run_model, features, labels, split_kfold=6,save=True,model_filename_prefix='last_trained_net'):   
    rand_state = 42 #For reproducability        
    if save:
        estimator = KerasRegressor(build_fn=lambda: run_model(features), epochs=4, batch_size=50, verbose=1)
        kfold = KFold(n_splits=split_kfold, shuffle=True, random_state=rand_state)
        results = cross_val_score(estimator, features, labels, cv=kfold)
        #save_model(model_filename_prefix,estimator.model)
    else:
        loaded_model = load_model(model_filename_prefix)
        loaded_model.compile(loss='mean_absolute_percentage_error', optimizer='adam',metrics=['accuracy'])
        results = loaded_model.evaluate(features, labels, verbose=0)
        
    estimator.fit(features,labels)
    prediction = estimator.predict(features)
    plt.plot(labels)
    plt.plot(prediction)
    plt.legend(['actual','prediction'],loc='upper left')
    plt.show()
    save_model(model_filename_prefix,estimator.model)
    
    return results

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i

def main():
    
    filenames = ["trainingdataband1", "trainingdataband2","trainingdataband3","trainingdataband4","trainingdataband5","trainingdataband6"]
    num_files = len(filenames)
    
    selected_model = cnn_model
    lookback_elements = 50
    extra_elements = 2
    split_kfold = 4
    
    n_rows = file_len(filenames[0])
    n_elements = (lookback_elements) + extra_elements
    data = np.zeros([num_files,n_rows,n_elements])
    
    
    for i, f in enumerate(filenames):
        if i == (num_files-1): #keep last training data columns
            extra_elements = 0
        print("loading file {} - {}".format(i+1,f))
        data[i,:,:] = load_csv(f)
    print("All files loaded")    

    data = np.rollaxis(data, 1)
    
    
    features = data[:,:,:-2]
    
    x_min = features.min(axis=(1, 2), keepdims=True)
    x_max = features.max(axis=(1, 2), keepdims=True)

    features = (features - x_min)/(x_max-x_min)
    
    labels = data[:,0,-2:-1]
    
    num_one = (labels==1).sum()
    num_zero = (labels==0).sum()
    print("Number of 0s: {}".format(num_zero))
    print("Number of 1s: {}".format(num_one))

    deletable = np.where(labels==0) #range(features.shape[0]) 
    idx = random.sample(list(deletable[0]), num_zero-num_one)
    features = np.delete(features, idx, 0)
    labels = np.delete(labels, idx, 0)
    
    num_one = (labels==1).sum()
    num_zero = (labels==0).sum()
    print("Deleted random number of zero cases")
    print("Number of 0s: {}".format(num_zero))
    print("Number of 1s: {}".format(num_one))
    
    print("Data shape: {}".format(data.shape))
    print("Features shape: {}".format(features.shape))
    print("Labels shape: {}".format(labels.shape))
    #(features, labels) = load_csv(csv_filename, selected_model, lookback_elements, future_element, stride=stride)
    #(features, labels) = load_mat(filename, selected_model)
    results = run_model(selected_model, features, labels, split_kfold=split_kfold,save=True)
    
    print(str.format("Results: mean:%.2f%% std(%.2f%%)" % (results.mean(), results.std())))
    
main()    