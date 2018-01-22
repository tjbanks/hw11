# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 19:42:37 2017

Goal: Determine if we can detect gamma signals 20ms before they occur

@author: Tyler Banks
"""

#================ IMPORTS ================
import sys, math

from scipy.signal import butter, lfilter, hilbert
import scipy.io as sio
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#================ MISC. ================
#fix random seed for reproducibility
np.random.seed(7)

#================ AUX FUNCTIONS ================
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def load_mat_file(filename, variableName):
    mat = sio.loadmat(filename)
    mdata = mat[variableName]
    return np.array(mdata).ravel()

#================ GLOBAL VARIABLES ================
plot_size_x = 15
plot_size_y = 10

matfile = "LFP_QW_long_tuning7_synr1.mat"
matvar = "LFP_array_all_sum_afterHP"
n_start = 1000
n_end = 4000
n_per_second = 1000

#

#================ COMMAND LINE ARGS ================

#Variables to track: filename, startrecord, endrecord, matfilevariablename

#================ MANIPULATE INPUT ================

#loads a file from start elements to end elements
def load_file_elements(filename, n_start, n_end, matvarname = "", plot=False):
    mat_extension = ".mat" #To generalize later if we have different filetypes
    
    x = [] #elements
    
    if(filename.endswith(mat_extension)):
        if not matvarname:
            sys.exit("MAT File variable not defined... Exiting")
        matx = load_mat_file(matfile, matvarname)
        matfs = matx.size #Number of samples
        if(n_start < 0 or n_end > matfs):
            sys.exit("Start or End out of bounds of file... Exiting.")
        x = matx[n_start:n_end]
        if plot:
            plot_signal(x, n_per_second, signame="Noisy Signal", newPlot=True)
    return x

#Generates a band filter and hilbert transform of the filter and returns both
def generate_band_hilbert_transform(x, lowcut, highcut, filter_order=2, plot=False):
    y = butter_bandpass_filter(x, lowcut, highcut, x.size, order=filter_order)
    analytic_signal = hilbert(y)
    amplitude_envelope = np.abs(analytic_signal)
    
    if plot:
        plot_signal(y, n_per_second, 'Filtered signal (%g - %g Hz)' % (lowcut, highcut), newPlot=True)
        plot_signal(amplitude_envelope, n_per_second, 'Envelope signal (%g - %g Hz)' % (lowcut, highcut), newPlot=False)
        
    return amplitude_envelope,y

#================ PLOTS ================
def plot_signal(x, n_per_second, signame, newPlot=True):
    
    if newPlot:
        plt.figure(figsize=(plot_size_x,plot_size_y))
        plt.clf()
        
    fs = x.size   
    t = np.linspace(0, fs/n_per_second, fs, endpoint=False)
    plt.plot(t, x, label=signame)
    plt.legend()
    plt.xlabel('Time (seconds)')

#================ PREPARE DATA FOR MODEL ================
def to_numpy_scaled_1D(x, plot=False):
    dataset = np.array(x)
    dataset = dataset.astype('float32')
    dataset = dataset.transpose()
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset)
    # frame as supervised learning
    
    if plot:
        plt.figure(figsize=(15,10))
        plt.plot(scaled)
        plt.show()
    
    return scaled, scaler

def to_numpy_scaled(x, plot=False):
    dataset = np.array(x)
    dataset = dataset.astype('float32')
    dataset = dataset.transpose()
    (ndata, nlen) = dataset.shape
    
    # integer encode direction
    #encoder = LabelEncoder()
    #dataset[:,nlen-1] = encoder.fit_transform(dataset[:,nlen-1])
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset)
    # frame as supervised learning
    
    if plot:
        groups = range(0,nlen)
        i=1
        plt.figure(figsize=(15,10))
        for group in groups:
            plt.subplot(len(groups), 1, i)
            plt.plot(scaled[:,group])
            i+=1
        plt.show()
    
    return scaled, scaler

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

    
    # convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
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
 
    
def series_to_supervised_DNNData(data, n_in=1, n_out=1, dropnan=True):
    
    reframed = series_to_supervised(data, n_in, n_out)
    (ndata, nlen) = data.shape
    
    startx = n_in*nlen
    endx = (startx-1) + (nlen*n_out)
    reframed.drop(reframed.columns[range(startx, endx)], axis=1, inplace=True)

    return reframed

def series_to_supervised_CNNData(data, n_in=1, n_out=1, dropnan=True):
    
    reframed = series_to_supervised(data, n_in, n_out)
    (ndata, nlen) = data.shape
    
    startx = n_in*nlen
    endx = (startx-1) + (nlen*n_out)
    reframed.drop(reframed.columns[range(startx, endx)], axis=1, inplace=True)

    return reframed

#================ MODEL DEFINITIONS ================

#initialCNN
def cnn_model(train_X):
    print("CNN Input data shape: ", train_X.shape)
    n_filters = 50
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
    #model.add(Dense(300, kernel_initializer='normal', activation='relu'))
    model.add(Dense(150, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    
    return model

def dnn_model(train_X):
    model = Sequential()
    model.add(Dense(400, kernel_initializer='normal', activation='relu',input_dim=train_X.shape[1]))
    #model.add(Dense(300, kernel_initializer='normal', activation='relu'))
    model.add(Dense(200, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
    return model

def rnn_model(look_back):
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
    
#================ MODEL AND DATA COUPLING ================

class TrainTest:
    trainX = ""
    trainY = ""
    testX = ""
    testY = ""
    
    def __init__(self, trX, trY, teX, teY):
        self.trainX = trX
        self.trainY = trY
        self.testX = teX
        self.testY = teY

def generate_CNN(data, train_size, look_back, future_element):
    binaryClassification = False
    thresh = 0.6

    time_elements, channels = data.shape
    out = series_to_supervised_CNNData(data, n_in=look_back, n_out=future_element, dropnan=True)
    #Data in format [Time][Band], var1(t-100)  var2(t-100)...var1(t-99)...varn(t+19)
    
    out = out.values #pandas to numpy
    
    #See it as a picture for fun
    #from PIL import Image
    #img = Image.fromarray(data.transpose(), 'RGB')
    #img.show()
        
    train = out[:train_size, :]
    test = out[train_size:, :]
    
    # split into input and outputs
    trainX, trainY = train[:, :-1], train[:, -1]
    testX, testY =    test[:, :-1],  test[:, -1]
    #print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    
    # reshape input to be [channels, time steps, features]
    temp = []   
    def myfunc(x):
        temp.append(np.reshape(x, (channels, look_back, 1)))
    np.apply_along_axis(myfunc, axis=1, arr=trainX )
    trainX = np.array(temp);
    
    temp = []
    np.apply_along_axis(myfunc, axis=1, arr=testX )
    testX = np.array(temp);

    if binaryClassification:
        trainY[trainY >= thresh] = 1
        trainY[trainY < thresh] = 0
        testY[testY >= thresh] = 1
        testY[testY < thresh] = 0

    model = cnn_model(trainX)

    tt = TrainTest(trainX, trainY, testX, testY)
    return tt, model

def generate_DNN(data, train_size, look_back, future_element):
    out = series_to_supervised_DNNData(data, n_in=look_back, n_out=future_element, dropnan=True)
    #Data in format [Time][Band], var1(t-100)  var2(t-100)...var1(t-99)...varn(t+19)
    
    out = out.values # pandas to numpy
    
    train = out[:train_size, :]
    test = out[train_size:, :]
    
    # split into input and outputs
    trainX, trainY = train[:, :-1], train[:, -1]
    testX, testY =    test[:, :-1],  test[:, -1]
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    
    # reshape input to be [samples, time steps, features]
    #trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    #testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    
    model = dnn_model(trainX)
    tt = TrainTest(trainX, trainY, testX, testY)
    return tt, model

def generate_RNN(data, train_size, look_back):
    #outx, outy = create_dataset(data, look_back)
    model = rnn_model(look_back)
    
    train, test = data[0:train_size,:], data[train_size:len(data),:]
    # reshape into X=t and Y=t+1
    
    trainX, trainY = create_dataset(train, look_back)
    return tt, model

def generate_RNN(data, train_size, look_back):
    #outx, outy = create_dataset(data, look_back)
    model = rnn_model(look_back)
    
    train, test = data[0:train_size,:], data[train_size:len(data),:]
    # reshape into X=t and Y=t+1
    
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    
    tt = TrainTest(trainX, trainY, testX, testY)
    return tt, model
#================ MAIN PROGRAM ================

def main():
    #load file
    x = load_file_elements(matfile, n_start, n_end, matvarname=matvar, plot=False)
    #x, scaler = to_numpy_scaled_1D(x, plot=True)#Testing
    #plot_signal(x, n_per_second, signame="Noisy Signal", newPlot=True)
    
    bands = [[2,10],[11,20],[21,30],[31,40],[41,50],[51,63],[64,84]] #in Hz, lower and upper bounds to be used as features, last is y
    #bands = [[64,84]] #in Hz, lower and upper bounds to be used as features

    bandSets = []
    
    #generate the bands we want to use as our input
    for band in bands:
        hilbert_t, b = generate_band_hilbert_transform(x, band[0], band[1],plot=False)
        bandSets.append(hilbert_t)
    #Data in format [Band][Time]
    
    #To Numpy Array for processing
    dataset, scaler = to_numpy_scaled(bandSets, plot=False)    

    train_size = int(len(dataset) * 0.67)
    look_back = 300
    future_element = 20
    epochs = 5
    
    tt, model = generate_CNN(dataset, train_size, look_back, future_element)
    #tt, model = generate_DNN(dataset, train_size, look_back, future_element)
    #tt, model = generate_RNN(dataset, train_size, look_back) #Simply predicts next element from past 20 elements, incapable of multichannel input!

    model_info = "Past elements considered: %d | Future element predicted: %d | Epochs: %d" % (look_back,future_element, epochs)
        
    print(model_info)
    print("Bands: ", bands)
    
    print("tt.trainX.shape:", tt.trainX.shape)
    print("tt.trainY.shape:", tt.trainY.shape)
    print("tt.testX.shape:", tt.testX.shape)
    print("tt.testY.shape:", tt.testY.shape)
    model.fit(tt.trainX, tt.trainY, epochs=epochs, batch_size=1, verbose=2)
    
    
    # make predictions
    trainPredict = model.predict(tt.trainX)
    testPredict = model.predict(tt.testX)
    
    print("trainPredict: ")
    print(np.array(trainPredict).flatten())
    print("trainActual: ")
    print(tt.trainY)
    
    print("testPredict: ")
    print(np.array(testPredict).flatten())
    print("testActual: ")
    print(tt.testY)
    
    # invert predictions
#    trainPredict = scaler.inverse_transform(trainPredict)
#    trainY = scaler.inverse_transform([tt.trainY])
#    testPredict = scaler.inverse_transform(testPredict)
#    testY = scaler.inverse_transform([tt.testY])
#    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(tt.trainY, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(tt.testY, testPredict))
    print('Test Score: %.2f RMSE' % (testScore))
#    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(x)
    trainPredictPlot[:] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back] = np.array(trainPredict).flatten()
#    plt.plot(testPredictPlot, label="Predicted Test Signal")
#    plt.legend()
#    plt.show()
    
    return


def main_args(filename, matvariable): #Instead of running this from cmd line we could call this from another program 
    matfile = filename
    matvar = matvariable
    main()
    return

#================ RUN PROGRAM ================
if __name__ == "__main__":
    main()
