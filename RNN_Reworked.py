# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:11:24 2020

@author: bbelek
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
daily_data = pd.DataFrame(pd.read_csv("daily_data_seasonality.csv",header=0, index_col=0, parse_dates=True, squeeze=True))
    
array_data = np.array(daily_data)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(array_data)

timestep = 14
def shapedArray(timestep,feature_count,data):
    bombos = np.zeros((len(data)-timestep,timestep,feature_count))
    for row in range(len(data)-(timestep)):
        for i in range(timestep):
            bombos[row][i] = data[row+i]
    return bombos

data_X = shapedArray(timestep,3,scaled_data)
data_Y = scaled_data[14:,0]

train_X = data_X[:365*3+1]
train_Y = data_Y[:365*3+1]
test_X = data_X[365*3+1:]
test_Y = data_Y[365*3+1:]



import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN, LSTM

opt = keras.optimizers.Adam(learning_rate=0.0001)
model_14 = Sequential()
model_14.add(SimpleRNN(50, input_shape=(timestep,3)))
model_14.add(Dense(1))
model_14.compile(loss = 'mean_squared_error',optimizer= opt)

model_14.fit(train_X,train_Y,epochs=500)
model_14.summary()

predictions = model_14.predict(test_X)

errors = test_Y - predictions

perc_errors = (errors/test_Y)
print("Mean Absolute Percentage Error ", str(abs(perc_errors).mean()))






