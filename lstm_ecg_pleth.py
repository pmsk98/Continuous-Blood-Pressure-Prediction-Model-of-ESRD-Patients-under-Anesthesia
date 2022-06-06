# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 16:33:14 2020

@author: user
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout

data1 =pd.read_csv('C:/Users/user/Desktop/capstone_project/final_data/1.csv')

data1=data1[0:10000]

data1=data1.drop(['Time','Unnamed: 0','ECG1','PLETH'],axis=1)

def MinMaxScaler(data):
    numerator=data-np.min(data,0)
    denominator=np.max(data,0)-np.min(data,0)
    return numerator / (denominator +1e-7)


data1=MinMaxScaler(data1)


datay=data1['IBP1']


y=datay.values.tolist()


data_x=[]
data_y=[]


window_size=100


for i in range(len(y)-window_size):

    _y=y[i+window_size]
    
    data_y.append(_y)

#train_dataset

train_size=int(len(data_y)*0.70)

train_x=np.array(data_x[0:train_size])


train_y=np.array(data_y[0:train_size])

#test_dataset
test_size=len(data_y)-train_size

test_x=np.array(data_x[train_size:len(data_x)])
test_y=np.array(data_y[train_size:len(data_y)])


model=Sequential()
model.add(LSTM(units=100,activation='relu',return_sequences=True,input_shape=(window_size,1)))
model.add(Dropout(0.1))
model.add(LSTM(units=100,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=1))
model.summary()

model.compile(optimizer='adam',loss='mean_squared_error')

model.fit(train_x,train_y,epochs=30,batch_size=30)

y_pred=model.predict(test_x)


plt.figure()
plt.plot(test_y,color='red',label='test')
plt.plot(y_pred,color='blue',label='pred')
plt.title('ibp pred')
plt.show
