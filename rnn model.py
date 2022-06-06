# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 15:56:45 2020

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 15:44:08 2020

@author: user
"""

import pandas as pd

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

from tf_slim.layers import layers as _layers;
# many-to-one 모델 


os.chdir('C:\\Users\\user\\Desktop\\capstone_project\\final_data')



# 정규화 
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

# 학습 파라미터 설정 
seq_length = 500
data_dim = 3
hidden_dim = 32
output_dim = 10
learning_rate = 0.001
iterations = 700

import glob
filenames = glob.glob('C:\\Users\\user\\Desktop\\capstone_project\\final_data'+"/*.csv")
filenames[:5]


MAPE_total = []
MAPE_dict = {}

for data in filenames:
    
    tf.reset_default_graph()

    xy = pd.read_csv(data, encoding = 'cp949')
    xy = xy.iloc[0:7509,2:5]

    xy = MinMaxScaler(xy)
    x = np.array(xy)
    y = np.array(xy.iloc[:, 0])  # 종속 변수 

    # build a dataset
    dataX = []
    dataY = []
    for i in range(0, len(y) - seq_length - 9):
        _x = x[i:i + seq_length]
        _y = np.array(y[(i + seq_length):(i + seq_length + 10)].flatten()) # Next close price
        dataX.append(_x)
        dataY.append(_y)
    
    # train/test split
    train_size = 6000
    test_size = len(dataY) - train_size

    trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
    trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])

    # input place holders
    X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
    Y = tf.placeholder(tf.float32, [None, 10])  # output_dim 수 = 2

    # build a LSTM network
    # cell = tf.contrib.rnn.GRUCell(num_units=hidden_dim, activation=tf.tanh)
    cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.tanh)

    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    # many-to-one 모델이기 때문에 -1
    Y_pred = _layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None) 
    # cost/loss
    loss = tf.reduce_sum(tf.square(Y_pred - Y)) # sum of the squares
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    # RMSE
    targets = tf.placeholder(tf.float32, [None, 10])
    predictions = tf.placeholder(tf.float32, [None, 10])
    rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
    mape = tf.reduce_mean(tf.abs(tf.divide(tf.subtract(predictions,targets),targets)))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Training step
        for i in range(iterations):
            _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
            print("[step: {}] loss: {}".format(i, step_loss))


            # Test step
            test_predict = sess.run(Y_pred, feed_dict={X: testX})
            rmse_val = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
            mape_val = sess.run(mape, feed_dict={targets: testY, predictions: test_predict})
        
        print(data)

        plt.plot(testY[:,0])
        plt.plot(test_predict[:,0])
        plt.xlabel("Time Period")
        plt.ylabel("IBP")
        plt.title(data.split('\\')[-1])
        plt.show()
        
    
    MAPE_total.append(mape)
    MAPE_dict[data] = mape


