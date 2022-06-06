# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 02:14:26 2020

@author: user
"""

#!/usr/bin/env python
# coding: utf-8

# In[141]:


import pandas as pd



import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph()
tf.reset_default_graph()
tf.set_random_seed(777) # reproducibility


# In[143]:


def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

# train Parameters
seq_length = 1000
data_dim = 3
hidden_dim = 10
output_dim = 100
learning_rate = 0.01
iterations = 500

# Open, High, Low, Volume, Close
xy = np.loadtxt('C:/Users/user/Documents/카카오톡 받은 파일/IBP_2.csv', delimiter=',',skiprows=1)


# In[144]:


xy = MinMaxScaler(xy)
x = xy[:10000,]
y = xy[:10000, [0]]  # 마지막 열이 주식 종가 




# In[147]:


# build a dataset
dataX = []
dataY = []
for i in range(0, len(y) - seq_length - 99):
    _x = x[i:i + seq_length]
    _y = np.array(y[(i + seq_length):(i + seq_length + 100)].flatten()) # Next close price
    dataX.append(_x)
    dataY.append(_y)
    
# train/test split
train_size = int(len(dataY) * 0.6)
test_size = len(dataY) - train_size

trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])


# In[148]:


testY.shape


# In[149]:


trainY.shape


# In[150]:


testX.shape


# In[151]:


trainX.shape


# In[152]:


# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 100])  # output_dim 수 = 2

# build a LSTM network
# cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim, activation=tf.tanh)
# cell = tf.contrib.rnn.GRUCell(num_units=hidden_dim, activation=tf.tanh)
cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_dim, activation=tf.tanh)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)


# In[153]:


from tf_slim.layers import layers as _layers;
# many-to-one 모델 
Y_pred = _layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None) 

# In[154]:


# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y)) # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 100])
predictions = tf.placeholder(tf.float32, [None, 100])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
mape = tf.reduce_mean(tf.abs(tf.divide(tf.subtract(predictions,targets),targets)))


# In[155]:


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

        print("RMSE: {}".format(rmse_val))
        print("MAPE: {}".format(mape_val))

        # Plot predictions
    plt.plot(testY)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()


# In[156]:





# In[157]:


test_predict


# In[158]:


testX.shape


# In[170]:


plt.plot(testY[:,0])
plt.plot(test_predict[:,0])
plt.xlabel("Time Period")
plt.ylabel("Stock Price")
plt.show()


# In[172]:


plt.plot(test_predict)
plt.show()


# In[174]:


plt.plot(y[0:5000,])
plt.show()

plt.plot(y[0:500,])
plt.show()
# In[ ]:




