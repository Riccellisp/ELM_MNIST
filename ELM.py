#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 00:12:43 2020

@author: riccelli
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def input_to_hidden(x):
    a = np.dot(x, Win)
    a = np.maximum(a, 0, a) # ReLU
    return a

def predict(x):
    x = input_to_hidden(x)
    y = np.dot(x, Wout)
    return y

train = pd.read_csv("/home/riccelli/ELM_MNIST/MNIST_DATASET/train.csv")
train.head()

x_train = train.iloc[:, 1:].values.astype('float32')
labels = train.iloc[:, 0].values.astype('int32')

fig = plt.figure(figsize=(12, 12))
for i in range(5):
    fig.add_subplot(1, 5, i+1)
    plt.title('Label: {label}'.format(label=labels[i]))
    plt.imshow(x_train[i].reshape(28, 28), cmap='Greys')

######################## ONE HOT ENCODING FROM SCRATCH ########################    
CLASSES = 10
y_train = np.zeros([labels.shape[0], CLASSES])

for i in range(labels.shape[0]):
    y_train[i][labels[i]] = 1
y_train.view(type=np.matrix)
######################## ONE HOT ENCODING FROM SCRATCH ########################

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)
print('Train size: {train}, Test size: {test}'.format(train=x_train.shape[0], test=x_test.shape[0]))

# The ELM algorithm is similar to other neural networks with 3 key differences:

# The number of hidden units is usually larger than in other neural networks that are trained using backpropagation.

# The weights from input to hidden layer are randomly generated, usually using values from a continuous uniform distribution.

# The output neurons are linear rather than sigmoidal, this means we can use least square errors regression to solve the output weights.

INPUT_LENGHT = x_train.shape[1] # 784 
HIDDEN_UNITS = 3000

Win = np.random.normal(size=[INPUT_LENGHT, HIDDEN_UNITS])
print('Input Weight shape: {shape}'.format(shape=Win.shape))

X = input_to_hidden(x_train)
Xt = np.transpose(X)
Wout = np.dot(np.linalg.inv(np.dot(Xt, X)), np.dot(Xt, y_train))
print('Output weights shape: {shape}'.format(shape=Wout.shape))

y = predict(x_test)
correct = 0
total = y.shape[0]
for i in range(total):
    predicted = np.argmax(y[i])
    test = np.argmax(y_test[i])
    correct = correct + (1 if predicted == test else 0)
print('Accuracy: {:f}'.format(correct/total))
