# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 22:15:14 2017

@author: 14094
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import scale
from random import random

def div(data, target, rate):
    data = scale(data)
    target[target==0] = -1
    train_data = np.array([])
    train_target = []
    test_data = np.array([])
    test_target = []
    for i in range(data.shape[0]):
        if random() < rate:
            if train_data.size == 0:
                train_data = data[i]
                train_target.append(target[i])
            else:
                train_data = np.vstack((train_data,data[i]))
                train_target.append(target[i])
        else:
            if test_data.size == 0:
                test_data = data[i]
                test_target.append(target[i])
            else:
                test_data = np.vstack((test_data,data[i]))
                test_target.append(target[i])
    return train_data, train_target, test_data, test_target

def sign(x):
    if x > 0:
        return 1
    else:
        return -1

class SVM(object):
    #train_data = np.array([])
    #train_target = np.array([])
    data_num = 0
    feature_size = 0
    weight = np.array([])
    bais = 0
    rate = 0.01
    learning_circle = 1000
    def __init__(self, alpha, circle):
        self.rate = alpha
        self.learning_circle = circle
    def fit(self, X, Y):
        #self.train_data = X
        #self.train_target = Y
        self.data_num = X.shape[0]
        self.feature_size = X.shape[1]
        self.weight = np.round(np.random.normal(0,1,self.feature_size),2)
        self.bais = np.random.normal()
        for i in range(self.learning_circle):
            delta_w = np.zeros((self.feature_size,))
            delta_b = 0
            for j in range(X.shape[0]):
                interval = Y[j]*(np.dot(self.weight,X[j]) + self.bais)
                if interval < 1:
                    delta_w += -Y[j]*X[j]
                    delta_b += Y[j]
            self.weight -= self.rate*(delta_w + self.weight)
            self.bais += self.rate*delta_b
    def predict(self, test_data):
        predict_target = []
        for i in range(test_data.shape[0]):
            y = np.dot(self.weight,test_data[i]) + self.bais
            predict_target.append(sign(y))
        return predict_target
    
    def evaluate(self, predict_target, test_target):
        err = 0
        for i in range(len(predict_target)):
            if predict_target[i] != test_target[i]:
                err += 1
        return 1-err/len(predict_target)

if __name__ == "__main__":
    cancer = load_breast_cancer()
    train_data, train_target, test_data, test_target = div(cancer.data, cancer.target,0.5)
    svm = SVM(0.01, 1000)
    svm.fit(train_data, train_target)
    predict = svm.predict(test_data)
    print(svm.evaluate(predict, test_target))