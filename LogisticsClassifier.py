# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 21:44:11 2017

@author: 14094
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import scale
from random import random
from numpy import random as nr

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

def sigmoid(x):
    return 1/(1+np.exp(-x))

def RandSam(train_data, train_target, sample_num):
    data_num = train_data.shape[0]
    if sample_num > data_num:
        return -1
    else:
        data = []
        target = []
        for i in range(sample_num):
            tmp = nr.randint(0,data_num)
            data.append(train_data[tmp])
            target.append(train_target[tmp])
    return np.array(data),np.array(target)
            
class LogisticClassifier(object):
    alpha = 0.01
    circle = 1000
    l2 = 0.01
    weight = np.array([])
    def __init__(self, learning_rate, circle_num, L2):
        self.alpha = learning_rate
        self.circle = circle_num
        self.l2 = L2
    def fit(self, train_data, train_target):
        data_num = train_data.shape[0]
        feature_size = train_data.shape[1]
        ones = np.ones((data_num,1))
        train_data = np.hstack((train_data,ones))
        #Y = train_target
        self.weight = np.round(np.random.normal(0,1,feature_size+1),2)
        for i in range(self.circle):
            delta = np.zeros((feature_size+1,))
            X,Y = RandSam(train_data, train_target, 50)
            for j in range(50):
                delta += (1-sigmoid(Y[j]*np.dot(X[j],self.weight)))* \
                          Y[j]*X[j]
            self.weight += self.alpha*delta-self.l2*self.weight
    
    def predict(self, test_data):
        data_num = test_data.shape[0]
        ones = np.ones((data_num,1))
        X = np.hstack((test_data,ones))
        return sigmoid(np.dot(X,self.weight))
    
    def evaluate(self, predict_target, test_target):
        predict_target[predict_target>=0.5] = 1
        predict_target[predict_target<0.5] = -1
        return sum(predict_target==test_target)/len(predict_target)

if __name__ == "__main__":
    cancer = load_breast_cancer()
    train_data, train_target, test_data, test_target = div(cancer.data, cancer.target,0.5)
    logistics = LogisticClassifier(0.01,2000, 0.01)
    logistics.fit(train_data, train_target)
    predict = logistics.predict(test_data)
    print('the accuracy is ',logistics.evaluate(predict, test_target),'.')

        