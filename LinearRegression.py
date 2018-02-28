# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:16:06 2017

@author: 14094
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import scale
from random import random

def div(data, target, rate):
    data = scale(data)
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

class LinearRegression(object):
    weight = np.array([])
    way = 'gd'
    def __init__(self, training_way = 'gd'):
        self.way = training_way
    def gradientDescent(self, X, Y, alpha, epoch):
        W = np.random.normal(0,1,size=(X.shape[1],))
        for i in range(epoch):
            W -= alpha*(X.T).dot(X.dot(W)-Y)/X.shape[0]
        return W
        
    def fit(self, train_data, train_target, alpha = 0.1, epoch = 300):
        X = np.ones((train_data.shape[0], train_data.shape[1]+1))
        X[:,0:-1] = train_data
        Y = train_target
        if self.way == 'gd':
            self.weight = self.gradientDescent(X, Y, alpha, epoch)
        else:
            self.weight = np.linalg.inv((X.T).dot(X)).dot(X.T).dot(Y)
    
    def predict(self, test_data):
        X = np.ones((test_data.shape[0], test_data.shape[1]+1))
        X[:,0:-1] = test_data
        return X.dot(self.weight)
    
    def evaluate(self, predict_target, test_target):
        predict_target[predict_target>=0.5] = 1
        predict_target[predict_target<0.5] = 0
        return sum(predict_target==test_target)/len(predict_target)

class LassoRegression(object):
    weight = np.array([])
    def __init__(self):
        return 
    def dif(self, W, c):
        w = np.array(W)
        w[W>0] = 1
        w[W<0] = -1
        return c*w
    
    def gradientDescent(self, X, Y, alpha, epoch, c):
        W = np.random.normal(0,1,size=(X.shape[1],))
        for i in range(epoch):
            W -= alpha*(X.T).dot(X.dot(W)-Y)/X.shape[0] + self.dif(W, c)
        return W
        
    def fit(self, train_data, train_target, alpha = 0.1, epoch = 300, c = 0.05):
        X = np.ones((train_data.shape[0], train_data.shape[1]+1))
        X[:,0:-1] = train_data
        Y = train_target
        self.weight = self.gradientDescent(X, Y, alpha, epoch, c)
    
    def predict(self, test_data):
        X = np.ones((test_data.shape[0], test_data.shape[1]+1))
        X[:,0:-1] = test_data
        return X.dot(self.weight)
    
    def evaluate(self, predict_target, test_target):
        predict_target[predict_target>=0.5] = 1
        predict_target[predict_target<0.5] = 0
        return sum(predict_target==test_target)/len(predict_target)

class RidgeRegression(object):
    weight = np.array([])
    way = 'gd'
    def __init__(self, training_way = 'gd'):
        self.way = training_way
    def gradientDescent(self, X, Y, alpha, epoch, c):
        W = np.random.normal(0,1,size=(X.shape[1],))
        for i in range(epoch):
            W -= alpha*(X.T).dot(X.dot(W)-Y)/X.shape[0] + c*W
        return W
        
    def fit(self, train_data, train_target, alpha = 0.1, epoch = 300, c = 0.05):
        X = np.ones((train_data.shape[0], train_data.shape[1]+1))
        X[:,0:-1] = train_data
        Y = train_target
        if self.way == 'gd':
            self.weight = self.gradientDescent(X, Y, alpha, epoch, c)
        else:
            I = np.eye(X.shape[1])
            self.weight = np.linalg.inv((X.T).dot(X)+c*I).dot(X.T).dot(Y)
    
    def predict(self, test_data):
        X = np.ones((test_data.shape[0], test_data.shape[1]+1))
        X[:,0:-1] = test_data
        return X.dot(self.weight)
    
    def evaluate(self, predict_target, test_target):
        predict_target[predict_target>=0.5] = 1
        predict_target[predict_target<0.5] = 0
        return sum(predict_target==test_target)/len(predict_target)

if __name__ == "__main__":
    cancer = load_breast_cancer()
    train_data, train_target, test_data, test_target = div(cancer.data, cancer.target,0.5)
    linear = LinearRegression(training_way = 'gd')
    linear.fit(train_data, train_target, alpha = 0.05, epoch = 1000)
    predict = linear.predict(test_data)
    print('linear regression accruacy:',linear.evaluate(predict, test_target))
    lasso = LassoRegression()
    lasso.fit(train_data, train_target, 0.05, 1000, 0.01)
    lassoPredict = lasso.predict(test_data)
    print('lasso regression accruacy:',lasso.evaluate(lassoPredict,test_target))
    ridge = RidgeRegression(training_way = 'gd')
    ridge.fit(train_data, train_target, 0.05, 1000, 0.01)
    ridgePredict = ridge.predict(test_data)
    print('ridge regression accuracy:',ridge.evaluate(ridgePredict, test_target))
    