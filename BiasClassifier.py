# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 22:58:39 2017

@author: 14094
"""

import numpy as np
import math
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

class BiasClassifier(object):
    X = np.array([])
    Y = np.array([])
    data_num = 0
    feature_size = 0
    Ppos = 0
    Pneg = 0
    Ave_pos = np.array([])
    Ave_neg = np.array([])
    Var_pos = np.array([])
    Var_neg = np.array([])
    Cov_pos = np.array([])
    Cov_neg = np.array([])
    
    def __init__(self, train_data, train_target):
        self.X = train_data
        self.Y = train_target
        self.data_num = train_data.shape[0]
        self.feature_size = train_data.shape[1]
        self.Ppos = sum(train_target)/self.data_num
        self.Pneg = 1 - self.Ppos
        posSum = np.zeros((self.feature_size,))
        negSum = np.zeros((self.feature_size,))
        posData = np.array([])
        negData = np.array([])
        for i in range(self.data_num):
            if train_target[i] == 1:
                posSum += train_data[i]
                if posData.size == 0:
                    posData = train_data[i]
                else:
                    posData = np.vstack((posData,train_data[i]))
            else:
                negSum += train_data[i]
                if negData.size == 0:
                    negData = train_data[i]
                else:
                    negData = np.vstack((negData,train_data[i]))
        self.Ave_pos = posSum/sum(train_target)
        self.Ave_neg = negSum/(self.data_num - sum(train_target))
        self.Var_pos = np.zeros((self.feature_size,))
        self.Var_neg = np.zeros((self.feature_size,))
        for i in range(self.data_num):
            self.Var_pos += (train_data[i] - self.Ave_pos)**2
            self.Var_neg += (train_data[i] - self.Ave_neg)**2
        self.Var_pos = self.Var_pos/sum(train_target)
        self.Var_neg = self.Var_neg/(self.data_num - sum(train_target))
        self.Cov_pos = np.cov(posData.T)
        self.Cov_neg = np.cov(negData.T)
    
    def predict(self, test_data):
        predict_target = []
        Cov_pos_det = np.sqrt(np.linalg.det(self.Cov_pos))
        Cov_neg_det = np.sqrt(np.linalg.det(self.Cov_neg))
        Cov_pos_inv = np.linalg.inv(self.Cov_pos)
        Cov_neg_inv = np.linalg.inv(self.Cov_neg)
        for i in range(test_data.shape[0]):
            tmp1 = math.pow((2*np.pi),test_data.shape[0]/2)#*np.sqrt(Cov_pos_det)
            tmp_pos = tmp1*Cov_pos_det
            tmp_neg = tmp1*Cov_neg_det
            tmp_pos_exp = -np.dot(np.dot((test_data[i]-self.Ave_pos),Cov_pos_inv),((test_data[i]-self.Ave_pos).T))/2
            tmp_neg_exp = -np.dot(np.dot((test_data[i]-self.Ave_neg),Cov_neg_inv),((test_data[i]-self.Ave_neg).T))/2
            P_X_Cpos = np.exp(tmp_pos_exp)/tmp_pos
            P_X_Cneg = np.exp(tmp_neg_exp)/tmp_neg
            tmp_target = self.Ppos*P_X_Cpos/(P_X_Cpos + P_X_Cneg)
            if tmp_target >= 0.5:
                predict_target.append(1)
            else:
                predict_target.append(0)
        return predict_target
    
    def evaluate(self, predict_target, test_target):
        err = 0
        for i in range(len(predict_target)):
            if predict_target[i] != test_target[i]:
                err += 1
        return err/len(predict_target)

if __name__ == "__main__":
    cancer = load_breast_cancer()
    train_data, train_target, test_data, test_target = div(cancer.data, cancer.target,0.5)
    bais = BiasClassifier(train_data, train_target)
    predict = bais.predict(test_data)
    print(bais.evaluate(predict,test_target))
              
            
            