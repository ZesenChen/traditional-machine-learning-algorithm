# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 21:17:59 2017

@author: 14094
"""

# coding=utf-8  
import numpy as np  
import struct
from numpy import random as nr

def loadImageSet(filename):  
    print("load image set",filename)  
    binfile= open(filename, 'rb')  
    buffers = binfile.read()  
   
    head = struct.unpack_from('>IIII' , buffers ,0)  
    print("head,",head)  
   
    offset = struct.calcsize('>IIII')  
    imgNum = head[1]  
    width = head[2]  
    height = head[3]  
    #[60000]*28*28  
    bits = imgNum * width * height  
    bitsString = '>' + str(bits) + 'B' #like '>47040000B'  
   
    imgs = struct.unpack_from(bitsString,buffers,offset)  
   
    binfile.close()  
    imgs = np.reshape(imgs,[imgNum,width*height])  
    print("load imgs finished")  
    return imgs  
   
def loadLabelSet(filename):  
   
    print("load label set",filename)  
    binfile = open(filename, 'rb')  
    buffers = binfile.read()  
   
    head = struct.unpack_from('>II' , buffers ,0)  
    print("head,",head)  
    imgNum=head[1]  
   
    offset = struct.calcsize('>II')  
    numString = '>'+str(imgNum)+"B"  
    labels = struct.unpack_from(numString , buffers , offset)  
    binfile.close()  
    labels = np.reshape(labels,[imgNum,1])  
   
    print('load label finished')  
    return labels  

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

def encode(target):
    #data = scale(data)
    tmp_target = np.zeros((target.shape[0],max(target)[0]+1))
    print(target.shape)
    for i in range(target.shape[0]):
        tmp_target[i][target[i]] = 1
    return tmp_target
  
def tanh(x):  
    return np.tanh(x)  
  
def tanh_deriv(x):  
    return 1.0- np.tanh(x)*np.tanh(x)  
  
def logistic(x):  
    return 1/(1+np.exp(-x))  
  
def logistic_derivative(x):  
    return logistic(x)*(1-logistic(x))  
  
class NeuralNetwork:  
    def __init__(self,layers,activation='tanh'):  
        """ 
 
        """  
        if activation == 'logistic':  
            self.activation = logistic  
            self.activation_deriv = logistic_derivative  
        elif activation=='tanh':  
            self.activation = tanh  
            self.activation_deriv=tanh_deriv  
  
        self.weights=[]  
        self.weights.append((2*np.random.random((layers[0]+1,layers[1]))-1)*0.25)  
        for i in range(2,len(layers)):  
            self.weights.append((2*np.random.random((layers[i-1],layers[i]))-1)*0.25)  
            #self.weights.append((2*np.random.random((layers[i]+1,layers[i+1]))-1)*0.25)  
  
    def fit(self,X,y,learning_rate=0.2,epochs=10000):  
        X = np.atleast_2d(X)  
            # atlest_2d函数:确认X至少二位的矩阵  
        temp = np.ones(([X.shape[0],X.shape[1]+1]))  
            #初始化矩阵全是1（行数，列数+1是为了有B这个偏向）  
        temp[:,0:-1]=X  
            #行全选，第一列到倒数第二列  
        X=temp  
        Y=np.array(y)  
            #数据结构转换  
        for k in range(epochs):  
                # 抽样梯度下降epochs抽样  
            x,y = RandSam(X, Y, 50)
            #i = np.random.randint(X.shape[0])  
            a = [x]  
  
            for l in range(len(self.weights)):  
                a.append(self.activation(np.dot(a[l],self.weights[l])))  
                # 向前传播，得到每个节点的输出结果  
            error = y-a[-1]  
                # 最后一层错误率  
            deltas=[error*self.activation_deriv(a[-1])]  
  
            for l in range(len(a)-2,0,-1):  
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))  
            deltas.reverse()  
            for i in range(len(self.weights)):  
                layer = np.atleast_2d(a[i])  
                delta = np.atleast_2d(deltas[i])  
                self.weights[i] += learning_rate*layer.T.dot(delta) 
  
    def predict(self,x):
        y = []
        for i in range(x.shape[0]):
        #x=np.array(x)  
            temp= np.ones(x.shape[1]+1)  
            temp[0:-1]=x[i]  
            a = temp  
            for l in range(0,len(self.weights)):  
                a=self.activation(np.dot(a,self.weights[l]))
            y.append(a)
        return np.array(y) 
    
    def evaluate(self, predict_target, test_target):
        count = 0
        for i in range(predict_target.shape[0]):
            if np.where(predict_target[i]==max(predict_target[i]))[0][0] == np.where(test_target[i]==1)[0][0]:
                count += 1
        return count/predict_target.shape[0]
        
if __name__ == "__main__":
    train_data = loadImageSet('train-images.idx3-ubyte')
    train_target = loadLabelSet('train-labels.idx1-ubyte')
    test_data = loadImageSet('t10k-images.idx3-ubyte')
    test_target = loadLabelSet('t10k-labels.idx1-ubyte')
    train_target = encode(train_target)
    test_target = encode(test_target)
    nn = NeuralNetwork([784,120,10], activation='logistic')
    nn.fit(train_data, train_target, learning_rate=0.01, epochs=1000)
    predict_target = nn.predict(test_data)
    print(nn.evaluate(predict_target, test_target))