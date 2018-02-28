# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 14:07:11 2017

@author: 14094
"""

import numpy as np
import struct

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

def encode(target):
    #data = scale(data)
    tmp_target = np.zeros((target.shape[0],max(target)[0]+1))
    print(target.shape)
    for i in range(target.shape[0]):
        tmp_target[i][target[i]] = 1
    return tmp_target


def sigmoid(x):
    return 1/(np.exp(-x)+1)

def softmax(x):
    y = np.zeros(x.shape)
    for i in range(x.shape[0]):
        y[i] = np.exp(x[i])/sum(np.exp(x[i]))
    return y

class BPClassifier(object):
    x_num = 0
    h_num = 0
    y_num = 0
    epochs = 1000
    batch = 20
    rate = 0.01
    L2 = 0.02
    acc = []
    Wxh = np.array([])
    Why = np.array([])
    Bxh = np.array([])
    Bhy = np.array([])
    def __init__(self, input_num, hidden_num, output_num, training_epochs, batch_num, learning_rate, l2):
        self.x_num = input_num
        self.h_num = hidden_num
        self.y_num = output_num
        self.epochs = training_epochs
        self.batch = batch_num
        self.rate = learning_rate
        self.L2 = l2
        #self.Wxh = (2*np.random.random(size=(input_num+1, hidden_num))-1)*0.25
        self.Wxh = np.random.normal(size=(input_num, hidden_num))
        self.Bxh = np.random.normal(size=(hidden_num,))
        self.Why = np.random.normal(size=(hidden_num, output_num))
        self.Bhy = np.random.normal(size=(output_num,))
    
    def SumRow(self, x):
        tmp = np.zeros((x.shape[1],))
        for i in range(x.shape[0]):
            tmp += x[i]
        return tmp
    
    def shuffle(self, train_data, train_target):
        for i in range(train_data.shape[0]-1, 0, -1):
            num = np.random.randint(0, i)
            tmpData = train_data[i]
            train_data[i] = train_data[num]
            train_data[num] = tmpData
            tmpTarget = train_target[i]
            train_target[i] = train_target[num]
            train_target[num] = tmpTarget
        return train_data, train_target
    
    def update_mini_batch(self, train_data, train_target, batch_size):
        tmpD = []
        tmpT = []
        for i in range(batch_size):
            num = np.random.randint(0,train_data.shape[0])
            tmpD.append(train_data[num])
            tmpT.append(train_target[num])
        return np.array(tmpD), np.array(tmpT)
    
    def fit(self, train_data, train_target, test_data, test_target):
        data_num = train_data.shape[0]
        for i in range(self.epochs):
            print('the ', i, 'th circle.')
            if i%200==0:
                predict_target = self.predict(test_data)
                self.acc.append(self.evaluate(predict_target, test_target))
                print('the accuracy is ', self.acc[-1])
            x,y = self.update_mini_batch(train_data, train_target, self.batch)
            zxh = np.dot(x,self.Wxh) + self.Bxh
            yxh = sigmoid(zxh)
            zhy = np.dot(yxh, self.Why) + self.Bhy
            yo = softmax(zhy)
                #print(yo)
            Res_hy = y-yo
            Res_xh = np.dot(Res_hy,self.Why.T)*yxh*(1-yxh)
            Delta_Why = np.dot(yxh.T,Res_hy)/self.batch
            Delta_Bhy = self.SumRow(Res_hy)/self.batch
            Delta_Wxh = np.dot(x.T, Res_xh)/self.batch
            Delta_Bxh = self.SumRow(Res_xh)/self.batch
            self.Why = (1 - self.rate*self.L2/data_num)*self.Why + self.rate*Delta_Why
            self.Wxh = (1 - self.rate*self.L2/data_num)*self.Wxh + self.rate*Delta_Wxh
            self.Bhy = self.Bhy + self.rate*Delta_Bhy
            self.Bxh = self.Bxh + self.rate*Delta_Bxh
            
    def predict(self, test_data):
        #temp = np.ones((test_data.shape[0], test_data.shape[1]+1))
        #temp[:,0:-1] = test_data
        yxh = sigmoid(np.dot(test_data, self.Wxh)+self.Bxh)
        yo = softmax(np.dot(yxh, self.Why)+self.Bhy)
        return yo
    
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
    bp = BPClassifier(784, 100, 10, 1000000, 100, 0.1, 0)
    bp.fit(train_data, train_target, test_data, test_target)
    predict_target = bp.predict(test_data)
    print(bp.acc)
    
                    