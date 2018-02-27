import numpy as np
import random

def KMeans(X, class_num, max_circle = 300):
    data_num, feature_num = X.shape[0], X.shape[1]
    init = [random.randint(0,class_num) for i in range(class_num)]
    kmeans = X[init]
    circle_num, conv_num = 0,0
    Y,tmpY = np.zeros((data_num,),dtype=int),np.zeros((data_num,),dtype=int)
    while True:
        for i in range(data_num):
            dis = np.sum((kmeans-X[i])**2,1)
            Y[i] = np.where(dis==min(dis))[0][0]
        conv_num = conv_num+1 if sum(Y==tmpY)==data_num else 0
        if conv_num > 0.05*max_circle or circle_num > max_circle:
            break
        for j in range(class_num):
            kmeans[j] = np.sum(X[Y==j],0)/X[Y==j].shape[0]
        tmpY = Y
        circle_num += 1
    return Y

if __name__ == '__main__':
    X = np.array([[1,2],[3,4],[-1,-2],[-9,-1.2],[5.6,3],[9,1], \
                  [-9,-3],[3,1],[-2,-3],[4,4]])
    print(KMeans(X,2))
