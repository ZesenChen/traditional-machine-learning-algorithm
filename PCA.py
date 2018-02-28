import numpy as np


#d:the dimension that we want X to reduce to
def PCA(X, d):
    num = X.shape[0]
    x_mean = np.sum(X,0)/num
    newX = X - x_mean
    S = newX.T.dot(newX)
    egnum, egvec = np.linalg.eig(S)
    tmpeg = np.hstack((egvec, egnum.reshape([egnum.shape[0],1])))
    sorted_eg = np.array(sorted(tmpeg, key = lambda s:s[-1], reverse = True))
    egnum, egvec = sorted_eg[0:d,-1], sorted_eg[0:d,0:-1]
    return egvec.T, np.dot(X,egvec.T)
    
if __name__ == '__main__':
    X = np.array([[1,2,3,4],[0,1,1,1],[-1,2,3,4],[5,5,5,7],[0,0,1,1]])
    eg, newX = PCA(X,4)
    print(newX)
