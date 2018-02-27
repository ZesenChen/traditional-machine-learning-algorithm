import numpy as np

#Moore-Penrose
def inverse(a):  
    """ 
    奇异值分解求矩阵的逆 
    :param a: 
    :return: 
    """  
    U, s, V = np.linalg.svd(a)  
    m, n = a.shape  
    min_dim = min(m, n)  
    S = np.zeros((n, m))  
    S[:min_dim, :min_dim] = np.diag(1.0 / s)  
    return np.dot(np.linalg.inv(V), np.dot(S, np.linalg.inv(U)))  

#binary classification
def BLDA(X, Y, d):
    c1,c2 = min(Y), max(Y)
    m1 = sum(X[Y==c1,:],0)/X[Y==c1,:].shape[0]
    m2 = sum(X[Y==c2,:],0)/X[Y==c2,:].shape[0]
    S1 = np.dot((X[Y==c1,:]-m1).T,(X[Y==c1,:]-m1))
    S2 = np.dot((X[Y==c2,:]-m2).T,(X[Y==c2,:]-m2))
    tmp = m1-m2
    SB = tmp.reshape([tmp.shape[0],1]).dot(tmp.reshape([1,tmp.shape[0]]))
    S = inverse(S1+S2).dot(SB)
    egnum, egvec = np.linalg.eig(S)
    tmpeg = np.hstack((egvec, egnum.reshape([egnum.shape[0],1])))
    sorted_eg = np.array(sorted(tmpeg, key = lambda s:s[-1], reverse = True))
    egnum, egvec = sorted_eg[0:d,-1], sorted_eg[0:d,0:-1]
    return egvec.T, np.dot(X,egvec.T)

if __name__ == '__main__':
    X = np.array([[1,2,3,4],[2,2,2,2],[-1,-1,-2,-3],[-1,-9,-3,-4]])
    Y = np.array([1,1,0,0])
    w,newX = BLDA(X,Y,1)
    print(newX)
