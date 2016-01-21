import numpy as np
#adds x0 to x
#initialize theta array
def dataInitialization(csvinput, normalizeFlag = False, addx0 = True):
    data=[]
    for row in csvinput:
        data.append(row[0:])
    data = np.array(data)
    nrow = len(data)
    ncol = data.shape[1]
     #add     a column of ones to X
    if (addx0):
        data = np.append(np.ones(nrow).reshape(nrow,1), data , 1).astype(np.float)
    #initializa theta vector
    y = data[::,ncol].reshape(nrow,1).astype(np.float)
    X = data[::,:ncol].astype(np.float)
    theta = np.zeros((ncol,1)).astype(np.float)
    
    #normalization
    if(normalizeFlag):
        X = featureNormalize(X,1, True)
        
    return X, y, theta
    
    
def featureNormalize(X,ddof = 1,withX0 = True):
    # X minus the mean then divides the std
    # does not modify normalize X0
    X_norm = X;
    mu = X.mean(axis = 0)
    sigma = X.std(axis = 0,ddof = ddof)
    if (withX0):
        sigma[0] = 1
        mu[0] = 0
    
    X_norm = X - mu
    X_norm = X_norm/sigma
    sigma[0] = 0
    mu[0] = 1
    return X_norm
