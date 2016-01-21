import numpy as np
import csv as csv
import DataInitialization as DI
from scipy import optimize
from math import exp
from astropy.convolution.tests.test_convolve_fft import options


input = csv.reader(open('C:/Users/Acer/Desktop/cs/books/Coursera/Machine Learning/mlclass-ex2-005/mlclass-ex2/ex2data1.txt'))
input2 = csv.reader(open('C:/Users/Acer/Desktop/cs/books/Coursera/Machine Learning/mlclass-ex2-005/mlclass-ex2/ex2data2.txt'))

def sigmoid(z):
    z = np.array(z);
    return (1/(1+np.exp(-z)))

def costFunction(theta, X , y):
    m = len(X)
    J =  (1/m)*(np.dot(-y.transpose(),np.log(sigmoid(np.dot(X,theta))))-
                     np.dot(1-y.transpose(),(np.log(1-(sigmoid(np.dot(X,theta)))))))
    grad = (1/m)*np.dot(X.transpose(),(sigmoid(np.dot(X,theta)) - y))
    return J,grad

def logisticRegression(X,y,initialTheta,options = {'full_output':True,'maxiter':400}):
    theta,cost,_,_,_ = \
        optimize.fmin(lambda t:costFunction(t, X, y),initialTheta,**options)
    return theta,cost
    
X,y,theta = DI.dataInitialization(input, False, True)
t,c = logisticRegression(X, y, theta)
X
y
Cost, grad = costFunction(theta,X,y)
#print ("Cost is %d"%costFunction(theta, X, y))
