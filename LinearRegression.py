import numpy as np
#import pandas as pd
import csv as csv
import DataInitialization as DI

input = csv.reader(open('C:/Users/Acer/Desktop/cs/books/Coursera/Machine Learning/mlclass-ex1-005/mlclass-ex1-005/mlclass-ex1py/Machine Learning/ex1data1.txt'))
input2 = csv.reader(open('C:/Users/Acer/Desktop/cs/books/Coursera/Machine Learning/mlclass-ex1-005/mlclass-ex1-005/mlclass-ex1py/Machine Learning/ex1data2.txt'))




def computeCost(X,y,theta):
    m = y.size
    prediction = np.dot(X,theta)
    sqErrors = (prediction - y) ** 2
    J = 1/(2*m)*sum(sqErrors)  
    return J

def gradientDescent(X, y, theta, alpha = 0.01, num_iters = 1500):
    #GRADIENTDESCENT Performs gradient descent to learn theta
    #theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
    #taking num_iters gradient steps with learning rate alpha
    print ('alpha = ', alpha,)
    print ('iterations = ', num_iters)
    m = y.size
    J_history = np.zeros((num_iters,1)).astype(np.float)
    for i in range(0,num_iters):
        theta = theta - alpha/m * (X.transpose().dot((np.dot(X,theta)-y)))
        J_history[i] = computeCost(X, y, theta);
        print ("Iteration %d"%i)
        
    return theta,J_history


    #X,y,theta = DI.dataInitialization(input2, True,True)
#theta,jhist = gradientDescent(X, y, theta,0.01,400)
#print ("New theta ",theta.flatten())
