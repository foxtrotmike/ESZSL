# -*- coding: utf-8 -*-
"""
@author: Dr. Fayyaz Minhas
@author-email: afsar at pieas dot edu dot pk
Implementation of embarrasingly simple zero shot learning
"""
from __future__ import print_function
from numpy.random import randn #importing randn
import numpy as np #importing numpy
from plotit import plotit
    
def accuracy(ytarget,ypredicted):
    return np.mean(ytarget == ypredicted)


def getExamples(n=100,d=2):
    """
    Generates n d-dimensional normally distributed examples of each class        
    The mean of the positive class is [1] and for the negative class it is [-1]
    DO NOT CHANGE THIS FUNCTION
    """
    Xp = randn(n,d)#+1   #generate n examples of the positie class
    Xp=Xp
    Xn = randn(int(n/2),d)#-1   #generate n examples of the negative class
    Xn=Xn-5
    Xn2 = randn(int(n/2),d)#-1   #generate n examples of the negative class
    Xn2=Xn2+5
    Xn = np.vstack((Xn,Xn2))
    X = np.vstack((Xp,Xn))  #Stack the examples together to a single matrix
    Y = np.array([0]*n+[1]*int(n/2)+[2]*int(n/2)) #Associate Labels
    return (X,Y) 


def poly(X1,X2,**kwargs):
    if 'degree' not in kwargs:
        d = 1
    else:
        d = kwargs['degree']
        
    return (np.dot(X1,X2.T)+1)**d

class ESZSL:
    """
    Implementation of Generalized Zero Shot Learning
    Author: Dr. Fayyaz Minhas, DCIS, PIEAS
    It implements the paper by Bernardino Romera-Paredes and Philip H. S. Torr
    An embarrassingly simple approach to zero-shot learning, (ICML 2015).
    No warranties. Under Creative Commons License.
    """
    def __init__(self, lambdap = 0.1, sigmap = 0.1, kernel = poly, **kwargs):
        """
        lambdap: Regularization parameter for kernel/feature space
        sigmap: Regularization parameter for Attribute Space
        kernel: kernel function (default is poly)
        kwargs: optional, any kernel arguments
        """
        self.sigmap = sigmap
        self.lambdap = lambdap
        self.A = None
        self.kwargs = kwargs
        self.kernel = kernel
        
    def fit(self,X,Y,S = None):
        """    X,Y = getExamples()
    clf = lambda x: 2*np.sum(x,axis=1)-2.5 #dummy classifier
    plotit(X = X, Y = Y, clf = clf, conts =[-1,0,1], colors = 'random')
    1/0
    plt.close("all")    
        Training:
            X: (mxd) kernel matrix (m is the number of triaining examples,d is feature dims)
            Y: (mxz) label matrix (z is the number of classes)
            S: (zxa) attribute matrix for all classes (a is the number of attributes)
                Optional: Default, S = I
        Training: (when self.kernel = "precomputed")
            X = K: (mxm) kernel matrix (m is the number of triaining examples)
        It computes mxa sized $$A=(K^TK+\lambda I)^{-1}KYS(S^TS+\sigma I)^{-1}$$
        
        """
        if S is None:
            S = np.eye(Y.shape[1])

        if self.kernel=='precomputed':
            K = X
        else:            
            self.X = X
            K = self.kernel(X,X,**self.kwargs) 
        KK = np.dot(K.T,K)
        KK = np.linalg.inv(KK+self.lambdap*(np.eye(K.shape[0])))
        KYS = np.dot(np.dot(K,Y),S)    
        SS = np.dot(S.T,S)
        SS = np.linalg.inv(SS+self.sigmap*np.eye(SS.shape[0]))
        self.A = np.dot(np.dot(KK,KYS),SS)
        
    def decision_function(self,X,S = None):
        """
        Testing:
            X: (m'xd) kernel matrix (m' is the number of test examples)
            S: (z'xa) attribute matrix (z' is the number of test classes)        
       
        Testing: (when self.kernel = "precomputed")
            X = K: (m'xm) kernel matrix (m' is the number of test examples)            
        It implements: $$Z = SA^TK^T$$ to generate class scores for each test example 
        Returns:
            Z: (m'xz') matrix of class scores
        
        """
        if S is None:
            S = np.eye(clf.A.shape[1])[0,:]

        if self.kernel=='precomputed':
            K = X
        else:
            K = self.kernel(X,self.X,**self.kwargs)
        Z = np.dot(np.dot(S,self.A.T),K.T).T
        return Z
    def predict(self,X,S=None):
        return np.argmax(self.decision_function(X,S),axis=1)
        
if __name__ == '__main__':
#%% Data Generation for simple classification 

    n = 500 #number of examples of each class
    d = 2 #number of dimensions
    Xtr,Ytr = getExamples(n=n,d=d) #Generate Training Examples    
    print("Number of positive examples in training: ", np.sum(Ytr==1))
    print("Number of negative examples in training: ", np.sum(Ytr==-1))
    print("Dimensions of the data: ", Xtr.shape[1])   
    Xtt,Ytt = getExamples(n=100,d=d) #Generate Testing Examples 
    z  = len(set(Ytt))
    #%% Setting up classlabel matrix Y and attribute matrix S for binary classification
    Y = -1*np.ones((Xtr.shape[0],z))
    for i in range(len(Y)):
        Y[i,Ytr[i]]=1
    S = np.eye(z,z)
    
    
         
    #%% Training and evaluation, plotting
    clf = ESZSL(sigmap = 0.1, lambdap = 0.05, kernel = poly, degree = 1)
    clf.fit(Xtr,Y,S)
    Z = clf.decision_function(Xtr,S)[:,1]
    print("Train accuracy",accuracy(Ytr,clf.predict(Xtr,S)))
    Z = clf.decision_function(Xtt,S)[:,1]
    print("Train accuracy",accuracy(Ytt,clf.predict(Xtt,S)))
    plotit(Xtr,Ytr,clf=clf.predict,S=S,colors='random')    
    
#%% Training and evaluation for precomputed matrix
    K = (np.dot(Xtr,Xtr.T)+1)**2
    clf = ESZSL(sigmap = 0.1, lambdap = 0.1, kernel = 'precomputed')
    clf.fit(K,Y,S)
    Ktt = (np.dot(Xtt,Xtr.T)+1)**2
    print("Train accuracy",accuracy(Ytr,clf.predict(K,S)))
    print("Test accuracy",accuracy(Ytt,clf.predict(Ktt,S)))
    

