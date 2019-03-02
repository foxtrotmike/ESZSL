# -*- coding: utf-8 -*-
"""
@author: Dr. Fayyaz Minhas
@author-email: afsar at pieas dot edu dot pk
Implementation of embarrasingly simple zero shot learning
"""
from numpy.random import randn #importing randn
import numpy as np #importing numpy
import matplotlib.pyplot as plt #importing plotting module
import itertools
import warnings

def plotit(X,Y=None,clf=None, markers = ('s','o'), hold = False, transform = None):
    """
    Just a function for showing a data scatter plot and classification boundary
    of a classifier clf
    """
    if X.shape[1]!=2:
        warnings.warn("Data Dimensionality is not 2. Unable to plot.")
        return
    
    eps=1e-6
    minx, maxx = np.min(X[:,0]), np.max(X[:,0])
    miny, maxy = np.min(X[:,1]), np.max(X[:,1])
    
    if clf is not None:
        npts = 150
        x = np.linspace(minx,maxx,npts)
        y = np.linspace(miny,maxy,npts)
        t = np.array(list(itertools.product(x,y)))
        if transform is not None:
            t = transform(t)
        z = clf(t)
        z = np.reshape(z,(npts,npts)).T        
        extent = [minx,maxx,miny,maxy]
        plt.contour(x,y,z,[-1+eps,0,1-eps],linewidths = [2],colors=('b','k','r'),extent=extent, label='f(x)=0')
        #plt.imshow(np.flipud(z), extent = extent, cmap=plt.cm.Purples, vmin = -2, vmax = +2); plt.colorbar()
        plt.pcolormesh(x, y, z,cmap=plt.cm.Purples,vmin=-2,vmax=+2);plt.colorbar()
        plt.axis([minx,maxx,miny,maxy])
    
    if Y is not None:
        
        plt.scatter(X[Y==1,0],X[Y==1,1],marker = markers[0], c = 'y', s = 30)
        plt.scatter(X[Y==-1,0],X[Y==-1,1],marker = markers[1],c = 'c', s = 30)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')        
         
    else:
        plt.scatter(X[:,0],X[:,1],marker = '.', c = 'k', s = 5)
    if not hold:
        plt.grid()
        
        plt.show()
    
def accuracy(ytarget,ypredicted):
    return np.sum(ytarget == ypredicted)/len(ytarget)


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
    Y = np.array([+1]*n+[-1]*int(n)) #Associate Labels
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
        """
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
        KK = np.linalg.inv(KK+self.lambdap*np.eye(KK.shape[0]))
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
            S = np.eye(clf.A.shape[1])[-1,:]
        if self.kernel=='precomputed':
            K = X
        else:
            K = self.kernel(X,self.X,**self.kwargs)
        Z = np.dot(np.dot(S,self.A.T),K.T).T
        return Z
        
if __name__ == '__main__':
    plt.close("all")    
    #%% Data Generation
    n = 500 #number of examples of each class
    d = 2 #number of dimensions
    Xtr,Ytr = getExamples(n=n,d=d) #Generate Training Examples    
    print("Number of positive examples in training: ", np.sum(Ytr==1))
    print("Number of negative examples in training: ", np.sum(Ytr==-1))
    print("Dimensions of the data: ", Xtr.shape[1])   
    Xtt,Ytt = getExamples(n=100,d=d) #Generate Testing Examples 
    #%% Setting up classlabel matrix Y and attribute matrix S for binary classification
    Y = -1*np.ones((Xtr.shape[0],2))
    for i,v in enumerate(np.array((Ytr+1)/2,dtype=np.int)):
        Y[i,v]=1
    S = np.eye(2)        
    #%% Training and evaluation, plotting
    clf = ESZSL(sigmap = 0.1, lambdap = 1.0, kernel = poly, degree = 3)
    clf.fit(Xtr,Y,S)
    Z = clf.decision_function(Xtr,S)[:,1]
    print("Train accuracy",accuracy(Ytr,2*(Z>0)-1))
    Z = clf.decision_function(Xtt,S)[:,1]
    print("Test accuracy",accuracy(Ytt,2*(Z>0)-1))
    plotit(Xtr,Ytr,clf=clf.decision_function)    
    #%% Training and evaluation for precomputed matrix
    K = (np.dot(Xtr,Xtr.T)+1)**2
    clf = ESZSL(sigmap = 0.1, lambdap = 0.1, kernel = 'precomputed')
    clf.fit(K,Y,S)
    Z = clf.decision_function(K,S)[:,1]
    print("Train accuracy",accuracy(Ytr,2*(Z>0)-1))
    Z = clf.decision_function((np.dot(Xtt,Xtr.T)+1)**2,S)[:,1]
    print("Test accuracy",accuracy(Ytt,2*(Z>0)-1))
    
