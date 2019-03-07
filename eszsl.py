# -*- coding: utf-8 -*-
"""
@author: Dr. Fayyaz Minhas
@author-email: afsar at pieas dot edu dot pk
Implementation of embarrasingly simple zero shot learning
"""
from __future__ import print_function
from numpy.random import randn #importing randn
import numpy as np #importing numpy
import matplotlib.pyplot as plt #importing plotting module
import itertools
import warnings

def plotit(X,Y=None,clf=None,  conts = None, ccolors = ('b','k','r'), colors = ('c','y'), markers = ('s','o'), hold = False, transform = None,**kwargs):
    """
    A function for showing data scatter plot and classification boundary
    of a classifier for 2D data
        X: nxd  matrix of data points
        Y: (optional) n vector of class labels
        clf: (optional) classification/discriminant function handle
        conts: (optional) contours (if None, contours are drawn for each class boundary)
        ccolors: (optional) colors for contours   
        colors: (optional) colors for each class (sorted wrt class id)
            can be 'scaled' or 'random' or a list/tuple of color ids
        markers: (optional) markers for each class (sorted wrt class id)
        hold: Whether to hold the plot or not for overlay (default: False).
        transform: (optional) a function handle for transforming data before passing to clf
        kwargs: any keyword arguments to be passed to clf (if any)        
    """
    if clf is not None and X.shape[1]!=2:
        warnings.warn("Data Dimensionality is not 2. Unable to plot.")
        return
    if markers is None:
        markers = ('.',)
        
    d0,d1 = (0,1)
    minx, maxx = np.min(X[:,d0]), np.max(X[:,d0])
    miny, maxy = np.min(X[:,d1]), np.max(X[:,d1])
    eps=1e-6

    
    if Y is not None:
        classes = sorted(set(Y))
        if conts is None:
            conts = list(classes)        
        vmin,vmax = classes[0]-eps,classes[-1]+eps
    else:
        vmin,vmax=-2-eps,2+eps
        if conts is None:            
            conts = sorted([-1+eps,0,1-eps])
        
    if clf is not None:
        npts = 150
        x = np.linspace(minx,maxx,npts)
        y = np.linspace(miny,maxy,npts)
        t = np.array(list(itertools.product(x,y)))
        if transform is not None:
            t = transform(t)
        z = clf(t,**kwargs)
        
        z = np.reshape(z,(npts,npts)).T        
        extent = [minx,maxx,miny,maxy]
        
        plt.contour(x,y,z,conts,linewidths = [2],colors=ccolors,extent=extent, label='f(x)=0')
        #plt.imshow(np.flipud(z), extent = extent, cmap=plt.cm.Purples, vmin = -2, vmax = +2); plt.colorbar()
        plt.pcolormesh(x, y, z,cmap=plt.cm.Purples,vmin=vmin,vmax=vmax);plt.colorbar()
        plt.axis([minx,maxx,miny,maxy])
    
    if Y is not None:        
        for i,y in enumerate(classes):
            if colors is None or colors=='scaled':
                cc = np.array([[i,i,i]])/float(len(classes))
            elif colors =='random':
                cc = np.array([[np.random.rand(),np.random.rand(),np.random.rand()]])
            else:
                cc = colors[i%len(colors)]
            mm = markers[i%len(markers)]
            plt.scatter(X[Y==y,d0],X[Y==y,d1], marker = mm,c = cc, s = 30)     
         
    else:
        plt.scatter(X[:,d0],X[:,d1],marker = markers[0], c = 'k', s = 5)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')   
    if not hold:
        plt.grid()        
        plt.show()
    
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

#    n = 500 #number of examples of each class
#    d = 2 #number of dimensions
#    Xtr,Ytr = getExamples(n=n,d=d) #Generate Training Examples    
#    print("Number of positive examples in training: ", np.sum(Ytr==1))
#    print("Number of negative examples in training: ", np.sum(Ytr==-1))
#    print("Dimensions of the data: ", Xtr.shape[1])   
#    Xtt,Ytt = getExamples(n=100,d=d) #Generate Testing Examples 
#    z  = len(set(Ytt))
#    #%% Setting up classlabel matrix Y and attribute matrix S for binary classification
#    Y = -1*np.ones((Xtr.shape[0],z))
#    for i in range(len(Y)):
#        Y[i,Ytr[i]]=1
#    S = np.eye(z,z)
#    
#    
#         
#    #%% Training and evaluation, plotting
#    clf = ESZSL(sigmap = 0.1, lambdap = 0.05, kernel = poly, degree = 1)
#    clf.fit(Xtr,Y,S)
#    Z = clf.decision_function(Xtr,S)[:,1]
#    print("Train accuracy",accuracy(Ytr,clf.predict(Xtr,S)))
#    Z = clf.decision_function(Xtt,S)[:,1]
#    print("Train accuracy",accuracy(Ytt,clf.predict(Xtt,S)))
#    plotit(Xtr,Ytr,clf=clf.predict,S=S)    
#    1/0
##%% Training and evaluation for precomputed matrix
#    K = (np.dot(Xtr,Xtr.T)+1)**2
#    clf = ESZSL(sigmap = 0.1, lambdap = 0.1, kernel = 'precomputed')
#    clf.fit(K,Y,S)
#    Z = clf.decision_function(K,S)[:,1]
#    print("Train accuracy",accuracy(Ytr,2*(Z>0)-1))
#    Z = clf.decision_function((np.dot(Xtt,Xtr.T)+1)**2,S)[:,1]
#    print("Test accuracy",accuracy(Ytt,2*(Z>0)-1))
#    
#%% Generalized zero shot learning
    class dataGenerator:
        def __init__(self,a=100,d=2):
            self.a = a
            self.d = d
            self.V = 1*np.random.randn(self.a,self.d)
        def getAttributes(self,z = 3):
            return np.random.binomial(1,0.5,(z,self.a))        
        def getData(self,S, n = 50, useOldNormalization = False):
            z,a = S.shape
            assert a==self.a
            d = self.d
            C = np.dot(S,self.V)
            X = []
            Y = -1*np.ones((n*z,z))
            for i in range(z):
                X.append(0.5*np.random.randn(n,d)+C[i,:])
                Y[i*n:(i+1)*n,i]=1.0
            X = np.vstack(X)
            if not useOldNormalization:
                self.mean = np.mean(X,axis=0)
                self.std = np.std(X,axis=0)
            X = (X-self.mean)/self.std
            return X,Y
            
    def MCAccuracy(Y,Z):
        return np.mean(np.argmax(Z,axis=1)==np.argmax(Y,axis=1))
    
    
    a = 100
    d = 2
    z = 8
    zd = 2
    sigmap = 0.1
    lambdap = 0.1
    
    dgen = dataGenerator(a,d)
    S = dgen.getAttributes(z=z)
    X,Y = dgen.getData(S, n = 100)
    yc = np.argmax(Y,axis=1)
    #    X2, Y2, S2 = X[yc>=z-zd,:], Y[yc>=z-zd,(z-zd):], S[(z-zd):]
    #    X, Y, S = X[yc<(z-zd),:], Y[yc<z-zd,:(z-zd)], S[:(z-zd)]
    clf = ESZSL(sigmap = sigmap, lambdap =lambdap, kernel = poly, degree = 3)
    clf.fit(X,Y,S)
    Z = clf.decision_function(X,S)
#                print("Train Accuracy",MCAccuracy(Y,Z))
    S1 = S
    X1,Y1 = dgen.getData(S1, n = 50, useOldNormalization = True)
    Z1 = clf.decision_function(X1,S1)
    print("Test Accuracy",MCAccuracy(Y1,Z1))
    S2 = dgen.getAttributes(z=zd)#S+(2*(np.random.rand(*S.shape)>0.5)-1)*0.2#
    X2,Y2 = dgen.getData(S2, n = 50, useOldNormalization = True)
    Z2 = clf.decision_function(X2,S2)
    zsla = MCAccuracy(Y2,Z2)
    print("ZSL Accuracy",zsla)

    plt.close('all')
    plotit(X,np.argmax(Y,axis=1),clf=clf.predict,S=S, colors = 'random', ccolors = None, markers = None);
    plt.figure()
    plotit(X2,np.argmax(Y2,axis=1),clf=clf.predict,S=S2)