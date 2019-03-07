# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 23:39:42 2019
ESZSL Synthetic Data Experiments
@author: afsar
"""
from eszsl import *
from plotit import plotit
import matplotlib.pyplot as plt
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


if __name__=='__main__':    
    
    a = 100 #number of attributes
    d = 10 #data dimensions
    z = 30 #number of training classes
    zd = 2 #number of test classes
    sigmap = 1.0
    lambdap = 1.0
    
    dgen = dataGenerator(a,d)
    S = dgen.getAttributes(z=z)
    X,Y = dgen.getData(S, n = 100)
    yc = np.argmax(Y,axis=1)
    #    X2, Y2, S2 = X[yc>=z-zd,:], Y[yc>=z-zd,(z-zd):], S[(z-zd):]
    #    X, Y, S = X[yc<(z-zd),:], Y[yc<z-zd,:(z-zd)], S[:(z-zd)]
    clf = ESZSL(sigmap = sigmap, lambdap = lambdap, kernel = poly, degree = 1)
    clf.fit(X,Y,S)
    
    print("Train Accuracy",accuracy(np.argmax(Y,axis=1),clf.predict(X,S)))
    S1 = S
    X1,Y1 = dgen.getData(S1, n = 50, useOldNormalization = True)
    
    print("Test Accuracy",accuracy(np.argmax(Y1,axis=1),clf.predict(X1,S1)))
    S2 = dgen.getAttributes(z=zd)#S+(2*(np.random.rand(*S.shape)>0.5)-1)*0.2#
    X2,Y2 = dgen.getData(S2, n = 50, useOldNormalization = True)    
    zsla = accuracy(np.argmax(Y2,axis=1),clf.predict(X2,S2))
    print("ZSL Accuracy",zsla)

    plt.close('all')
    plotit(X,np.argmax(Y,axis=1),clf=clf.predict,S=S, colors = 'random', ccolors = None, markers = None);
    plt.figure()
    plotit(X2,np.argmax(Y2,axis=1),clf=clf.predict,S=S2)
