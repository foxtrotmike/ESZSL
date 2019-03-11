# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:26:04 2019
ESZSL SUN Data Results
Requires mat files in ddir
Perfect Reconstruction of results in the original paper
Download the data mat files from the URL below and put them in the folder "matsun"
URL: https://drive.google.com/open?id=1-Y-KbAu_YVz7tXbbYHztFyDXOwrLrcWx

@author: afsar
"""

from eszsl import *
import os
import scipy.io
import numpy as np
ddir = "./matsun"
kf = os.path.join(ddir,"kernel.mat")
expidxf = os.path.join(ddir,"experimentIndices.mat")
sf = os.path.join(ddir,"attrClasses.mat")

expidx = scipy.io.loadmat(expidxf)
tridx = expidx['trainInstancesIndices'].flatten()-1
Ytr = expidx['trainInstancesLabels'].flatten()-1
Citr = expidx['trainClassesIndices'].flatten()-1
ttidx = expidx['testInstancesIndices'].flatten()-1
Ytt = expidx['testInstancesLabels'].flatten()-1
Citt = expidx['testClassesIndices'].flatten()-1


S = scipy.io.loadmat(sf)['attrClasses']
Str = S[Citr,:]
Stt = S[Citt,:]

K = scipy.io.loadmat(kf)['K']
Ktt = K[ttidx][:,tridx]
K = K[tridx][:,tridx]

Y = -0.0*np.ones((K.shape[0],len(set(Ytr))))
for i,k in enumerate(Ytr):
    Y[i,k]=1.0

lambdap = 1e-2
sigmap = 1e1

clf = ESZSL(sigmap = sigmap, lambdap = lambdap, kernel = 'precomputed')
clf.fit(K,Y,Str)
Z = clf.predict(Ktt,S=Stt)
print(np.mean(Z==Ytt))
