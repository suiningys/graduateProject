# -*- coding: utf-8 -*-
"""
__title__ = ""
__author__ = "Sol Young"
__mtime__ = "2018/3/8"
bug fuck off!!!

Robnik-Sikonja M, Kononenko I.
An adaptation of Relief for attribute estimation in regression[J].
šikonja, 1997.
"""
import numpy as np
from sklearn import preprocessing

def distance(instancei,instancej):
    diff = instancei - instancej
    return np.sqrt(np.dot(diff, diff.T))

def diff(A,I1,I2):
    return np.abs(I1[A] - I2[A])

def RReliefF(xData, yData, m=250, k=30, sigma = 0.01):
    dataScaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    xDataScaled = dataScaler.fit_transform(xData)
    yDataScaled = dataScaler.fit_transform(yData)
    # xDataScaled = xData
    # yDataScaled = yData
    instances, features = xDataScaled.shape
    if k > instances:
        k = instances
    Ndc = 0 #np.zeros([1,features])
    NdA = np.zeros(features)
    NdCdA = np.zeros(features)
    W = np.zeros(features)
    for ii in range(m):
        selectInestance = np.random.randint(0,instances)
        distanceDict = {}
        for ii in range(instances):
            distanceDict.update({ii:distance(xDataScaled[selectInestance,:],
                                             xDataScaled[ii,:])})
        sortedIndex =[v[0] for v in sorted(distanceDict.items(),
                             key=lambda p:p[1])]
        kNearestInstance = [sortedIndex[ii] for ii in range(1,k+1)]
        d1 = {}
        # sigma = 0.01
        for ins in kNearestInstance:
            d1.update({ins:np.exp(-(kNearestInstance.index(ins)/sigma)**2)})
        d = {}
        for ins in kNearestInstance:
            d.update({ins:d1[ins]/sum(d1.values())})
        for ins in kNearestInstance:
            Ndc += np.abs(yDataScaled[selectInestance] - yDataScaled[ins])*d[ins]
            for ii in range(features):
                NdA[ii] = NdA[ii] + diff(ii,xDataScaled[selectInestance,:],
                                         xDataScaled[ins,:])*d[ins]
                NdCdA[ii] = NdCdA[ii] + np.abs(yDataScaled[selectInestance] - yDataScaled[ins])*d[ins]\
                            *diff(ii,xDataScaled[selectInestance,:],xDataScaled[ins,:])*d[ins]
    for ii in range(features):
        W[ii] = NdCdA[ii]/Ndc - (NdA[ii] - NdCdA[ii])/(m - Ndc)
    return W
