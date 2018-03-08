# -*- coding: utf-8 -*-
"""
__title__ = ""
__author__ = "Sol Young"
__mtime__ = "2018/3/8"
bug fuck off!!!
"""

import numpy as np
from sklearn import preprocessing

def createData():
    data = np.array([[3, -1.5, 2, -5.4], [0, 4, -0.3, 2.1], [1, 3.3,-1.9, -4.3]])
    return data

def processingData(data):
    dataScaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    dataScaled = dataScaler.fit_transform(data)
    print("Min max scaled data =", dataScaled)
    return dataScaled