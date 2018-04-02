# -*- coding: utf-8 -*-
"""
__title__ = ""
__author__ = "Sol Young"
__mtime__ = "2018/4/2"
bug fuck off!!!
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from readData import readData
from readData import UVECV
from readData import UVE
from readData import plsRegressAnalysis
from readData import useLasso
from readData import useElasticNet

from heuristic_algorithm import individual
from heuristic_algorithm import GeneAlgorithm

from FPTreeAlgorithm import *
from dataPreprocessing import *
from RReliefF import *

from ACOalgorithm import acoAlgorithm

from generateOrthogonalArrays import *

'''
read data
'''
CO,CO2,CH4,specData = readData()
d = pd.read_excel('./data/NONO2SO2.xlsx')
lines = d.shape[0]
NO = d['NO'].as_matrix().reshape(lines,1)
NO2 = d['NO2'].as_matrix().reshape(lines,1)
SO2 = d['SO2'].as_matrix().reshape(lines,1)
specData2 = d.iloc[:,4:].as_matrix()
'''
split the data set
If int, random_state is the seed used by the random number generator; 
If RandomState instance, random_state is the random number generator; If None, 
the random number generator is the RandomState instance used by np.random.
'''
xTrain, xTest, yTrain, yTest = \
            train_test_split(specData, CO, test_size=0.25, random_state=42)

'''
fliter algorithm
'''
#RReliefF
w = RReliefF(xTrain,yTrain)
#UVE
rmsecvStd, uveLvStd, finalRes = UVE(xTrain,yTrain)
'''
wrapper algorithm
'''
#generate init state
idv = 10
trans, features = xTrain.shape
initRandomState = np.random.randint(0,2,size = [idv,features])#初始化基因

#SA

#GA
GACO = GeneAlgorithm(xTrain, yTrain,idv=idv,Chromo=initRandomState)
globalIndivalGA, currentFitnessTraceGA, globalFitnessTraceGA = GACO.GAAlgorithm()
print(globalFitnessTraceGA)
#ACO
ACOcase = acoAlgorithm(xTrain, yTrain)
globalIndivalACO, currentFitnessTraceACO, globalFitnessTraceACO = ACOcase.ACOAlgorithm()
print(globalFitnessTraceACO)
#LASSO
yPredictLasso, ceofLasso = useLasso(xTest, yTest, xTrain, yTrain)
#EN
yPredictEN,coefEN = useElasticNet(xTest, yTest, xTrain, yTrain)
#FP-Tree
orthArray = generateOrthArray(features)

