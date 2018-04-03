# -*- coding: utf-8 -*-
"""
__title__ = ""
__author__ = "Sol Young"
__mtime__ = "2018/4/2"
bug fuck off!!!
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from readData import readData
from readData import UVE
from readData import plsRegressAnalysis
from readData import useLasso
from readData import useElasticNet



from FPTreeAlgorithm import *
from RReliefF import *
from heuristic_algorithm import GeneAlgorithm
from ACOalgorithm import acoAlgorithm
from SAalgorithm import SA
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
SACase = SA(xTrain,yTrain)
globalIndivalSA, currentFitnessTraceSA, globalFitnessTraceSA = SACase.SAAlgorithm()
print(globalFitnessTraceSA)

#GA
GACO = GeneAlgorithm(xTrain, yTrain,idv=idv,Chromo=initRandomState)
globalIndivalGA, currentFitnessTraceGA, globalFitnessTraceGA = GACO.GAAlgorithm()
print(globalFitnessTraceGA)

#ACO
ACOcase = acoAlgorithm(xTrain, yTrain)
globalIndivalACO, currentFitnessTraceACO, globalFitnessTraceACO = ACOcase.ACOAlgorithm()
print(globalFitnessTraceACO)

'''
Embedded
'''
#LASSO
yPredictLasso, ceofLasso = useLasso(xTest, yTest, xTrain, yTrain)

#EN
yPredictEN,coefEN = useElasticNet(xTest, yTest, xTrain, yTrain)
'''
my algorithm
'''
#FP-Tree
orthArray = generateOrthArray(features)
orthArray = orthArray[:,0:features]
fitnessSave = np.array([])
lvSave = np.array([])
transSave = []
for ii in range(1,orthArray.shape[0]):
    selectionPlan = orthArray[ii,:]
    selectedIndex = np.where(selectionPlan == 1)[0]
    xSelected = xTrain[:, selectedIndex]
    # 程序设定不输入测试集时只输出rmsecv作为fitness
    rmsecvTemp, lvTemp = plsRegressAnalysis(xSelected, yTrain)
    transSave.append(selectedIndex)
    np.append(fitnessSave, rmsecvTemp)
    np.append(lvSave, lvTemp)
#保存数据
saveData(transSave=transSave,fitnessSave=fitnessSave,lvSave=lvSave)
rmsecvMed = np.median(fitnessSave)
goodPlanIndex = np.where(fitnessSave<=rmsecvMed)[0]
goodTrans = [transSave[index] for index in goodPlanIndex]
testTree, headerTable = createTree(goodTrans, minSup=0.2)
branchs = []
ergodicTreeBranch(testTree,branchs)
bestBranch = []
maxSumCount = 0
for branch in branchs:
    sumCount = 0
    branchItem = []
    for item in branch:
        sumCount += item.count
        branchItem.append(item.name)
    if sumCount>maxSumCount:
        maxSumCount = sumCount
        bestBranch = branchItem

