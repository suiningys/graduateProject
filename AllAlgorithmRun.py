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
from sklearn import preprocessing
import matplotlib.pyplot as plt

from readData import readData
from readData import UVE
from readData import plsRegressAnalysis
from readData import PLS
from readData import useLasso
from readData import useElasticNetCV
from readData import useElasticNet



from FPTreeAlgorithm import *
from RReliefF import *
from heuristic_algorithm import *
#from heuristic_algorithm import GeneAlgorithm
#from heuristic_algorithm import predictUsingIdv
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
xTrainOrigin, xTestOrigin, yTrainOrigin, yTestOrigin = \
            train_test_split(specData, CO, test_size=0.25, random_state=42)

'''
数据预处理
'''
#mean removal
scalerX = preprocessing.StandardScaler().fit(xTrainOrigin)
xTrain = scalerX.transform(xTrainOrigin)
xTest = scalerX.transform(xTestOrigin)
scalerY = preprocessing.StandardScaler().fit(yTrainOrigin)
yTrain = scalerY.transform(yTrainOrigin)
yTest = scalerY.transform(yTestOrigin)




'''
fliter algorithm
'''
#RReliefF
print("run RreliefF")
w = RReliefF(xTrain,yTrain)
#UVE
print("run UVE")
rmsecvStd, uveLvStd, finalRes ,uveSelectedIndex= UVE(xTrain,yTrain)
'''
wrapper algorithm
'''
#generate init state
idv = 10
trans, features = xTrain.shape
initRandomState = np.random.randint(0,2,size = [idv,features])#初始化基因

#SA
print("run SA")
SACase = SA(xTrain,yTrain)
globalIndivalSA, currentFitnessTraceSA, globalFitnessTraceSA = SACase.SAAlgorithm()
print(globalFitnessTraceSA)

#GA
print("run GA")
GACO = GeneAlgorithm(xTrain, yTrain,idv=idv,Chromo=initRandomState)
globalIndivalGA, currentFitnessTraceGA, globalFitnessTraceGA = GACO.GAAlgorithm()
print(globalFitnessTraceGA)

#ACO
print("run ACO")
ACOcase = acoAlgorithm(xTrain, yTrain)
globalIndivalACO, currentFitnessTraceACO, globalFitnessTraceACO = ACOcase.ACOAlgorithm()
print(globalFitnessTraceACO)

'''
Embedded
'''
#LASSO
print("run LASSO")
yPredictLasso, ceofLasso = useLasso(xTest, yTest, xTrain, yTrain)

#EN
print("run EN")
yPredictEN,coefEN = useElasticNetCV(xTest, yTest, xTrain, yTrain)
'''
my algorithm
'''
#FP-Tree
print("run FP-TREE")
orthArray = generateOrthArray(features)
orthArray = orthArray[:,0:features]
fitnessSave = np.array([])
lvSave = np.array([])
transSave = []
for ii in range(1,orthArray.shape[0]):
    print("run %d experiment" %(ii))
    selectionPlan = orthArray[ii,:]
    selectedIndex = np.where(selectionPlan == 1)[0]
    xSelected = xTrain[:, selectedIndex]
    # 程序设定不输入测试集时只输出rmsecv作为fitness
    rmsecvTemp, lvTemp = plsRegressAnalysis(xSelected, yTrain)
    transSave.append(selectedIndex)
    fitnessSave = np.append(fitnessSave, rmsecvTemp)
    lvSave = np.append(lvSave, lvTemp)
#保存数据
saveData(transSave=transSave,fitnessSave=fitnessSave,lvSave=lvSave)

rmsecvMed = np.median(fitnessSave)
goodPlanIndex = np.where(fitnessSave<=rmsecvMed)[0]
goodTrans = [transSave[index] for index in goodPlanIndex]
print("generate FP-Tree")
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
xSelected = xTrain[:, bestBranch]
# 程序设定不输入测试集时只输出rmsecv作为fitness
rmsecvTemp, lvTemp = plsRegressAnalysis(xSelected, yTrain)
#预测
def predictUsingIdv(idv,xTest, yTest, xTrain, yTrain):
    selected = np.where(idv.chromo==1)[0]
    yPredict, coef = PLS(xTest[:,selected], yTest, xTrain[:,selected], yTrain, idv.lv)
    return yPredict


yPredictUVE, ceofs = PLS(xTest[:,uveSelectedIndex], yTest, xTrain[:,uveSelectedIndex], yTrain, uveLvStd)
yPredictSA = predictUsingIdv(globalIndivalSA,xTest, yTest, xTrain, yTrain)
yPredictGA = predictUsingIdv(globalIndivalGA,xTest, yTest, xTrain, yTrain)
yPredictACO = predictUsingIdv(globalIndivalACO,xTest, yTest, xTrain, yTrain)
yPredictFPTree, ceofs = PLS(xTest[:,bestBranch], yTest, xTrain[:,bestBranch], yTrain,lvTemp)

#draw pic
plt.rcParams['font.sans-serif']='NSimSun,Times New Roman'
plt.rcParams['axes.unicode_minus']=False

def drawRes(yPredictScale):
    yTure = scalerY.inverse_transform(yTest)
    yPredict = scalerY.inverse_transform(yPredictScale)
    minLim = min(yTure.min(), yPredict.min())
    maxLim = max(yTure.max(), yPredict.max())
    plt.scatter(yTure, yPredict, c='k', alpha=0.5)
    plt.plot([minLim * 1.3, maxLim * 1.1], [minLim * 1.3, maxLim * 1.1], c='k')
    plt.xlim(minLim * 1.3, yTest.max() * 1.1)
    plt.ylim(minLim * 1.3, yTest.max() * 1.1)

fig = plt.figure()
ax1 = fig.add_subplot(4,2,1)
yPredictTemp = yPredictUVE
minLim = min(yTest.min(),yPredictTemp.min())
maxLim = max(yTest.max(),yPredictTemp.max())
plt.scatter(yTest,yPredictTemp,c='k',alpha=0.5)
plt.plot([minLim*1.3,maxLim*1.1],[minLim*1.3,maxLim*1.1],c='k')
plt.xlim(minLim*1.3, yTest.max()*1.1)
plt.ylim(minLim*1.3, yTest.max()*1.1)

ax1 = fig.add_subplot(4,2,2)
yPredictTemp = yPredictSA
minLim = min(yTest.min(),yPredictTemp.min())
maxLim = max(yTest.max(),yPredictTemp.max())
plt.scatter(yTest,yPredictTemp,c='k',alpha=0.5)
plt.plot([minLim*1.3,maxLim*1.1],[minLim*1.3,maxLim*1.1],c='k')
plt.xlim(minLim*1.3, yTest.max()*1.1)
plt.ylim(minLim*1.3, yTest.max()*1.1)

ax1 = fig.add_subplot(4,2,3)
yPredictTemp = yPredictGA
minLim = min(yTest.min(),yPredictTemp.min())
maxLim = max(yTest.max(),yPredictTemp.max())
plt.scatter(yTest,yPredictTemp,c='k',alpha=0.5)
plt.plot([minLim*1.3,maxLim*1.1],[minLim*1.3,maxLim*1.1],c='k')
plt.xlim(minLim*1.3, yTest.max()*1.1)
plt.ylim(minLim*1.3, yTest.max()*1.1)

ax1 = fig.add_subplot(4,2,4)
yPredictTemp = yPredictACO
minLim = min(yTest.min(),yPredictTemp.min())
maxLim = max(yTest.max(),yPredictTemp.max())
plt.scatter(yTest,yPredictTemp,c='k',alpha=0.5)
plt.plot([minLim*1.3,maxLim*1.1],[minLim*1.3,maxLim*1.1],c='k')
plt.xlim(minLim*1.3, yTest.max()*1.1)
plt.ylim(minLim*1.3, yTest.max()*1.1)

ax1 = fig.add_subplot(4,2,5)
yPredictTemp = yPredictLasso
minLim = min(yTest.min(),yPredictTemp.min())
maxLim = max(yTest.max(),yPredictTemp.max())
plt.scatter(yTest,yPredictTemp,c='k',alpha=0.5)
plt.plot([minLim*1.3,maxLim*1.1],[minLim*1.3,maxLim*1.1],c='k')
plt.xlim(minLim*1.3, yTest.max()*1.1)
plt.ylim(minLim*1.3, yTest.max()*1.1)

ax1 = fig.add_subplot(4,2,6)
yPredictTemp = yPredictEN
minLim = min(yTest.min(),yPredictTemp.min())
maxLim = max(yTest.max(),yPredictTemp.max())
plt.scatter(yTest,yPredictTemp,c='k',alpha=0.5)
plt.plot([minLim*1.3,maxLim*1.1],[minLim*1.3,maxLim*1.1],c='k')
plt.xlim(minLim*1.3, yTest.max()*1.1)
plt.ylim(minLim*1.3, yTest.max()*1.1)

ax1 = fig.add_subplot(4,2,7)
yPredictTemp = yPredictFPTree
minLim = min(yTest.min(),yPredictTemp.min())
maxLim = max(yTest.max(),yPredictTemp.max())
plt.scatter(yTest,yPredictTemp,c='k',alpha=0.5)
plt.plot([minLim*1.3,maxLim*1.1],[minLim*1.3,maxLim*1.1],c='k')
plt.xlim(minLim*1.3, yTest.max()*1.1)
plt.ylim(minLim*1.3, yTest.max()*1.1)

plt.show()

