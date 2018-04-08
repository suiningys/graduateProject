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
            train_test_split(specData, CO, test_size=0.2, random_state=42)

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
th = np.percentile(w, 80)#计算分位数
RFselectedIndex = np.where(w>=th)[0]
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
yPredictLasso, LassoModel = useLasso(xTestOrigin, yTestOrigin, xTrainOrigin, yTrainOrigin)
R2Lasso = sm.r2_score(yPredictLasso,yTestOrigin)
RMSEPlasso = np.sqrt(sm.mean_squared_error(yPredictLasso,yTestOrigin))
#EN
print("run EN")
def ElasticNetCVOwn(xTest, yTest, xTrain, yTrain, plot=False):
    kf = model_selection.KFold(n_splits=5, random_state=10)
    trans, features = xTrain.shape
    enModel = linear_model.ElasticNet()
    rmsecvBest = np.inf
    bestAlpha = 0
    bestL1  = 1
    l1Cand = np.arange(0,1,0.1)
    alpheCand = np.arange(0,1,0.1)
    rmsecvSave = np.zeros([l1Cand.shape[0],alpheCand.shape[0]])
    for ii in range(l1Cand.shape[0]):
        l1 = l1Cand[ii]
        for jj in range(alpheCand.shape[0]):
            alpha = alpheCand[jj]
            squareArray = np.array([[]])
            for train, test in kf.split(xTrain):
                xTrainTemp = xTrain[train, :]
                yTrainTemp = yTrain[train]
                xTestTemp = xTrain[test, :]
                yTestTemp = yTrain[test]
                enModel.set_params(alpha=alpha,l1_ratio=l1)
                enModel.fit(xTrainTemp,yTrainTemp)
                yPredictTemp = enModel.predict(xTestTemp)
                yPredictTempTrue = scalerY.inverse_transform(yPredictTemp)
                yTestTempTrue = scalerY.inverse_transform(yTestTemp)
                residual = yPredictTempTrue - yTestTempTrue
                square = np.dot(residual.T, residual)
                squareArray = np.append(squareArray, square)
                # squareArray.append(square)
            RMSECV = np.sqrt(np.sum(squareArray) / xTrain.shape[0])
            rmsecvSave[ii][jj] = RMSECV
            if RMSECV < rmsecvBest:
                rmsecvBest = RMSECV
                bestAlpha = alpha
                bestL1 = l1
    if(plot):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = Axes3D(fig)
        X,Y = np.meshgrid(l1Cand, alpheCand)
        ax.plot_surface(X,Y,rmsecvSave)
        ax.set_xlabel(r'L1')
        ax.set_ylabel(r'$\alpha$')
        ax.set_zlabel(r'RMSECV')
        plt.show()

    return bestAlpha, bestL1

# bestAlpha, bestL1 = ElasticNetCVOwn(xTest, yTest, xTrain, yTrain, plot=True)
# yPredictEN0, ENModel0 = useElasticNet(xTest, yTest, xTrain, yTrain,l1_ratio=bestL1,alpha=bestAlpha)
yPredictEN,ENModel = useElasticNetCV(xTest, yTest, xTrain, yTrain)
R2EN = sm.r2_score(yPredictEN,yTest)
RMSEPEN = np.sqrt(sm.mean_squared_error(scalerY.inverse_transform(yPredictEN),yTestOrigin))

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

#rmsecvMed = np.median(fitnessSave)
rmsecvMed = np.percentile(fitnessSave,60)
goodPlanIndex = np.where(fitnessSave<=rmsecvMed)[0]
goodTrans = [transSave[index] for index in goodPlanIndex]
print("generate FP-Tree")

def findBestMinsup():
    matrix = np.zeros([len(goodTrans),xTrain.shape[1]])
    for ii in range(len(goodTrans)):
        for jj in goodTrans[ii]:
            matrix[ii][jj] = 1
    varSup = matrix.sum(axis=0)/len(goodTrans)
    supMin = varSup.min()
    supMax = varSup.max()
    rmsecvBestFP = np.inf
    globalBestBranch = []
    rmsecvSave = []
    for minSup in np.arange(supMin,supMax,0.01):
        testTree, headerTable = createTree(goodTrans, minSup=minSup)
        branchs = []
        ergodicTreeBranch(testTree, branchs)
        bestBranch = []
        maxSumCount = 0
        for branch in branchs:
            sumCount = 0
            branchItem = []
            for item in branch:
                sumCount += item.count
                branchItem.append(item.name)
            if sumCount > maxSumCount:
                maxSumCount = sumCount
                bestBranch = branchItem
        xSelected = xTrainOrigin[:, bestBranch]
        # 程序设定不输入测试集时只输出rmsecv作为fitness
        rmsecvTemp, lvTemp = plsRegressAnalysis(xSelected, scalerY.inverse_transform(yTrain))
        rmsecvSave.append(rmsecvTemp)
        print(rmsecvTemp)
        print(len(bestBranch))
        if rmsecvTemp< rmsecvBestFP:
            rmsecvBestFP = rmsecvTemp
            bestMinsup = minSup
            globalBestBranch = bestBranch
    
    return bestMinsup, globalBestBranch, rmsecvSave, np.arange(supMin,supMax,0.01)
bestMinSup, bestBranch, rmsecvSave, xnum = findBestMinsup()
#draw picture
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(xnum, rmsecvSave,marker='.',c='k')
ax.set_xlabel("minSup")
ax.set_ylabel("RMSECV")
'''
模型比较
'''
#预测
def predictUsingIdv(idv,xTest, yTest, xTrain, yTrain):
    selected = np.where(idv.chromo==1)[0]
    yPredict, coef = PLS(xTest[:,selected], yTest, xTrain[:,selected], yTrain, idv.lv)
    return yPredict

yPredictPLS, R2PLS, rmsecvBestPLS, R2PPLS,MSEPLS, lvBestPLS = plsRegressAnalysis(xTrain, yTrain, xTest, yTest)
yPredictRF, R2RF, rmsecvBestRF, R2PRF,MSERF, lvBestRF = plsRegressAnalysis(xTrain[:,RFselectedIndex], yTrain, xTest[:,RFselectedIndex], yTest)
yPredictUVE, ceofs = PLS(xTest[:,uveSelectedIndex], yTest, xTrain[:,uveSelectedIndex], yTrain, uveLvStd)
yPredictSA = predictUsingIdv(globalIndivalSA,xTest, yTest, xTrain, yTrain)
yPredictGA = predictUsingIdv(globalIndivalGA,xTest, yTest, xTrain, yTrain)
yPredictACO = predictUsingIdv(globalIndivalACO,xTest, yTest, xTrain, yTrain)
yPredictFPTree, ceofs = PLS(xTest[:,bestBranch], yTest, xTrain[:,bestBranch], yTrain,lvTemp)

'''
计算模型评价参数
'''
def modelMerit(SelectedIndex):
    xTrainSelected = xTrain[:,SelectedIndex]
    xTestSelected = xTest[:,SelectedIndex]
    kf = model_selection.KFold(n_splits=5, random_state=10)
    trans, features = xTrainSelected.shape
    lvMax = int(min(trans, features) / 3)
    lvBest = 0
    rmsecvBest = np.inf
    for lvTemp in range(1, lvMax + 1):
        squareArray = np.array([[]])
        for train, test in kf.split(xTrainSelected):
            xTrainTemp = xTrainSelected[train, :]
            yTrainTemp = yTrain[train]
            xTestTemp = xTrainSelected[test, :]
            yTestTemp = yTrain[test]
            yPredictTemp, coefTemp = PLS(xTestTemp, yTestTemp, xTrainTemp, yTrainTemp, lvTemp)
            yPredictTempTrue = scalerY.inverse_transform(yPredictTemp)
            yTestTempTrue = scalerY.inverse_transform(yTestTemp)
            residual = yPredictTempTrue - yTestTempTrue
            square = np.dot(residual.T, residual)
            squareArray = np.append(squareArray, square)
            # squareArray.append(square)
        RMSECV = np.sqrt(np.sum(squareArray) / xTrainSelected.shape[0])
        if RMSECV < rmsecvBest:
            rmsecvBest = RMSECV
            lvBest = lvTemp
    
    plsModel = cross_decomposition.PLSRegression(n_components=lvBest)
    plsModel.fit(xTrainSelected, yTrain)

    yPredict = plsModel.predict(xTestSelected)
    yTrainPredict = plsModel.predict(xTrainSelected)
    yTrainPredictTrue = scalerY.inverse_transform(yTrainPredict)
    yPredictTrue = scalerY.inverse_transform(yPredict)

    MEST = sm.mean_squared_error(yTrainOrigin,yTrainPredictTrue)
    RMSET = np.sqrt(MEST)
    R2T = sm.r2_score(yTrainOrigin, yTrainPredictTrue)
    MSEP = sm.mean_squared_error(yTestOrigin, yPredictTrue)
    RMSEP = np.sqrt(MSEP)
    R2P = sm.r2_score(yTestOrigin, yPredictTrue)
    #计算交叉验证误差
    yPredictCV = np.array([[]])
    yTrueCV = np.array([[]])
    for train, test in kf.split(xTrainSelected):
        xTrainTemp = xTrainSelected[train, :]
        yTrainTemp = yTrain[train]
        xTestTemp = xTrainSelected[test, :]
        yTestTemp = yTrain[test]
        yPredictTemp, coefTemp = PLS(xTestTemp, yTestTemp, xTrainTemp, yTrainTemp, lvBest)
        yPredictTempTrue = scalerY.inverse_transform(yPredictTemp)
        yTestTempTrue = scalerY.inverse_transform(yTestTemp)
        yPredictCV = np.append(yPredictCV,yPredictTempTrue)
        yTrueCV = np.append(yTrueCV,yTestTempTrue)
        residual = yPredictTempTrue - yTestTempTrue
        square = np.dot(residual.T, residual)
        squareArray = np.append(squareArray, square)
        # squareArray.append(square)
    RMSECV = np.sqrt(np.sum(squareArray) / xTrainSelected.shape[0]) 
    R2CV = sm.r2_score(yPredictCV, yTrueCV)
    
    CR = 1 - len(SelectedIndex)/xTrain.shape[1]
    return rmsecvBest, RMSET, RMSEP, R2T, R2P, R2CV, CR


#计算
MeritPLSAll = modelMerit(list(range(xTrain.shape[1])))
MeritRF = modelMerit(RFselectedIndex)
MeritUVE = modelMerit(uveSelectedIndex)
SAselectedIndex = np.where(globalIndivalSA.chromo==1)[0]
MeritSA = modelMerit(SAselectedIndex)
GAselectedIndex = np.where(globalIndivalGA.chromo==1)[0]
MeritGA = modelMerit(GAselectedIndex)
ACOselectedIndex = np.where(globalIndivalACO.chromo==1)[0]
MeritACO = modelMerit(ACOselectedIndex)
MeritFP = modelMerit(bestBranch)


#draw pic
plt.rcParams['font.sans-serif']='NSimSun,Times New Roman'
plt.rcParams['axes.unicode_minus']=False

minLim = min(scalerY.inverse_transform(np.array([[min(yTestOrigin.min(), yPredictRF.min(),\
             yPredictUVE.min(), yPredictSA.min(),\
             yPredictGA.min(), yPredictACO.min(),\
             yPredictEN.min())]]))[0][0],yPredictLasso.min())
maxLim = max(scalerY.inverse_transform(np.array([[min(yTestOrigin.max(), yPredictRF.max(),\
             yPredictUVE.max(), yPredictSA.max(),\
             yPredictGA.max(), yPredictACO.max(),\
             yPredictEN.max())]]))[0][0],yPredictLasso.max())

def drawRes(yPredictScale, ax):
    yTure = scalerY.inverse_transform(yTest)
    yPredict = scalerY.inverse_transform(yPredictScale)
    # minLim = min(yTure.min(), yPredict.min())
    # maxLim = max(yTure.max(), yPredict.max())
    plt.scatter(yTure, yPredict, c='k', marker=".", alpha=0.5)
    plt.plot([minLim * 1.3, maxLim * 1.01], [minLim * 1.3, maxLim * 1.01], c='k')
    plt.xlim(minLim * 1.3, maxLim * 1.01)
    plt.ylim(minLim * 1.3, maxLim * 1.01)
    ax.set_xlabel("测量值")
    ax.set_ylabel("预测值")


fig = plt.figure()
ax0 = fig.add_subplot(2,2,1)
yPredictTemp = yPredictPLS
drawRes(yPredictTemp, ax0)
ax0.set_title('(a)')

ax1 = fig.add_subplot(2,2,2)
yPredictTemp = yPredictRF
drawRes(yPredictTemp, ax1)
ax1.set_title('(b)')


ax2 = fig.add_subplot(2,2,3)
yPredictTemp = yPredictUVE
drawRes(yPredictTemp, ax2)
ax2.set_title('(c)')

ax3 = fig.add_subplot(2,2,4)
yPredictTemp = yPredictSA
drawRes(yPredictTemp, ax3)
ax3.set_title('(d)')
plt.tight_layout()#自动调整图片，使文字不重叠

fig = plt.figure()
ax4 = fig.add_subplot(2,2,1)
yPredictTemp = yPredictGA
drawRes(yPredictTemp, ax4)
ax4.set_title('(e)')

#fig = plt.figure()
ax5 = fig.add_subplot(2,2,2)
yPredictTemp = yPredictACO
drawRes(yPredictTemp, ax5)
ax5.set_title('(f)')

ax6 = fig.add_subplot(2,2,3)
yPredictTemp = yPredictLasso
# minLim = min(yTestOrigin.min(),yPredictTemp.min())
# maxLim = max(yTestOrigin.max(),yPredictTemp.max())
plt.scatter(yTestOrigin,yPredictTemp,c='k',marker=".",alpha=0.5)
plt.plot([minLim*1.3,maxLim*1.01],[minLim*1.3,maxLim*1.01],c='k')
plt.xlim(minLim*1.3, maxLim*1.01)
plt.ylim(minLim*1.3, maxLim*1.01)
ax6.set_xlabel("测量值")
ax6.set_ylabel("预测值")
ax6.set_title('(g)')

ax7 = fig.add_subplot(2,2,4)
yPredictTemp = yPredictEN
drawRes(yPredictTemp, ax7)
ax7.set_title('(h)')
plt.tight_layout()#自动调整图片，使文字不重叠

fig = plt.figure()
ax8 = fig.add_subplot(1,1,1)
yPredictTemp = yPredictFPTree
drawRes(yPredictTemp, ax8)

plt.show()

