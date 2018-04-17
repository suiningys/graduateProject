# -*- coding: utf-8 -*-
"""
__title__ = ""
__author__ = "Sol Young"
__mtime__ = "2018/2/26"
bug fuck off!!!
"""
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

from sklearn import linear_model
from sklearn import cross_decomposition
from sklearn import model_selection
from sklearn import metrics as sm

def readData(dataPath='./data/COCO2CH4.xlsx'):
    # dataPath = './data/COCO2CH4.xlsx'
    d = pd.read_excel(dataPath)
    lines = d.shape[0]
    CO = d['CO'].as_matrix().reshape(lines,1)
    CH4 = d['CH4'].as_matrix().reshape(lines,1)
    CO2 = d['CO2'].as_matrix().reshape(lines,1)
    specData = d.iloc[:,5:].as_matrix()
    return CO,CO2,CH4,specData

def saveData(**kwargs):
    params = {}
    params.update(kwargs)
    for key in params.keys():
        savePath = './saveData/'+key
        np.save(savePath,params[key])

def loadData(**kwargs):
    Dict = {}
    for key in kwargs:
        savePath = './saveData/' + key
        Dict[key] = np.load(savePath)
    return Dict

def PLS(xTest, yTest, xTrain, yTrain, nComponents):
    plsModel = cross_decomposition.PLSRegression(n_components=nComponents)
    plsModel.fit(xTrain,yTrain)
    coef = plsModel.coef_
    yPredict = plsModel.predict(xTest)
    return yPredict, plsModel

def useRidgeRegression(xTest, yTest, xTrain, yTrain):
    ridgeModel = linear_model.Ridge(alpha=0.01, fit_intercept=True,max_iter=10000)
    ridgeModel.fit(xTrain,yTrain)
    ceof = ridgeModel.coef_
    yPredict = ridgeModel.predict(xTest)
    return yPredict, ridgeModel

def useLasso(xTest, yTest, xTrain, yTrain):
    lassoModel = linear_model.Lasso(alpha=1.0, fit_intercept=True, max_iter=10000)
    lassoModel.fit(xTrain,yTrain)
    ceof = lassoModel.coef_
    yPredict = lassoModel.predict(xTest)
    return yPredict, lassoModel

def useElasticNet(xTest, yTest, xTrain, yTrain,alpha=1,l1_ratio=0.5):
    enModel = linear_model.ElasticNet(l1_ratio=l1_ratio)
    enModel.set_params(alpha=alpha)
    enModel.fit(xTrain,yTrain)
    coef = enModel.coef_
    yPredict = enModel.predict(xTest)
    return yPredict,enModel

def useElasticNetCV(xTest, yTest, xTrain, yTrain):
    enModel = linear_model.ElasticNetCV(cv=10,random_state=0)
    # enModel.set_params(alpha=0.1)
    enModel.fit(xTrain,yTrain)
    coef = enModel.coef_
    yPredict = enModel.predict(xTest)
    return yPredict,enModel

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
                # yPredictTempTrue = scalerY.inverse_transform(yPredictTemp)
                # yTestTempTrue = scalerY.inverse_transform(yTestTemp)
                residual = yPredictTemp - yTestTemp
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

def calMetric(yPredict, yTest):
    #residual = yPredict - yTest
    #MSE = np.dot(residual.T,residual)/residual.shape[0]
    MSE = sm.mean_squared_error(yTest,yPredict)
    R2 = sm.r2_score(yTest,yPredict)

def plsRegressAnalysis(xTrain, yTrain, xTest = None, yTest = None):
    kf = model_selection.KFold(n_splits=5,random_state=10)
    trans, features = xTrain.shape
    lvMax = int(min(trans, features)/3)
    lvBest = 0
    rmsecvBest = np.inf
    for lvTemp in range(1,lvMax+1):
        squareArray = np.array([[]])
        for train, test in kf.split(xTrain):
            xTrainTemp = xTrain[train, :]
            yTrainTemp = yTrain[train]
            xTestTemp = xTrain[test, :]
            yTestTemp = yTrain[test]
            yPredictTemp, coefTemp = PLS(xTestTemp, yTestTemp, xTrainTemp, yTrainTemp, lvTemp)
            residual = yPredictTemp - yTestTemp
            square = np.dot(residual.T, residual)
            squareArray = np.append(squareArray, square)
            # squareArray.append(square)
        RMSECV = np.sqrt(np.sum(squareArray) / xTrain.shape[0])
        if RMSECV<rmsecvBest:
            rmsecvBest = RMSECV
            lvBest = lvTemp
    if xTest is None:
        return rmsecvBest, lvBest
    else:
        plsModel = cross_decomposition.PLSRegression(n_components=lvBest)
        plsModel.fit(xTrain, yTrain)
        coef = plsModel.coef_
        yPredict = plsModel.predict(xTest)
        yTrainPredict = plsModel.predict(xTrain)
        R2 = sm.r2_score(yTrain,yTrainPredict)
        MSE = sm.mean_squared_error(yTest,yPredict)
        R2P = sm.r2_score(yTest, yPredict)
        return yPredict, R2, rmsecvBest, R2P,MSE, lvBest



def UVECV(xTest, yTest, uveLv):
    # kf = model_selection.KFold(n_splits=5,random_state=10)
    loo = model_selection.LeaveOneOut()
    squareArray = np.array([[]])
    coefs = np.array([[]])
    for train, test in loo.split(xTest):
        xTrainTemp = xTest[train, :]
        yTrainTemp = yTest[train]
        xTestTemp = xTest[test, :]
        yTestTemp = yTest[test]
        yPredictTemp, plsModes = PLS(xTestTemp,yTestTemp,xTrainTemp,yTrainTemp,uveLv)
        coefTemp = plsModes.coef_.T
        if coefs.shape[1]==0:
            coefs = coefTemp
        else:
            coefs = np.append(coefs,coefTemp,axis=0)
        residual = yPredictTemp - yTestTemp
        square = np.dot(residual.T,residual)
        squareArray = np.append(squareArray,square)
        #squareArray.append(square)
    RMSECV = np.sqrt(np.sum(squareArray)/xTest.shape[0])
    return RMSECV,coefs

def UVE(xTest, yTest):
    trans, features = xTest.shape
    uveLvMax = round(min(trans,features)/3)
    uveLvStd = uveLvMax
    rmsecvInit, coefsInit = UVECV(xTest,yTest,uveLvMax)
    rmsecvStd = rmsecvInit
    uveLvSave = [uveLvMax]
    uveRand = np.random.random([trans,features])*1e-6
    uveMartix = np.append(xTest,uveRand,axis=1)#合并矩阵

    rmsecvTemp, coefsTemp = UVECV(uveMartix, yTest, uveLvSave[0])
    Ctemp = np.mean(coefsTemp, axis=0) / np.std(coefsTemp, axis=0)
    maxRandC = np.max(np.abs(Ctemp[features:]))
    newIndex = np.where(Ctemp > maxRandC)[0]
    finalIndex = newIndex
    uveLvStd = finalIndex.shape[1]

    uveIterStart = 1#停止标志
    uveIterCount = 0#迭代计数

    while uveIterStart!=0:
        rmsecvTemp, coefsTemp = UVECV(uveMartix,yTest,uveLvSave[uveIterCount])
        Ctemp = np.mean(coefsTemp,axis=0)/np.std(coefsTemp,axis=0)
        maxRandC = np.max(np.abs(Ctemp[features:]))
        newIndex = np.where(Ctemp>maxRandC)[0]
        xTestSelected = xTest[:,newIndex]
        nTrans, nFeatures = xTestSelected.shape
        if nFeatures>=uveLvSave[uveIterCount]:
            rmsecvTemp,coefsTemp = UVECV(xTestSelected,yTest,uveLvSave[uveIterCount])
            if rmsecvTemp>=rmsecvStd:
                uveIterStart=0
            else:
                rmsecvStd = rmsecvTemp
                uveLvStd = uveLvSave[uveIterCount]
                finalIndex = newIndex
                uveLvSave.append(nFeatures-1)
                uveIterCount += 1
        else:
            uveIterCount+=1
            uveLvSave.append(nFeatures)
    finalRes = xTest[:,finalIndex]
    return rmsecvStd, uveLvStd, finalRes, finalIndex

if __name__=="__main__":
    CO, CO2, CH4, specData = readData()
    xData = specData
    yData = CO
    xTrain, xTest, yTrain, yTest = \
        model_selection.train_test_split(xData,yData,test_size=0.25, random_state=42)
