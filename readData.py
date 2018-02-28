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

def readData():
    dataPath = './COCO2CH4.xlsx'
    d = pd.read_excel(dataPath)
    CO = d['CO']
    CH4 = d['CH4']
    CO2 = d['CO2']
    specData = d.iloc[:,5:]
    return CO,CO2,CH4,specData

def PLS(xTest, yTest, xTrain, yTrain, nComponents):
    plsModel = cross_decomposition.PLSRegression(n_components=nComponents)
    plsModel.fit(xTrain,yTrain)
    coef = plsModel.coef_
    yPredict = plsModel.predict(xTest)
    return yPredict, coef

def useRidgeRegression(xTest, yTest, xTrain, yTrain):
    ridgeModel = linear_model.Ridge(alpha=0.01, fit_intercept=True,max_iter=10000)
    ridgeModel.fit(xTrain,yTrain)
    ceof = ridgeModel.coef_
    yPredict = ridgeModel.predict(xTest)
    return yPredict, ceof

def useLasso(xTest, yTest, xTrain, yTrain):
    lassoModel = linear_model.Lasso(alpha=1.0, fit_intercept=True, max_iter=10000)
    lassoModel.fit(xTrain,yTrain)
    ceof = lassoModel.coef_
    yPredict = lassoModel.predict(xTest)
    return yPredict, ceof

def useElasticNet(xTest, yTest, xTrain, yTrain):
    enModel = linear_model.ElasticNet(l1_ratio=0.7)
    enModel.set_params(alpha=0.1)
    enModel.fit(xTrain,yTrain)
    coef = enModel.coef_
    yPredict = enModel.predict(xTest)
    return yPredict,coef

if __name__=="__main__":
    CO, CO2, CH4, specData = readData()
    xData = specData
    yData = CO
    xTrain, xTest, yTrain, yTest = \
        model_selection.train_test_split(xData,yData,test_size=0.25, random_state=42)
