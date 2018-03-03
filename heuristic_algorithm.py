# -*- coding: utf-8 -*-
"""
__title__ = ""
__author__ = "Sol Young"
__mtime__ = "2018/3/1"
bug fuck off!!!
"""
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

from sklearn import linear_model
from sklearn import cross_decomposition
from sklearn import model_selection
from sklearn import metrics as sm

from readData import readData
from readData import plsRegressAnalysis

class individual(object):
    def __init__(self,chromo = None,fitness = None,lv = None):
        self.chromo = chromo
        self.fitness = fitness
        self.lv = lv

class GeneAlgorithm(object):
    def __init__(self,xTrain,yTrain,
                 idv = 10, crossProb = 0.6, mutationProb = 0.01,mutBits = 1,
                 Chromo = None, maxIter = 100, fitnessFunction = plsRegressAnalysis):
        trans, features = xTrain.shape
        self.xData = xTrain
        self.yData = yTrain
        self.idv = idv#个体个数
        self.crossProb = crossProb#交叉概率
        self.mutationProb = mutationProb#变异概率
        self.chromeBits = features#基因位点
        self.mutBits = mutBits#每次变异的位数
        self.fitnessFunction = fitnessFunction
        if Chromo is None:#可以接受外部传入的初始化基因
            self.Chromo = np.random.randint(0,2,size = [self.idv,self.chromeBits])#初始化基因
        else:
            self.Chromo = Chromo
        #全局最优变量
        self.globalBestFitness = np.inf
        self.globalBestChromo = np.random.random([1,self.chromeBits])
        self.globalBestLv = features

        #构造种群
        self.population = []
        for ii in range(self.idv):
            self.population.append(individual())
        #初始化种群
        for indiv, ii in zip(self.population,list(range(self.idv))):
            indiv.chromo = self.Chromo[ii,:]
            indiv.fitness = np.inf
            indiv.lv = indiv.chromo.shape[0]
        #局部最优
        self.currentBestFitness = np.inf
        self.currentBestChromo = None

        self.maxIter = maxIter#最大迭代次数

    def mutation(self,Chromo):
        # mutProp = np.random.random()
        # if mutProp<self.mutationProb:
        mutBitPos = np.random.randint(0,self.chromeBits)
        Chromo[mutBitPos] ^= 1#取反

    def crossOver(self,Chrome1,Chrome2):
        crossBitPos = np.random.randint(0,self.chromeBits)
        Chrome1Tail = Chrome1[crossBitPos:].copy()
        Chrome2Tail = Chrome2[crossBitPos:].copy()
        Chrome1[crossBitPos:] = Chrome2Tail
        Chrome2[crossBitPos:] = Chrome1Tail

    def calPopulationFitness(self):
        fitnessSave = []
        for idv in self.population:
            #np.where()返回的值是一个元组，不是np.ndarray
            selectedIndex = np.where(idv.chromo==1)[0]
            xSelected = self.xData[:,selectedIndex]
            #程序设定不输入测试集时只输出rmsecv作为fitness
            rmsecvTemp = self.fitnessFunction(xSelected, self.yData)
            idv.fitness = rmsecvTemp
            fitnessSave.append(rmsecvTemp)
        return fitnessSave
