# -*- coding: utf-8 -*-
"""
__title__ = ""
__author__ = "Sol Young"
__mtime__ = "2018/3/8"
bug fuck off!!!
"""

import numpy as np

from readData import plsRegressAnalysis
from geneticAlgorithm import individual

class acoAlgorithm(object):
    def __init__(self, xTrain, yTrain,idv=10,
                 PathAll=None, maxIter=100,
                 fitnessFunction=plsRegressAnalysis):
        trans, features = xTrain.shape
        self.features = features
        self.xData = xTrain
        self.yData = yTrain
        self.idv = idv  # 个体个数
        self.pathNode = features  # 路径位点
        #蚁群算法参数
        self.Q = 1.8 #过大会造成停滞
        self.Rho = 0.9 #决定搜索与利用间的平衡
        self.q0 = 0.1
        self.p = 3 #每p次用全局最优进行信息素更新
        self.lamda = 1.3

        self.fitnessFunction = fitnessFunction
        if PathAll is None:#可以接受外部传入的初始化基因
            self.PathAll = np.random.randint(0,2,size = [self.idv,self.pathNode])#初始化基因
        else:
            self.PathAll = PathAll.copy()

        self.Tau = np.ones([2,features])
        self.Prob = np.zeros([features,1])


        # 全局最优变量
        self.globalBestFitness = np.inf
        self.globalBestPath = np.random.random([1, self.pathNode])
        self.globalBestLv = features
        self.globalIndival = individual(chromo=self.globalBestPath,
                                        fitness=self.globalBestFitness,
                                        lv=self.globalBestLv)

        # 构造种群
        self.population = []
        for ii in range(self.idv):
            self.population.append(individual())
        # 初始化种群
        for indiv, ii in zip(self.population, list(range(self.idv))):
            indiv.chromo = self.PathAll[ii, :].copy()
            indiv.fitness = np.inf
            indiv.lv = indiv.chromo.shape[0]
        # 局部最优
        self.currentBestFitness = np.inf
        self.currentBestPath = None
        self.maxIter = maxIter  # 最大迭代次数

        self.fitnessDict = {}

    def calPopulationFitness(self):
        fitnessSave = []
        for idv in self.population:
            #np.where()返回的值是一个元组,第一项为行坐标，第二项为列坐标，不是np.ndarray
            selectedIndex = np.where(idv.chromo==1)[0]
            selectedStr = ''.join(str(ii) for ii in idv.chromo)
            if selectedStr in self.fitnessDict.keys():
                rmsecvTemp, lvTemp = self.fitnessDict[selectedStr]
            else:
                xSelected = self.xData[:,selectedIndex]
                #程序设定不输入测试集时只输出rmsecv作为fitness
                rmsecvTemp, lvTemp = self.fitnessFunction(xSelected, self.yData)
                self.fitnessDict[selectedStr] = (rmsecvTemp, lvTemp)
            idv.fitness = rmsecvTemp
            idv.lv = lvTemp
            fitnessSave.append(rmsecvTemp)
        return fitnessSave

    def ACOAlgorithm(self):
        currentFitnessTrace = []
        globalFitnessTrace = []
        for ii in range(self.maxIter):
            # 计算概率矩阵
            for u in range(self.features):
                self.Prob[u] = self.Tau[0,u]/(self.Tau[0,u]+self.Tau[1,u])
            for idv in range(self.idv):
                for u in range(self.features):
                    if np.random.random()<self.q0:
                        if self.Tau[0,u]>self.Tau[1,u]:
                            self.population[idv].chromo[u] = 0
                        else:
                            self.population[idv].chromo[u] = 1
                    else:
                        if np.random.random()<self.Prob[u]:
                            self.population[idv].chromo[u] = 0
                        else:
                            self.population[idv].chromo[u] = 1
            fitnessAll = self.calPopulationFitness()
