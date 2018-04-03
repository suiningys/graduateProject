# -*- coding: utf-8 -*-
"""
__title__ = ""
__author__ = "Sol Young"
__mtime__ = "2018/4/3"
bug fuck off!!!
"""

from geneticAlgorithm import individual
import numpy as np

from readData import plsRegressAnalysis

class SA(object):

    def __init__(self,xTrain,yTrain,
                 idv=10, T=100, mutBits = 10,alpha = 0.98,
                 initState = None, maxIter = 100, fitnessFunction = plsRegressAnalysis):
        trans, features = xTrain.shape
        self.featurens = features
        self.xData = xTrain
        self.yData = yTrain
        self.idv = idv  # 个体个数
        self.fitnessFunction = fitnessFunction
        self.T = T  # 固体的温度
        self.alpha = alpha # 温度下降速度
        if initState is None:
            self.State = np.random.randint(0, 2, size=[self.idv, features])  # 初始化基因
        else:
            self.State = initState.copy()
        # 全局最优变量
        self.globalBestFitness = np.inf
        self.globalBestChromo = np.random.random([1, features])
        self.globalBestLv = features
        self.globalIndival = individual(chromo=self.globalBestChromo,
                                        fitness=self.globalBestFitness,
                                        lv=self.globalBestLv)
        self.mutBits = mutBits
        # 构造种群
        self.population = []
        for ii in range(self.idv):
            self.population.append(individual())
        # 初始化种群
        for indiv, ii in zip(self.population, list(range(self.idv))):
            indiv.chromo = self.State[ii, :].copy()
            indiv.fitness = np.inf
            indiv.lv = indiv.chromo.shape[0]
        # 局部最优
        self.currentBestFitness = np.inf
        self.currentBestChromo = None

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

    def neighbor(self):
        for index in range(self.idv):
            for ii in range(self.mutBits):
                mutBitPos = np.random.randint(0,self.featurens)
                self.population[index].chromo[mutBitPos] ^=1

    def updatePopulation(self, goodIdv):
        for idv in self.population:
            idv.chromo = goodIdv.chromo.copy()

    def SAAlgorithm(self):
        currentFitnessTrace = []
        globalFitnessTrace = []
        for ii in range(self.maxIter):
            print(ii)
            #计算能量
            fitnessAll = self.calPopulationFitness()
            self.currentBestFitness = min(fitnessAll)

            #记录最优
            currentFitnessTrace.append(self.currentBestFitness)
            currentIdvIndex = fitnessAll.index(self.currentBestFitness)
            self.currentBestChromo = \
                self.population[currentIdvIndex].chromo
            if self.currentBestFitness < self.globalBestFitness:
                self.globalBestFitness = self.currentBestFitness
                self.globalBestChromo = self.currentBestChromo.copy()
                self.globalBestLv = self.population[currentIdvIndex].lv
                self.globalIndival.setParameter(chromo=self.globalBestChromo,
                                                fitness=self.globalBestFitness,
                                                lv=self.globalBestLv)
            globalFitnessTrace.append(self.globalBestFitness)

            #metropolis准则
            currentIndv = self.population[currentIdvIndex]
            #如果当前最优小于全局最优，使用当前最优
            #否则，按照metropolis准则以概率p接受当前优解
            if self.currentBestFitness > self.globalBestFitness:
                detaFit = self.currentBestFitness - self.globalBestFitness
                acceptP = np.exp(-detaFit/self.T)
                prob = np.random.random()
                if prob <acceptP:
                    self.updatePopulation(currentIndv)
                else:
                    self.updatePopulation(self.globalIndival)
            else:
                self.updatePopulation(currentIndv)

            #搜索近邻
            self.neighbor()
            #温度降低
            self.T = self.T * self.alpha
        return self.globalIndival, currentFitnessTrace, globalFitnessTrace