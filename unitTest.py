# -*- coding: utf-8 -*-
"""
__title__ = ""
__author__ = "Sol Young"
__mtime__ = "2018/2/28"
bug fuck off!!!
"""

import unittest

import numpy as np
from sklearn.model_selection import train_test_split

from readData import readData
from readData import UVECV
from readData import UVE
#from readData import PLSwithAllFeatures

from heuristic_algorithm import individual
from heuristic_algorithm import GeneAlgorithm

class testCV(unittest.TestCase):
    def test_init(self):
        pass
        # CO, CO2, CH4, specData = readData()
        # self.CO = CO
        # self.CO2 = CO2
        # self.CH4 = CH4
        # self.specData = specData

    # def testReadData(self):
    #     self.assertTrue(readData())

    def test_CV(self):
        CO, CO2, CH4, specData = readData()
        res,coefs = UVECV(specData,CO,50)
        print(res)
        self.assertIsInstance(res, float)

    def test_UVE(self):
        CO, CO2, CH4, specData = readData()
        rmsecvStd, uveLvStd, finalRes = UVE(specData,CO)
        self.assertIsInstance(rmsecvStd,float)

    def test_PLSAll(self):
        CO, CO2, CH4, specData = readData()
        xTrain, xTest, yTrain, yTest = \
            train_test_split(specData, CO, test_size=0.25, random_state=42)
        yPredict, R2, MSE, R2P = PLSwithAllFeatures(xTest, yTest, xTrain, yTrain)
        print("R2P is", R2P)
        self.assertIsInstance(yPredict, np.ndarray)

class testGA(unittest.TestCase):

    def testCO(self):
        CO, CO2, CH4, specData = readData()
        xTrain, xTest, yTrain, yTest = \
            train_test_split(specData, CO, test_size=0.25, random_state=42)
        GACO = GeneAlgorithm(xTrain,yTrain)
        # ind1 = individual(chrome=np.random.randint(0,1,size=[1,10]))
        # ind2 = individual(chrome=np.random.randint(0, 1, size=[1, 10]))
        GACO.population[0].chromo = np.ones(GACO.chromeBits)
        GACO.population[1].chromo = np.zeros(GACO.chromeBits)
        GACO.crossOver(GACO.population[0].chromo,GACO.population[1].chromo)
        print(np.sum(GACO.population[0].chromo))
        print(np.sum(GACO.population[1].chromo))
        #GACO.crossOver(np.ones([1,GACO.chromeBits]), np.zeros([1,GACO.chromeBits]))

    def testCalFitness(self):
        CO, CO2, CH4, specData = readData()
        xTrain, xTest, yTrain, yTest = \
            train_test_split(specData, CO, test_size=0.25, random_state=42)
        GACO = GeneAlgorithm(xTrain, yTrain)
        GACO.calPopulationFitness()


if __name__=="__main__":
    unittest.main()