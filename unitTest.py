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
from readData import plsRegressAnalysis
from readData import ElasticNetCVOwn

from heuristic_algorithm import individual
from heuristic_algorithm import GeneAlgorithm

from FPTreeAlgorithm import *
from dataPreprocessing import *
from RReliefF import *

from ACOalgorithm import acoAlgorithm

from generateOrthogonalArrays import *
from SAalgorithm import *
from readData import useElasticNetCV
from readData import ElasticNetCVOwn

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
        rmsecvStd, uveLvStd, finalRes, uveSelectedIndex= UVE(specData,CO2)
        self.assertIsInstance(rmsecvStd,float)

    def test_PLSAll(self):
        CO, CO2, CH4, specData = readData()
        xTrain, xTest, yTrain, yTest = \
            train_test_split(specData, CO, test_size=0.25, random_state=42)
        yPredict, R2, MSE, R2P = plsRegressAnalysis(xTest, yTest, xTrain, yTrain)
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
        GACO.crossOverChrome(GACO.population[0].chromo,GACO.population[1].chromo)
        print(np.sum(GACO.population[0].chromo))
        print(np.sum(GACO.population[1].chromo))
        #GACO.crossOver(np.ones([1,GACO.chromeBits]), np.zeros([1,GACO.chromeBits]))

    def testCalFitness(self):
        CO, CO2, CH4, specData = readData()
        xTrain, xTest, yTrain, yTest = \
            train_test_split(specData, CO, test_size=0.25, random_state=42)
        GACO = GeneAlgorithm(xTrain, yTrain)
        GACO.calPopulationFitness()
        for ind in GACO.population:
            print(ind.fitness)

    def testSelection(self):
        CO, CO2, CH4, specData = readData()
        xTrain, xTest, yTrain, yTest = \
            train_test_split(specData, CO, test_size=0.25, random_state=42)
        GACO = GeneAlgorithm(xTrain, yTrain)
        fitnessAll = GACO.calPopulationFitness()
        # xTrain = np.array([[1,2,3],[4,5,6]])
        # yTrain = np.array([[1],[2]])
        GACO.selection(fitnessAll)

    def testGA(self):
        CO, CO2, CH4, specData = readData()
        xTrain, xTest, yTrain, yTest = \
            train_test_split(specData, CO, test_size=0.25, random_state=42)
        GACO = GeneAlgorithm(xTrain, yTrain)
        globalIndival, currentFitnessTrace, globalFitnessTrace = GACO.GAAlgorithm()
        print(globalFitnessTrace)

class testFPtree(unittest.TestCase):

    def testFP(self):
        testTrans = [['a','b'],
                     ['b','c','d'],
                     ['a','c','d','e'],
                     ['a','d','e'],
                     ['a','b','c'],
                     ['a','b','c','d'],
                     ['a'],
                     ['a','b','c'],
                     ['a','b','d'],
                     ['b','c','e']]
        transactions,transferDict = trans2Array(testTrans)
        testTree, headerTable = createTree(transactions,minSup=0.0)
        structArray = []
        ergodicTree(testTree,structArray,0)
        print(structArray)
        branch = []
        ergodicTreeBranch(testTree,branch)
        for bc in branch:
            bcTemp = []
            for item in bc:
                bcTemp.append(item.name)
            print(bcTemp)

    def testPlot(self):
        parentNode = treeNode('parent', plotPos=[0, 0])
        childtNode = treeNode('child', plotPos=[1, 1])
        axes = createPlot()
        plotBranch(axes,parentNode,childtNode)

    def testDraw(self):
        testTrans = [['a', 'b'],
                     ['b', 'c', 'd'],
                     ['a', 'c', 'd', 'e'],
                     ['a', 'd', 'e'],
                     ['a', 'b', 'c'],
                     ['a', 'b', 'c', 'd'],
                     ['a'],
                     ['a', 'b', 'c'],
                     ['a', 'b', 'd'],
                     ['b', 'c', 'e']]
        transactions, transferDict = trans2Array(testTrans)
        testTree, headerTable = createTree(transactions, minSup=0)
        structArray = []
        axes, fig = createPlot()
        drawTreeSimple(axes, testTree, structArray, 0)
        plt.show()

    def testDraw2(self):
        testTrans = [['a', 'b','e'],
                     ['b', 'c'],
                     ['b', 'd', 'e'],
                     ['a'],
                     ['a']]
        transactions, transferDict = trans2Array(testTrans)
        testTree, headerTable = createTree(transactions, minSup=0)
        structArray = []
        axes, fig = createPlot()
        drawTree2(axes, testTree, structArray, 0)
        plt.show()

    def testUseFP(self):
        CO, CO2, CH4, specData = readData()
        xTrain, xTest, yTrain, yTest = \
            train_test_split(specData, CO, test_size=0.25, random_state=42)
        useFPtree(xTrain,yTrain)

class testPreprocess(unittest.TestCase):

    def testScale(self):
        data = createData()
        processingData(data)

    def testRReliefF(self):
        # CO, CO2, CH4, specData = readData()
        # xTrain, xTest, yTrain, yTest = \
        #     train_test_split(specData, CO, test_size=0.25, random_state=42)
        sampleNumbers = 500
        features = 10
        xTrain = np.random.random([sampleNumbers,features])
        yTrain = (xTrain[:,0]*1 + 1 *xTrain[:,1]).reshape(sampleNumbers,1)
        W = RReliefF(xTrain,yTrain)
        axes, fig = createPlot(axis='on')
        axes.plot(list(range(1,11)),W)
        plt.show()
        print(W)

class acoAlgorithmTest(unittest.TestCase):
    def testACO(self):
        CO, CO2, CH4, specData = readData()
        xTrain, xTest, yTrain, yTest = \
            train_test_split(specData, CO, test_size=0.25, random_state=42)
        ACOcase = acoAlgorithm(xTrain, yTrain)
        globalIndival, currentFitnessTrace, globalFitnessTrace = ACOcase.ACOAlgorithm()
        print(globalFitnessTrace)

class saAlgorithm(unittest.TestCase):
    def testSA(self):
        CO, CO2, CH4, specData = readData()
        xTrain, xTest, yTrain, yTest = \
            train_test_split(specData, CO, test_size=0.25, random_state=42)
        SACase = SA(xTrain,yTrain)
        globalIndival, currentFitnessTrace, globalFitnessTrace = SACase.SAAlgorithm()
        print(globalFitnessTrace)

class OrthTest(unittest.TestCase):

    def testSetBase(self):
        a = setBase(1,4)
        self.assertIsInstance(a[0][0],np.int32)
        self.assertIsInstance(a[-1][0], np.int32)

    def testGenerateBase(self):
        base = generateBase(3)
        print(base)

    def testGenerateOrder(self):
        order = generateOrder(2)
        print(order)

    def testGenerateNewColumn(self):
        base = generateBase(3)
        order = generateOrder(3)
        newColumn = xor(base[:,0],base[:,1])
        print(newColumn)
        newColumn = generateNewColumn([1],2,base)
        print(newColumn)

    def testGenerateOrth(self):
        orthArray = generateOrthArray(1000)
        print(orthArray)

class ENTest(unittest.TestCase):
    def testENCVO(self):
        CO, CO2, CH4, specData = readData()
        xTrain, xTest, yTrain, yTest = \
            train_test_split(specData, CO, test_size=0.25, random_state=42)
        yPredictEN, ENModel = useElasticNetCV(xTest, yTest, xTrain, yTrain)
        bestAlpha, bestL1 = ElasticNetCVOwn(xTest, yTest, xTrain, yTrain)

if __name__=="__main__":
    unittest.main()
