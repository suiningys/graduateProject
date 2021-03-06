# -*- coding: utf-8 -*-
"""
__title__ = ""
__author__ = "Sol Young"
__mtime__ = "2018/3/5"
bug fuck off!!!
"""
import numpy as np
import matplotlib.pyplot as plt
from generateOrthogonalArrays import generateOrthArray
from readData import plsRegressAnalysis
from readData import saveData, loadData

class treeNode(object):
    def __init__(self, name = None, numOccur = 1,
                 parentNode = None, level = None, plotPos = []):
        self.name = name
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}
        self.level = level
        self.plotPos = plotPos #画在图上的位置
    def inc(self,numOccur = 1):
        self.count += numOccur

#node name -> node true name
A2T = {"root node":"root"}

def trans2Array(trans):
    T2A = {}
    number = 0
    for tran in trans:
        for item in tran:
            if not item in T2A.keys():
                T2A[item] = number
                A2T[number] = item
                number +=1
    itemTotal = max(T2A.values())
    arrayTemp = np.zeros([len(trans),itemTotal+1])
    for ii in range(len(trans)):
        for item in trans[ii]:
            arrayTemp[ii,T2A[item]] = 1
    return arrayTemp, T2A

def array2Trans(mat):
    rows, columns = mat.shape
    transcations = []
    for row in range(rows):
        lineTemp = mat[row,:]
        trans = np.where(lineTemp==1)[0]
        transcations.append(trans)
    return transcations

def createTree(mat, minSup=0.8):
    if isinstance(mat,np.ndarray):#如果输入的是个矩阵，先转换成事物
        transactions = array2Trans(mat)
    else:#如果输入的就是事物类型，直接赋值
        transactions = mat

    headerTable = {}
    for tran in transactions:
        for item in tran:
            headerTable[item] = headerTable.get(item,0) + 1
    for key in list(headerTable.keys()):
        if headerTable[key]/len(transactions)<minSup:
            del (headerTable[key])
    frequentItemSet = set(headerTable.keys())
    if len(frequentItemSet)==0: return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    rootNode = treeNode('root node', 1, None,0)
    for tranSet in transactions:
        localDict = {}
        for item in tranSet:
            if item in frequentItemSet:
                localDict[item] = headerTable[item][0]
        if len(localDict)>0:
            orderedItems = [v[0] for v in sorted(localDict.items(),
                                                 key=lambda p:p[1],
                                                 reverse=True)]
            updateTree(orderedItems, rootNode,headerTable)
    return rootNode, headerTable

def updateTree(items, inTree, headerTable):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(1)
    else:
        inTree.children[items[0]] = treeNode(items[0],1,inTree,inTree.level+1)
        if headerTable[items[0]][1] is None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1],
                         inTree.children[items[0]])
    if len(items)>1:
        updateTree(items[1:],inTree.children[items[0]],headerTable)

def updateHeader(nodeToTest, targetNode):
    while(not nodeToTest.nodeLink is None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def ergodicTree(rootNode, array = [], level = 0):
    #level is the deep of rootNode
    if level==0: array.append([]), array[0].append(rootNode)
    if len(rootNode.children)==0: return
    for child in rootNode.children.values():
        if len(array)==level+1: array.append([])
        array[level+1].append(child)
        ergodicTree(child,array,level=level+1)

nodePlt = dict(boxstyle="round64", fc="0.8")
arrowArgs = dict(arrowstyle="<-")

def createPlot(hold = True,axis = 'off'):
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    axes = fig.add_subplot(111)
    axes.hold(hold)
    axes.axis(axis)
    return axes, fig

def plotBranch(axes, ParentNode,ChildNode):
    xTick = [ParentNode.plotPos[0],ChildNode.plotPos[0]]
    yTick = [ParentNode.plotPos[1],ChildNode.plotPos[1]]
    # axes.plot(ParentNode.plotPos,ChildNode.plotPos,'-bo')
    axes.plot(xTick,yTick,'-b.')
    axes.annotate(r'%s:%d' % (A2T[ParentNode.name], ParentNode.count), xy=[x for x in ParentNode.plotPos])
    axes.annotate(r'%s:%d' % (A2T[ChildNode.name], ChildNode.count), xy=[x for x in ChildNode.plotPos])
    # axes.annotate(r'%s:%d' %(A2T[ParentNode.name],ParentNode.count),xy=[x+0.1 for x in ParentNode.plotPos])
    # axes.annotate(r'%s:%d' %(A2T[ChildNode.name],ChildNode.count), xy=[x+0.1 for x in ChildNode.plotPos])

def drawTreeSimple(axes, rootNode, structArray = [], level = 0):
    array = structArray
    if level==0:
        array.append([])
        array[0].append(rootNode)
        rootNode.plotPos = [0,0]
    if len(rootNode.children)==0: return
    for child in rootNode.children.values():
        if len(array)==level+1:
            array.append([])
        array[level+1].append(child)
        child.plotPos = [len(array[level+1])-1,-level-1]
        plotBranch(axes,rootNode,child)
        drawTreeSimple(axes,child,array,level=level+1)

def drawTree2(axes, rootNode, structArray = [], level = 0):
    array = structArray
    if level==0:

        array.append(0)
        rootNode.plotPos = [0,0]
    if len(rootNode.children)==0: return
    for child in rootNode.children.values():
        if len(array)==level+1:
            array.append(0)
        # array[level+1].append(child)
        child.plotPos = [max(array[level],rootNode.plotPos[0]),-level-1]
        if array[level]<rootNode.plotPos[0]:
            a = 10
        plotBranch(axes,rootNode,child)
        array[level] =child.plotPos[0]+1
        drawTree2(axes,child,array,level=level+1)

def ergodicTreeBranch(Node, branch = [], branchTemp = []):
    if Node.name!='root node':
        branchTemp.append(Node)
    if(len(Node.children)==0):
        branch.append(branchTemp)
        return
    else:
        for child in Node.children.values():
            branchTemp2 = branchTemp.copy()
            ergodicTreeBranch(child,branch,branchTemp2)


def useFPtree(xTrain, yTrain):
    trans, features = xTrain.shape
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
    return bestBranch


def testAll():
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
    testTree, headerTable = createTree(transactions, minSup=0.0)
    structArray = []
    axes, fig = createPlot()
    drawTreeSimple(axes,testTree,structArray,0)
