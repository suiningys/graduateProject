# -*- coding: utf-8 -*-
"""
__title__ = ""
__author__ = "Sol Young"
__mtime__ = "2018/3/5"
bug fuck off!!!
"""
import numpy as np

class treeNode(object):
    def __init__(self, name = None, numOccur = 1, parentNode = None):
        self.name = name
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self,numOccur = 1):
        self.count += numOccur

def trans2Array(trans):
    T2A = {}
    number = 0
    for tran in trans:
        for item in tran:
            if not item in T2A.keys():
                T2A[item] = number
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
    transactions = array2Trans(mat)

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
    rootNode = treeNode('root node', 1, None)
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
        inTree.children[items[0]] = treeNode(items[0],1,inTree)
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