# -*- coding: utf-8 -*-
"""
__title__ = ""
__author__ = "Sol Young"
__mtime__ = "2018/4/1"
bug fuck off!!!
"""
from numpy import log2
import numpy as np
from itertools import combinations, permutations

'''
函数：setBase
功能：设置正交表的第ii个基准列
入参：ii：第ii个基准列，ii∈[1,bottom]
     bottom：正交表的s
出参：无
返回：基准列
'''
def setBase(ii,bottom):
    n = 2**bottom
    column = np.zeros([n,1]).astype(int)#返回的列
    step = 2**(bottom - ii)#步长
    K = 2**(ii-1) #块的个数
    KBegin = list(range(0,n,2* step))
    oneIndex = [ii+step for ii in KBegin]
    for oneTemp in oneIndex:
        for ii in range(step):
            column[oneTemp+ii]=1
    return column

'''
函数：generateBase
功能：生成正交表的基准列
入参：bottom：正交表的s
出参：无
返回：基准
'''
def generateBase(bottom):
    base = setBase(1,bottom)
    for ii in range(2,bottom+1):
        newColumn = setBase(ii,bottom)
        base = np.hstack((base,newColumn))
    return base

'''
函数：generateOrder
功能：设置生成正交表列所需要的基准列
入参：n：第n个区间的列，n∈[2,bottom]
出参：无
返回：序
'''
def generateOrder(n):
    # n = n+1
    order = []
    if n<1:
        return order
    else:
        v = list(range(0,n))
        for ii in range(1,n+1):
            combations = list(combinations(v,ii))
            order.extend(combations)
    return order
'''
列异或
'''
def xor(col1, col2):
    return np.array([col1[ii]^col2[ii] for ii in range(max(col1.shape))])

'''
函数：generateNewColumn
功能：生成正交表的一列
入参：item:生成该列需要异或的基准列
     nowBase：该列的基准列
出参：无
返回：正交表新的列
'''
def generateNewColumn(item, nowBase, base):
    baseNum = len(item)
    newColumn = xor(base[:,item[0]],base[:,nowBase-1])
    if baseNum>=2:
        for ii in range(1,baseNum):
            newColumn = xor(newColumn,base[:,item[ii]])
    return newColumn
'''
函数：generateOrthArray
功能：生成正交表
入参：f：因素的个数
出参：无
返回：正交表
'''
def generateOrthArray(f):
    bottom = int(log2(f))+1
    n = 2**bottom
    r = n-1
    base = generateBase(bottom)
    orthArray = np.zeros([n,r]).astype(int)
    for ii in range(bottom):
        orthArray[:,2**ii-1] = base[:,ii]
    for ii in range(2,bottom+1):
        colNum = 2**(ii-1)-1
        order = generateOrder(ii-1)
        combNum = len(order)
        for jj in range(1,combNum+1):
            orthArray[:,colNum+jj] = generateNewColumn(order[jj-1],ii,base)
    return orthArray