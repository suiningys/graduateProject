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

def readData():
    dataPath = './COCO2CH4.xlsx'
    d = pd.read_excel(dataPath)
    CO = d['CO']
    CH4 = d['CH4']
    C02 = d['CO2']
    specData = d.iloc[:,5:]


if __name__=="__main__":
    pass