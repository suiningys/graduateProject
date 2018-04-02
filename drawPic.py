# -*- coding: utf-8 -*-
"""
__title__ = ""
__author__ = "Sol Young"
__mtime__ = "2018/4/2"
bug fuck off!!!
"""
import numpy as np
import pandas as pd
from readData import readData
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif']='NSimSun,Times New Roman'
plt.rcParams['axes.unicode_minus']=False

def xticksIndex(data, strList):
    L = len(strList)
    width = data.max() - data.min()
    return [data.min()+ii*(width/L) for ii in range(L)]

CO,CO2,CH4,specData = readData()
wavelength = np.arange(549.44,4238.28,7.82)
specLine1 = specData[20,:]
plt.figure()
plt.plot(wavelength,specLine1,linewidth=1.0,color='k')
plt.xlim(wavelength.min()*0.9, wavelength.max()*1.01)
plt.ylim(specLine1.min()*1.1, specLine1.max()*1.1)
plt.xlabel("wavelength($nm^{-1}$)")
plt.ylabel("吸收率")
# plt.xticks([],['1000','1500','2000','2500','3000','3500','4000'])


d = pd.read_excel('./data/NONO2SO2.xlsx')
lines = d.shape[0]
NO = d['NO'].as_matrix().reshape(lines,1)
NO2 = d['NO2'].as_matrix().reshape(lines,1)
SO2 = d['SO2'].as_matrix().reshape(lines,1)
wavelength = np.arange(187.87,1026.97,0.41)
specData2 = d.iloc[:,4:].as_matrix()
specLine2 = specData2[20,:]
print(specLine2.shape)
plt.figure()
plt.plot(wavelength,specLine2,linewidth=1.0,color='k')
plt.xlim(wavelength.min()*0.9, wavelength.max()*1.01)
# plt.ylim(specLine1.min()*1.1, specLine1.max()*1.1)
plt.xlabel("wavelength($nm^{-1}$)")
plt.ylabel("吸收率")
plt.show()