# -*- coding: utf-8 -*-
"""
__title__ = ""
__author__ = "Sol Young"
__mtime__ = "2018/3/8"
bug fuck off!!!
"""

import numpy as np

from sklearn.datasets import load_iris
from sklearn.feature_selection import *
from sklearn.cross_decomposition import PLSRegression

def variacce(xData,p=0.8):
    sel = VarianceThreshold(threshold=p*(1-p))
    xDataSelected = sel.fit_transform(xData)
    return xDataSelected

def useSelectKBest(xData,yData,scoringFunction = f_regression):
    # For regression: f_regression, mutual_info_regression
    # For classification: chi2, f_classif, mutual_info_classif
    # Warning Beware not to use a regression scoring function with a classification problem, you will get useless results.
    xDataNew = SelectKBest(scoringFunction,k=2).fit_transform(xData,yData)
    return xDataNew

def useRFECV(xData,yData,lv = 10):
    estimator = PLSRegression(n_components=10)
    selector = RFECV(estimator,step=1,cv=5)
    selector = selector.fit(xData,yData)