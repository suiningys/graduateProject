# -*- coding: utf-8 -*-
"""
__title__ = ""
__author__ = "Sol Young"
__mtime__ = "2018/2/28"
bug fuck off!!!
"""

import unittest

import numpy as np

from readData import readData
from readData import UVECV

class testCV(unittest.TestCase):
    def testInit(self):
        pass

    def testReadData(self):
        self.assertTrue(readData())

    def test_CV(self):
        CO, CO2, CH4, specData = readData()
        res = UVECV(specData,CO,50)
        print(res)
        self.assertIsInstance(res, np.ndarray)

if __name__=="__main__":
    unittest.main()