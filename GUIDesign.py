# -*- coding: utf-8 -*-
"""
__title__ = ""
__author__ = "Sol Young"
__mtime__ = "2018/4/10"
bug fuck off!!!
"""

import sys
import random

import matplotlib
matplotlib.use("Qt5Agg")

from PyQt5 import QtCore
from PyQt5.QtWidgets import * #QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget
from PyQt5.QtGui import QColor
import statsmodels.api as sm

from numpy import arange, sin, pi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties

from statsmodels.tsa.arima_model import ARIMA
from random import random

import datetime
import time
import gc

class MyMplCanvas(FigureCanvas):
    """这是一个窗口部件，即QWidget（当然也是FigureCanvasAgg）"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        self.axes = fig.add_subplot(111,projection='3d')
        #self.axes.hold(True)

        self.compute_initial_figure()

        #
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.fig = fig
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class ApplicationWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        # 判断操作系统
        import platform
        self.operationSys = platform.platform()
        if self.operationSys[:5] == 'Windo':
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 如果是window系统可以直接设置matplotlib字体为黑体
        # 如果是linux系统，需要手动设置设置字体
        else:
            if plt.rcParams['font.sans-serif'][0] != u'Microsoft YaHei':
                self.noFontWarning()

        self.currentPath = sys.path[0]  # 程序运行的路径
        #界面设计
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("程序主窗口")

    def noFontWarning(self):
        QMessageBox.about(self, "matplotlib字体缺失",
                          """
                  没有合适的matplotlib字体，图像显示可能出现问题
                  自带的matplotlib字体在 fonts文件夹下
                  设置过程参考https://www.zhihu.com/question/25404709
                          """)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    aw = ApplicationWindow()
    aw.setWindowTitle("变量选择工具箱")
    aw.show()
    app.exec_()