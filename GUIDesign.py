# -*- coding: utf-8 -*-
"""
__title__ = ""
__author__ = "Sol Young"
__mtime__ = "2018/4/10"
bug fuck off!!!
"""

import sys
import random
import gc

from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow, QAction, QPushButton, QApplication,\
                             QLabel, QWidget, QMessageBox, QHBoxLayout, QVBoxLayout,\
                             QGridLayout,QSizePolicy, QComboBox
from PyQt5.Qt import QLineEdit
from PyQt5.QtGui import QIcon,QFont,QColor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.pyplot import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties

# 判断操作系统
import platform
operationSys = platform.platform()
if operationSys[:5] == 'Windo':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 如果是window系统可以直接设置matplotlib字体为黑体
# 如果是linux系统，需要手动设置设置字体
else:
    if plt.rcParams['font.sans-serif'][0] != u'Microsoft YaHei':
        print('没有合适的matplotlib字体，图像显示可能出现问题。设置过程参考https://www.zhihu.com/question/25404709')


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

    def compute_initial_figure(self):
        pass

class DynamicDrawMachines(MyMplCanvas):
    def __init__(self, *args, **kwargs):
        self.points = 100
        MyMplCanvas.__init__(self, *args, **kwargs)


    def compute_initial_figure(self):
        self.xData = list(range(self.points ))
        self.yData = [0]*self.points
        # self.axes.plot(self.xData,self.yData,'b')
        self.axes.set_ylim([0,100])
        self.axes.set_xlim([0, self.points])
        self.axes.set_yticks(range(0,101,10))
        self.axes.grid(True)


    def update_figure(self):
        self.yData = self.yData[1:] + [random.randint(20, 80)]
        self.axes.plot(self.xData,self.yData,'b')
        self.axes.set_ylim([0, 100])
        self.axes.set_xlim([0, self.points])
        self.axes.set_yticks(range(0, 101, 10))
        self.axes.grid(True)
        self.draw()

    def cla(self):

        self.axes.clear()
        self.axes.cla()
        #self.axes.clf()
        gc.collect()
        self.draw()

    def plot(self, *args, **kwargs):
        self.axes.plot(*args, **kwargs)
        self.axes.set_ylim([0, 100])
        self.axes.set_xlim([0, 100])
        self.axes.set_yticks(range(0, 101, 10))
        self.axes.set_ylabel(u'使用率')
        self.axes.grid(True)
        self.draw()

    def grid(self, default=True):
        if default:
            self.axes.grid()
        else:
            self.axes.grid(False)

    def scatter(self, *args, **kwargs):
        self.axes.scatter(*args, **kwargs)
        self.axes.set_ylim([0, 100])
        self.axes.set_xlim([0, 100])
        self.axes.set_zlim([0, 100])
        #self.axes.set_yticks(range(0, 101, 10))
        self.axes.set_xlabel(u'内存')
        self.axes.set_ylabel(u'磁盘')
        self.axes.set_zlabel(u'CPU')
        self.axes.grid(True)
        self.draw()

    def xlabel(self,Xlabel = 'X'):
        self.axes.set_xlabel(Xlabel)

    def ylabel(self,Ylabel = 'X'):
        self.axes.set_ylabel(Ylabel)

class ApplicationWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.currentPath = sys.path[0]  # 程序运行的路径
        self.initUI()




    def initUI(self):
        self.setWindowIcon(QIcon(r'G:\myRepostory\graduateProject\logo\xjtu.jpg'))
        #菜单
        #Qaction
        openFileAct = QAction('&打开',self,triggered = self.openFile)
        openFileAct.setShortcut('Ctrl+O')
        exitAct = QAction('&退出',self,triggered=self.fileQuit)
        exitAct.setShortcut('Ctrl+Q')


        menubar = self.menuBar()
        #开始菜单
        fileMenu = menubar.addMenu('&开始')
        fileMenu.addAction(openFileAct)
        fileMenu.addAction(exitAct)
        #方法

        #帮助
        helpMenu = menubar.addMenu('&帮助')
        aboutAction = QAction('&关于',self,triggered = self.about)
        helpMenu.addAction(aboutAction)

        #状态栏
        self.statusBar().showMessage('准备就绪',2000)

        #主界面布局
        self.main_widget = QWidget(self)
        mainFunctionLabel = QLabel('命令')
        importButton = QPushButton("导入数据")
        startButton = QPushButton('开始')
        stopButton = QPushButton('停止')
        exitButton = QPushButton('退出')
        drawPic = DynamicDrawMachines(self.main_widget,width=5,height=4,dpi=100)

        self.hboxButton = QHBoxLayout()
        self.hboxButton.addWidget(importButton)
        self.hboxButton.addWidget(startButton)
        self.hboxButton.addWidget(stopButton)
        self.hboxButton.addWidget(exitButton)

        self.gboxAlgo = QGridLayout()
        experNumLabel = QLabel('试验次数')
        experNumTxt = QLineEdit()
        experNumTxt.setText('100')
        experNumTxt.setFixedWidth(100)
        algorithmsLable = QLabel('算法')
        availableAlgos = ['PLS','RReliefF','UVE-PLS','SA-PLS','GA-PLS','ACO-PLS','LASSO','Elastic Net','FP-Tree-PLS']
        algorithmBlock = QComboBox()
        algorithmBlock.insertItems(1,availableAlgos)

        self.gboxAlgo.addWidget(algorithmsLable, 0, 0)
        self.gboxAlgo.addWidget(algorithmBlock, 0, 1)
        self.gboxAlgo.addWidget(experNumLabel,0,2)
        self.gboxAlgo.addWidget(experNumTxt,0,3)

        analysisLabel = QLabel('分析结果')
        rmsecvLabel = QLabel('RMSECV')
        rmsetLabel = QLabel('RMSET')
        rmsepLabel = QLabel('RMSEP')
        r2cvLabel = QLabel('R2CV')
        r2pLabel = QLabel('R2P')
        CRlabel = QLabel('CR')
        numrmsecvLabel = QLabel('0')
        numrmsetLabel = QLabel('0')
        numrmsepLabel = QLabel('0')
        numr2cvLabel = QLabel('1')
        numr2pLabel = QLabel('1')
        numCRlabel = QLabel('1')

        self.gboxAnalysis = QGridLayout()
        self.gboxAnalysis.addWidget(analysisLabel,0,0)
        self.gboxAnalysis.addWidget(rmsecvLabel,1,0)
        self.gboxAnalysis.addWidget(rmsetLabel,1,1)
        self.gboxAnalysis.addWidget(rmsepLabel,1,2)
        self.gboxAnalysis.addWidget(r2cvLabel,1,3)
        self.gboxAnalysis.addWidget(r2pLabel,1,4)
        self.gboxAnalysis.addWidget(CRlabel,1,5)
        self.gboxAnalysis.addWidget(numrmsecvLabel, 2, 0)
        self.gboxAnalysis.addWidget(numrmsetLabel, 2, 1)
        self.gboxAnalysis.addWidget(numrmsepLabel, 2, 2)
        self.gboxAnalysis.addWidget(numr2cvLabel, 2, 3)
        self.gboxAnalysis.addWidget(numr2pLabel, 2, 4)
        self.gboxAnalysis.addWidget(numCRlabel, 2, 5)

        self.vboxButton = QVBoxLayout(self.main_widget)
        self.vboxButton.addWidget(mainFunctionLabel)
        self.vboxButton.addLayout(self.hboxButton)
        self.vboxButton.addLayout(self.gboxAlgo)
        self.vboxButton.addWidget(drawPic)
        self.vboxButton.addLayout(self.gboxAnalysis)
        self.setLayout(self.vboxButton)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        # 设置分辨率
        self.setGeometry(300,300,600,600)
        self.show()

    def about(self):
        QMessageBox.about(self, "关于",
                          """
sol yang 2018
                          """
                          )
    def openFile(self):
        pass

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ApplicationWindow()
    ex.setWindowTitle("变量选择工具箱")
    # ex.show()
    sys.exit(app.exec_())