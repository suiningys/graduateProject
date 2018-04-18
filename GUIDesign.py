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
import pandas as pd
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow, QAction, QPushButton, QApplication,\
                             QLabel, QWidget, QMessageBox, QHBoxLayout, QVBoxLayout,\
                             QGridLayout,QSizePolicy, QComboBox,QFileDialog, QGroupBox
from PyQt5.Qt import QLineEdit
from PyQt5.QtGui import QIcon,QFont,QColor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.pyplot import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from readData import *
from GUIChangePara import *
from RReliefF import RReliefF

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

        # self.axes = fig.add_subplot(111,projection='3d')
        self.axes = fig.add_subplot(111)
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
        self.axes.set_xlabel(u'真实值')
        self.axes.set_ylabel(u'预测值')
        self.axes.grid(False)


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
        # self.axes.set_yticks(range(0, 101, 10))
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

    def scatter2D(self, *args, **kwargs):
        minLim = min(args[0].min(), args[1].min())
        maxLim = max(args[0].max(), args[1].max())
        self.axes.scatter(*args, **kwargs)
        self.axes.plot([minLim * 1.3, maxLim * 1.01], [minLim * 1.3, maxLim * 1.01], c='k')
        self.axes.set_xlim(minLim * 1.3, maxLim * 1.01)
        self.axes.set_ylim(minLim * 1.3, maxLim * 1.01)
        self.axes.grid(False)
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
        self.xTrain = None


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
        # mainFunctionLabel = QLabel('命令')
        importButton = QPushButton("导入数据")
        importButton.clicked.connect(self.loadData)
        startButton = QPushButton('开始')
        startButton.clicked.connect(self.startSelection)
        stopButton = QPushButton('停止')
        stopButton.clicked.connect(self.stopSelection)
        plotButton = QPushButton('绘图')
        plotButton.clicked.connect(self.plotPic)
        exitButton = QPushButton('退出')
        exitButton.clicked.connect(self.fileQuit)
        self.drawPic = DynamicDrawMachines(self.main_widget,width=5,height=4,dpi=100)

        self.hboxButtonBox = QGroupBox('命令')
        self.hboxButton = QHBoxLayout()
        self.hboxButton.addWidget(importButton)
        self.hboxButton.addWidget(startButton)
        self.hboxButton.addWidget(stopButton)
        self.hboxButton.addWidget(plotButton)
        self.hboxButton.addWidget(exitButton)
        self.hboxButtonBox.setLayout(self.hboxButton)


        self.gboxAlgo = QGridLayout()
        # experNumLabel = QLabel('试验次数')
        # experNumTxt = QLineEdit()
        # experNumTxt.setText('100')
        # experNumTxt.setFixedWidth(100)
        algorithmsLable = QLabel('算法')
        self.availableAlgos = ['PLS','RReliefF','UVE-PLS','SA-PLS','GA-PLS','ACO-PLS','LASSO','Elastic Net','FP-Tree-PLS']
        self.algorithmBlock = QComboBox()
        self.algorithmBlock.insertItems(1,self.availableAlgos)
        self.algorithmBlock.currentIndexChanged.connect(self.changeAlgorithm)
        self.changeParameter = QPushButton('修改算法参数')
        self.changeParameter.clicked.connect(self.changeAlgoParameter)

        self.gboxAlgoBox = QGroupBox('算法')
        self.gboxAlgo.addWidget(algorithmsLable, 0, 0)
        self.gboxAlgo.addWidget(self.algorithmBlock, 0, 1)
        self.gboxAlgo.addWidget(self.changeParameter,0,2)
        # self.gboxAlgo.addWidget(experNumLabel,0,3)
        # self.gboxAlgo.addWidget(experNumTxt,0,4)
        self.gboxAlgoBox.setLayout(self.gboxAlgo)

        analysisLabel = QLabel('分析结果')
        rmsecvLabel = QLabel('RMSECV')
        rmsetLabel = QLabel('RMSET')
        rmsepLabel = QLabel('RMSEP')
        r2cvLabel = QLabel('R2CV')
        r2pLabel = QLabel('R2P')
        CRlabel = QLabel('CR')
        self.numrmsecvLabel = QLabel('0')
        self.numrmsetLabel = QLabel('0')
        self.numrmsepLabel = QLabel('0')
        self.numr2cvLabel = QLabel('1')
        self.numr2pLabel = QLabel('1')
        self.numCRlabel = QLabel('1')

        self.gboxAnalysisBox = QGroupBox('分析结果')
        self.gboxAnalysis = QGridLayout()
        # self.gboxAnalysis.addWidget(analysisLabel,0,0)
        self.gboxAnalysis.addWidget(rmsecvLabel,1,0)
        self.gboxAnalysis.addWidget(rmsetLabel,1,1)
        self.gboxAnalysis.addWidget(rmsepLabel,1,2)
        self.gboxAnalysis.addWidget(r2cvLabel,1,3)
        self.gboxAnalysis.addWidget(r2pLabel,1,4)
        self.gboxAnalysis.addWidget(CRlabel,1,5)
        self.gboxAnalysis.addWidget(self.numrmsecvLabel, 2, 0)
        self.gboxAnalysis.addWidget(self.numrmsetLabel, 2, 1)
        self.gboxAnalysis.addWidget(self.numrmsepLabel, 2, 2)
        self.gboxAnalysis.addWidget(self.numr2cvLabel, 2, 3)
        self.gboxAnalysis.addWidget(self.numr2pLabel, 2, 4)
        self.gboxAnalysis.addWidget(self.numCRlabel, 2, 5)
        self.gboxAnalysisBox.setLayout(self.gboxAnalysis)

        self.vboxButton = QVBoxLayout(self.main_widget)
        # self.vboxButton.addWidget(mainFunctionLabel)
        # self.vboxButton.addLayout(self.hboxButton)
        self.vboxButton.addWidget(self.hboxButtonBox)
        # self.vboxButton.addLayout(self.gboxAlgo)
        self.vboxButton.addWidget(self.gboxAlgoBox)
        self.vboxButton.addWidget(self.drawPic)
        # self.vboxButton.addLayout(self.gboxAnalysis)
        self.vboxButton.addWidget(self.gboxAnalysisBox)
        self.setLayout(self.vboxButton)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        # 设置分辨率
        self.setGeometry(300,300,600,700)
        self.show()

        #更改参数窗口
        # self.changeParaWindow = changeParaWindow()
        # self.changeParaWindow.setWindowTitle("修改参数")

    def about(self):
        QMessageBox.about(self, "关于",
                          """
sol yang 2018
                          """
                          )

    def noDataWarning(self):
        QMessageBox.about(self,"没有数据",
                         """
请先读取数据
                         """)

    def openFile(self):
        fname = QFileDialog.getOpenFileName(self, '打开文件', self.currentPath,"CSV Files (*.csv);;Excel File (*.xlsx);;所有文件 (*)")

        if fname[0]:
            f = open(fname[0], 'r')

            with f:
                data = f.read()


    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def loadData(self):
        self.statusBar().showMessage('正在加载数据', 2000)
        '''
        read data
        '''
        self.CO, self.CO2, self.CH4, self.specData = readData()
        d = pd.read_excel('./data/NONO2SO2.xlsx')
        lines = d.shape[0]
        self.NO = d['NO'].as_matrix().reshape(lines, 1)
        self.NO2 = d['NO2'].as_matrix().reshape(lines, 1)
        self.SO2 = d['SO2'].as_matrix().reshape(lines, 1)
        self.specData2 = d.iloc[:, 4:].as_matrix()
        self.statusBar().showMessage('加载数据完成', 2000)
        '''
        split the data set
        If int, random_state is the seed used by the random number generator; 
        If RandomState instance, random_state is the random number generator; If None, 
        the random number generator is the RandomState instance used by np.random.
        '''
        self.xTrainOrigin, self.xTestOrigin, self.yTrainOrigin, self.yTestOrigin = \
            train_test_split(self.specData2, self.SO2, test_size=0.2, random_state=42)

        '''
        数据预处理
        '''
        # mean removal
        self.scalerX = preprocessing.StandardScaler().fit(self.xTrainOrigin)
        self.xTrain = self.scalerX.transform(self.xTrainOrigin)
        self.xTest = self.scalerX.transform(self.xTestOrigin)
        self.scalerY = preprocessing.StandardScaler().fit(self.yTrainOrigin)
        self.yTrain = self.scalerY.transform(self.yTrainOrigin)
        self.yTest = self.scalerY.transform(self.yTestOrigin)

    '''
    计算模型评价参数
    '''
    def modelMerit(self, SelectedIndex):
        xTrainSelected = self.xTrain[:, SelectedIndex]
        xTestSelected = self.xTest[:, SelectedIndex]
        kf = model_selection.KFold(n_splits=5, random_state=10)
        trans, features = xTrainSelected.shape
        lvMax = int(min(trans, features) / 3)
        lvBest = 0
        rmsecvBest = np.inf
        for lvTemp in range(1, lvMax + 1):
            squareArray = np.array([[]])
            for train, test in kf.split(xTrainSelected):
                xTrainTemp = xTrainSelected[train, :]
                yTrainTemp = self.yTrain[train]
                xTestTemp = xTrainSelected[test, :]
                yTestTemp = self.yTrain[test]
                yPredictTemp, coefTemp = PLS(xTestTemp, yTestTemp, xTrainTemp, yTrainTemp, lvTemp)
                yPredictTempTrue = self.scalerY.inverse_transform(yPredictTemp)
                yTestTempTrue = self.scalerY.inverse_transform(yTestTemp)
                residual = yPredictTempTrue - yTestTempTrue
                square = np.dot(residual.T, residual)
                squareArray = np.append(squareArray, square)
                # squareArray.append(square)
            RMSECV = np.sqrt(np.sum(squareArray) / xTrainSelected.shape[0])
            if RMSECV < rmsecvBest:
                rmsecvBest = RMSECV
                lvBest = lvTemp

        plsModel = cross_decomposition.PLSRegression(n_components=lvBest)
        plsModel.fit(xTrainSelected, self.yTrain)

        yPredict = plsModel.predict(xTestSelected)
        yTrainPredict = plsModel.predict(xTrainSelected)
        yTrainPredictTrue = self.scalerY.inverse_transform(yTrainPredict)
        yPredictTrue = self.scalerY.inverse_transform(yPredict)
        self.yPredcit = yPredict

        MEST = sm.mean_squared_error(self.yTrainOrigin, yTrainPredictTrue)
        RMSET = np.sqrt(MEST)
        R2T = sm.r2_score(self.yTrainOrigin, yTrainPredictTrue)
        MSEP = sm.mean_squared_error(self.yTestOrigin, yPredictTrue)
        RMSEP = np.sqrt(MSEP)
        R2P = sm.r2_score(self.yTestOrigin, yPredictTrue)
        # 计算交叉验证误差
        yPredictCV = np.array([[]])
        yTrueCV = np.array([[]])
        for train, test in kf.split(xTrainSelected):
            xTrainTemp = xTrainSelected[train, :]
            yTrainTemp = self.yTrain[train]
            xTestTemp = xTrainSelected[test, :]
            yTestTemp = self.yTrain[test]
            yPredictTemp, coefTemp = PLS(xTestTemp, yTestTemp, xTrainTemp, yTrainTemp, lvBest)
            yPredictTempTrue = self.scalerY.inverse_transform(yPredictTemp)
            yTestTempTrue = self.scalerY.inverse_transform(yTestTemp)
            yPredictCV = np.append(yPredictCV, yPredictTempTrue)
            yTrueCV = np.append(yTrueCV, yTestTempTrue)
            residual = yPredictTempTrue - yTestTempTrue
            square = np.dot(residual.T, residual)
            squareArray = np.append(squareArray, square)
            # squareArray.append(square)
        RMSECV = np.sqrt(np.sum(squareArray) / xTrainSelected.shape[0])
        R2CV = sm.r2_score(yPredictCV, yTrueCV)

        CR = 1 - len(SelectedIndex) / self.xTrain.shape[1]
        return rmsecvBest, RMSET, RMSEP, R2T, R2P, R2CV, CR


    def startSelection(self):
        self.selectedAlgorithm = self.algorithmBlock.currentText()
        if(self.xTrain is None):
            self.noDataWarning()
            return
        xTrain = self.xTrain
        xTest = self.xTest
        yTrain = self.yTrain
        yTest = self.yTest
        '''
        'PLS','RReliefF','UVE-PLS','SA-PLS','GA-PLS',
        'ACO-PLS','LASSO','Elastic Net','FP-Tree-PLS'
        '''
        self.statusBar().showMessage('%s算法开始' % self.selectedAlgorithm, 1000)
        if self.selectedAlgorithm=='RReliefF':
            w = RReliefF(self.xTrain, self.yTrain)
            th = np.percentile(w, 80)  # 计算分位数
            selectedIndex = np.where(w >= th)[0]

        elif self.selectedAlgorithm=='PLS':
            selectedIndex = list(range(self.xTrain.shape[1]))

        modelMerit = self.modelMerit(selectedIndex)
        self.displayMerit(modelMerit)
        self.statusBar().showMessage('%s算法结束' % self.selectedAlgorithm)

    def displayMerit(self,modelMerit):
        self.numrmsecvLabel.setText('%.2f' %modelMerit[0])
        self.numrmsetLabel.setText('%.2f' %modelMerit[1])
        self.numrmsepLabel.setText('%.2f' %modelMerit[2])
        self.numr2cvLabel.setText('%.2f' %modelMerit[-2])
        self.numr2pLabel.setText('%.2f' %modelMerit[4])
        self.numCRlabel.setText('%.2f' %modelMerit[-1])

    def stopSelection(self):
        pass


    def plotPic(self):
        if (self.xTrain is None):
            self.noDataWarning()
            return
        yTure = self.scalerY.inverse_transform(self.yTest)
        yPredict = self.scalerY.inverse_transform(self.yPredcit)
        self.drawPic.scatter2D(yTure,yPredict)


    def changeAlgorithm(self):
        self.selectedAlgorithm = self.algorithmBlock.currentText()
        self.statusBar().showMessage('选择%s算法' %self.selectedAlgorithm, 1000)

    def changeAlgoParameter(self):
        '''
        'PLS','RReliefF','UVE-PLS','SA-PLS','GA-PLS',
        'ACO-PLS','LASSO','Elastic Net','FP-Tree-PLS'
        '''
        self.selectedAlgorithm = self.algorithmBlock.currentText()
        if self.selectedAlgorithm == 'FP-Tree-PLS':
            self.changeParaWindow = changeFPParaWindow()
        elif self.selectedAlgorithm == 'SA-PLS':
            self.changeParaWindow = changeSAParaWindow()
        else:
            self.changeParaWindow = classicWindow()
        self.changeParaWindow.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ApplicationWindow()
    ex.setWindowTitle("变量选择工具箱")
    # ex.show()
    sys.exit(app.exec_())