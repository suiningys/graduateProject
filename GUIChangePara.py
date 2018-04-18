# -*- coding: utf-8 -*-
"""
__title__ = ""
__author__ = "Sol Young"
__mtime__ = "2018/4/18"
bug fuck off!!!
"""
import sys

from PyQt5.QtWidgets import QMainWindow, QLabel, QWidget, QGridLayout
from PyQt5.Qt import QLineEdit
from PyQt5.QtGui import QIcon


class changeParaWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.currentPath = sys.path[0]  # 程序运行的路径

        self.setWindowIcon(QIcon(r'G:\myRepostory\graduateProject\logo\xjtu.jpg'))
        self.main_widget = QWidget(self)
        para1 = QLabel('训练集比例')
        para1.setFixedSize(20,20)
        self.para1Txt = QLineEdit()
        self.para1Txt.setText('0.8')
        self.para1Txt.setFixedSize(50,20)
        para2 = QLabel('β')
        para2.setFixedSize(20, 20)
        self.para2Txt = QLineEdit()
        self.para2Txt.setText('0')
        self.para2Txt.setFixedSize(50,20)
        # para1Txt.setFixedWidth(100)
        self.paraSettingLayout = QGridLayout(self.main_widget)
        self.paraSettingLayout.addWidget(para1,0,0)
        self.paraSettingLayout.addWidget(self.para1Txt,0,1)
        self.paraSettingLayout.addWidget(para2, 1, 0)
        self.paraSettingLayout.addWidget(self.para2Txt, 1, 1)
        self.setLayout(self.paraSettingLayout)
        # self.setGeometry(300, 300, 300, 300)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

class classicWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.currentPath = sys.path[0]  # 程序运行的路径

        self.setWindowIcon(QIcon(r'G:\myRepostory\graduateProject\logo\xjtu.jpg'))
        self.main_widget = QWidget(self)
        para1 = QLabel('α')
        para1.setFixedSize(20,20)
        self.para1Txt = QLineEdit()
        self.para1Txt.setText('0')
        self.para1Txt.setFixedSize(50,20)
        para2 = QLabel('β')
        para2.setFixedSize(20, 20)
        self.para2Txt = QLineEdit()
        self.para2Txt.setText('0')
        self.para2Txt.setFixedSize(50, 20)
        # para1Txt.setFixedWidth(100)
        self.paraSettingLayout = QGridLayout(self.main_widget)
        self.paraSettingLayout.addWidget(para1,0,0)
        self.paraSettingLayout.addWidget(self.para1Txt,0,1)
        self.paraSettingLayout.addWidget(para1, 1, 0)
        self.paraSettingLayout.addWidget(self.para1Txt, 1, 1)
        self.setLayout(self.paraSettingLayout)
        # self.setGeometry(300, 300, 300, 300)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)


class changeFPParaWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.currentPath = sys.path[0]  # 程序运行的路径

        self.setWindowIcon(QIcon(r'G:\myRepostory\graduateProject\logo\xjtu.jpg'))
        self.main_widget = QWidget(self)
        self.para1 = QLabel('minSup')
        self.para1.setFixedSize(50,20)
        self.para1Txt = QLineEdit()
        self.para1Txt.setText('0')
        self.para1Txt.setFixedSize(50,20)
        # para1Txt.setFixedWidth(100)
        self.paraSettingLayout = QGridLayout(self.main_widget)
        self.paraSettingLayout.addWidget(self.para1,0,0)
        self.paraSettingLayout.addWidget(self.para1Txt,0,1)
        self.setLayout(self.paraSettingLayout)
        # self.setGeometry(300, 300, 300, 300)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

class changeGAParaWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.currentPath = sys.path[0]  # 程序运行的路径

        self.setWindowIcon(QIcon(r'G:\myRepostory\graduateProject\logo\xjtu.jpg'))
        self.main_widget = QWidget(self)
        self.para0 = QLabel('种群个体个数')
        self.para0.setFixedSize(50, 20)
        self.para0Txt = QLineEdit()
        self.para0Txt.setText('10')
        self.para0Txt.setFixedSize(50, 20)
        self.para1 = QLabel('交叉率')
        self.para1.setFixedSize(50,20)
        self.para1Txt = QLineEdit()
        self.para1Txt.setText('0.8')
        self.para1Txt.setFixedSize(50,20)
        self.para2 = QLabel('变异率')
        self.para2.setFixedSize(50, 20)
        self.para2Txt = QLineEdit()
        self.para2Txt.setText('0.05')
        self.para2Txt.setFixedSize(50, 20)
        self.para3 = QLabel('迭代次数')
        self.para3.setFixedSize(50, 20)
        self.para3Txt = QLineEdit()
        self.para3Txt.setText('100')
        self.para3Txt.setFixedSize(50, 20)
        # para1Txt.setFixedWidth(100)
        self.paraSettingLayout = QGridLayout(self.main_widget)
        self.paraSettingLayout.addWidget(self.para0, 0, 0)
        self.paraSettingLayout.addWidget(self.para0Txt, 0, 1)
        self.paraSettingLayout.addWidget(self.para1,1,0)
        self.paraSettingLayout.addWidget(self.para1Txt,1,1)
        self.paraSettingLayout.addWidget(self.para2, 2, 0)
        self.paraSettingLayout.addWidget(self.para2Txt, 2, 1)
        self.paraSettingLayout.addWidget(self.para3, 3, 0)
        self.paraSettingLayout.addWidget(self.para3Txt, 3, 1)
        self.setLayout(self.paraSettingLayout)
        # self.setGeometry(300, 300, 300, 300)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

class changeSAParaWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.currentPath = sys.path[0]  # 程序运行的路径

        self.setWindowIcon(QIcon(r'G:\myRepostory\graduateProject\logo\xjtu.jpg'))
        self.main_widget = QWidget(self)
        self.para1 = QLabel('初始温度')
        self.para1.setFixedSize(50,20)
        self.para1Txt = QLineEdit()
        self.para1Txt.setText('100')
        self.para1Txt.setFixedSize(50,20)
        self.para2 = QLabel('降温系数α')
        self.para2.setFixedSize(50, 20)
        self.para2Txt = QLineEdit()
        self.para2Txt.setText('0.9')
        self.para2Txt.setFixedSize(50, 20)
        self.para3 = QLabel('内迭代次数')
        self.para3.setFixedSize(50, 20)
        self.para3Txt = QLineEdit()
        self.para3Txt.setText('10')
        self.para3Txt.setFixedSize(50, 20)
        # para1Txt.setFixedWidth(100)
        self.paraSettingLayout = QGridLayout(self.main_widget)
        self.paraSettingLayout.addWidget(self.para1,0,0)
        self.paraSettingLayout.addWidget(self.para1Txt,0,1)
        self.paraSettingLayout.addWidget(self.para2, 1, 0)
        self.paraSettingLayout.addWidget(self.para2Txt, 1, 1)
        self.paraSettingLayout.addWidget(self.para3, 2, 0)
        self.paraSettingLayout.addWidget(self.para3Txt, 2, 1)
        self.setLayout(self.paraSettingLayout)
        # self.setGeometry(300, 300, 300, 300)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

class changeACOParaWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.currentPath = sys.path[0]  # 程序运行的路径

        self.setWindowIcon(QIcon(r'G:\myRepostory\graduateProject\logo\xjtu.jpg'))
        self.main_widget = QWidget(self)
        self.para1 = QLabel('蚂蚁个数')
        self.para1.setFixedSize(50,20)
        self.para1Txt = QLineEdit()
        self.para1Txt.setText('10')
        self.para1Txt.setFixedSize(50,20)
        self.para2 = QLabel('迭代次数')
        self.para2.setFixedSize(50, 20)
        self.para2Txt = QLineEdit()
        self.para2Txt.setText('100')
        self.para2Txt.setFixedSize(50, 20)
        self.para3 = QLabel('p0')
        self.para3.setFixedSize(50, 20)
        self.para3Txt = QLineEdit()
        self.para3Txt.setText('0.8')
        self.para3Txt.setFixedSize(50, 20)
        self.para4 = QLabel('p0')
        self.para4.setFixedSize(50, 20)
        self.para4Txt = QLineEdit()
        self.para4Txt.setText('0.8')
        self.para4Txt.setFixedSize(50, 20)
        self.para5 = QLabel('ρ')
        self.para5.setFixedSize(50, 20)
        self.para5Txt = QLineEdit()
        self.para5Txt.setText('0.8')
        self.para5Txt.setFixedSize(50, 20)
        self.para6 = QLabel('Q')
        self.para6.setFixedSize(50, 20)
        self.para6Txt = QLineEdit()
        self.para6Txt.setText('0.8')
        self.para6Txt.setFixedSize(50, 20)
        # para1Txt.setFixedWidth(100)
        self.paraSettingLayout = QGridLayout(self.main_widget)
        self.paraSettingLayout.addWidget(self.para1,0,0)
        self.paraSettingLayout.addWidget(self.para1Txt,0,1)
        self.paraSettingLayout.addWidget(self.para2, 1, 0)
        self.paraSettingLayout.addWidget(self.para2Txt, 1, 1)
        self.paraSettingLayout.addWidget(self.para3, 2, 0)
        self.paraSettingLayout.addWidget(self.para3Txt, 2, 1)
        self.paraSettingLayout.addWidget(self.para4, 3, 0)
        self.paraSettingLayout.addWidget(self.para4Txt, 3, 1)
        self.paraSettingLayout.addWidget(self.para5, 4, 0)
        self.paraSettingLayout.addWidget(self.para5Txt, 4, 1)
        self.paraSettingLayout.addWidget(self.para6, 5, 0)
        self.paraSettingLayout.addWidget(self.para6Txt, 5, 1)
        self.setLayout(self.paraSettingLayout)
        # self.setGeometry(300, 300, 300, 300)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)